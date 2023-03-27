"""
AnomalyDetector is class for building and training a convolutional autoencoder model to detect anomalies in time
series data. The model is trained to reconstruct input windows and the reconstruction error is used as a measure
of anomaly. The model is targeted at data obtained from the operation of the Mitsubishi MELFA robotic arm, where energy
consumption is recorded on 6 arm joints. This energy consumption is then divided into per-second intervals,
which corresponds to approximately 288 records of 6 values (288, 6).
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Concatenate
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.threshold = None

    def build_model(self, input_shape=(288, 6)):
        """
        Creates a convolutional autoencoder model for anomaly detection.
        Use the show_summary() method to display the model architecture.

        :param input_shape: Shape of the window, for example (288, 6). Be careful, not every shape is compatible.
        You can check compatibility by showing the architecture of the model and maintaining the same shape of
        output as input.
        """
        input_layer = Input(shape=input_shape)

        # Encoder
        encoder = Conv1D(filters=512, kernel_size=13, activation='relu', padding='same')(input_layer)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(rate=0.2)(encoder)
        encoder = MaxPooling1D(pool_size=2)(encoder)

        encoder = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(rate=0.2)(encoder)
        # Residual connection 1
        residual_1 = encoder
        encoder = MaxPooling1D(pool_size=2)(encoder)

        encoder = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(rate=0.2)(encoder)
        # Residual connection 2
        residual_2 = encoder
        encoder = MaxPooling1D(pool_size=2)(encoder)

        encoder = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(rate=0.2)(encoder)
        # Residual connection 3
        residual_3 = encoder
        encoder = MaxPooling1D(pool_size=2)(encoder)

        # Decoder
        decoder = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(encoder)  # Do not forget encoder
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(rate=0.2)(decoder)
        decoder = UpSampling1D(size=2)(decoder)

        # Connect residual connection 3
        decoder = Concatenate()([decoder, residual_3])
        decoder = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(rate=0.2)(decoder)
        decoder = UpSampling1D(size=2)(decoder)

        # Connect residual connection 2
        decoder = Concatenate()([decoder, residual_2])
        decoder = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(rate=0.2)(decoder)
        decoder = UpSampling1D(size=2)(decoder)

        # Connect residual connection 1
        decoder = Concatenate()([decoder, residual_1])
        decoder = Conv1D(filters=512, kernel_size=13, activation='relu', padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(rate=0.2)(decoder)
        decoder = UpSampling1D(size=2)(decoder)

        decoder = Conv1D(filters=input_shape[1], kernel_size=5, activation='linear', padding='same')(decoder)

        # Compile
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        self.model = autoencoder

    def train_model(self, x_train, y_train, epochs, batch_size, shuffle, patience, verbose):
        """
        Train the built model of the convolutional autoencoder.

        :param x_train: Training windows X.
        :param y_train: Training windows Y. Same as x_train because the autoencoder is learning to reconstruct inputs.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size to use during training.
        :param shuffle: True/False, determines whether to shuffle windows before every epoch.
        :param patience: False/Integer. False will remove EarlyStopping, an integer sets its patience.
        :param verbose: True/False, whether to display learning status or not.
        """
        fit_callbacks = []
        if patience:
            fit_callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min')]

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_split=0.1,
                       callbacks=fit_callbacks, verbose=verbose)

    def detect(self, signal_windows):
        """
        Use the trained model to detect anomalies by calculating the reconstruction error of each window and comparing
        it to a threshold. Any window with a reconstruction error above the threshold is marked as an anomaly (1),
        while any window with a reconstruction error below the threshold is marked as normal (0).

        :param signal_windows: Windows in the same shape as the training windows.
        :return: An array of ones and zeros, representing whether each window is an anomaly (1) or normal (0).
        """
        reconstructed_windows = self.model.predict(signal_windows, verbose=False)
        mse_loss = np.mean(np.square(signal_windows - reconstructed_windows), axis=(1, 2))
        detection_results = np.where(mse_loss > self.threshold, 1, 0)
        return detection_results

    def evaluate(self, x_test, y_test):
        """
        The method is used to evaluate the performance of a trained model. The results can be used to compare newly
        trained models and decide whether to make changes to the monitoring system. Best practise is when x_test 50%
        anomalies.

        :param x_test: Testing windows.
        :param y_test: True labels of windows, an array of ones and zeros representing anomalies/normal windows.
        :return: A dictionary containing the reconstruction error on normal windows,
        reconstruction error on anomalous windows, AUCROC, and AUCPR.
        """
        x_test_reconstructed = self.model.predict(x_test, verbose=False)
        mse = np.mean(np.square(x_test - x_test_reconstructed), axis=(1, 2))
        detection_results = self.detect(x_test)

        num_0 = 0
        num_1 = 0
        for i in range(len(y_test)):
            if y_test[i] == 0:
                if detection_results[i] == y_test[i]:
                    num_0 += 1

            if y_test[i] == 1:
                if detection_results[i] == y_test[i]:
                    num_1 += 1

        #TODO
        print('total: ', len(y_test))
        print('0: ', num_0)
        print('1: ', num_1)

        # mean reconstruction error on anomal and normal windows
        anomal_windows_mse = np.mean(mse[y_test == 1])
        normal_windows_mse = np.mean(mse[y_test == 0])

        # ROC curve
        fpr, tpr, threshold_roc = roc_curve(y_test, detection_results)
        roc_auc = auc(fpr, tpr)

        # PR curve
        precision, recall, threshold_pr = precision_recall_curve(y_test, detection_results)
        auc_pr = auc(recall, precision)

        result_dictionary = {
            'mse_normal': normal_windows_mse,
            'mse_anomal': anomal_windows_mse,
            'aucroc': roc_auc,
            'aucpr': auc_pr
        }

        return result_dictionary

    def reconstruction_plot_evaluation(self, signal_windows, plots_number):
        """
        Display 'plot_number' of figures with  2 graphs.
        First graph represent true signal and second graph represents reconstructed signal.

        :param signal_windows: True signal windows.
        :param plots_number: Number of figures.
        """
        reconstructed_signals = self.model.predict(signal_windows, verbose=False)
        for i in range(plots_number):
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
            ax1.plot(signal_windows[i])
            ax1.set_title('Real signal ' + str(i + 1))
            ax2.plot(reconstructed_signals[i])
            ax2.set_title('Reconstructed signal ' + str(i + 1))
            fig.subplots_adjust(hspace=0.5)
            plt.show()

    def save_model(self, name=None, folder='models/'):
        """
        Save trained model.

        :param name: The name of the folder to be saved.
        If none is provided, the default name 'anomaly_detector' will be used.
        :param folder: The destination folder for the saved model, with a '/' character.
        If none is provided, the default folder 'models/' will be used.
        """
        if name is None:
            filepath = folder + 'anomaly_detector'
        else:
            filepath = folder + name

        self.model.save(filepath)

    def load_model(self, name=None, folder='models/'):
        """
        Load trained model.

        :param name: The name of the folder to be saved.
        If none is provided, the default name 'anomaly_detector' will be used.
        :param folder: The destination folder for the saved model, with a '/' character.
        If none is provided, the default folder 'models/' will be used.
        """
        if name is None:
            filepath = folder + 'anomaly_detector'
        else:
            filepath = folder + name

        self.model = keras.models.load_model(filepath)

    def estimate_threshold(self, x_train, y_train, threshold_rate=1.0):
        """
        Estimate the threshold for the current model.
        This threshold is chosen as the maximal reconstruction error on a given dataset.
        Note that the autoencoder is used for reconstruction, so the x_train and y_train parameters should be the same.

        :param x_train: The windows for reconstruction.
        :param y_train: The windows that represent the reconstruction output.
        :param threshold_rate: The rate of threshold to use,
        where for example 0.8 takes only 80% of the computed threshold.
        :return: The estimated threshold value.
        """
        x_train_reconstructed = self.model.predict(x_train)
        mse_loss = np.mean(np.square(y_train - x_train_reconstructed), axis=(1, 2))
        threshold = threshold_rate * np.max(mse_loss)
        return threshold

    def get_threshold(self):
        """
        Threshold getter.

        :return: Threshold.
        """
        return self.threshold

    def set_threshold(self, threshold):
        """
        Threshold setter.

        :param threshold: Threshold value.
        """
        self.threshold = threshold

    def show_summary(self):
        """
        Display architecture of current model.
        """
        self.model.summary()
