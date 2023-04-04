"""
AnomalyClassifier is an implementation of a Convolutional Neural Network (CNN) for classifying time series signals
into different anomaly classes. The network has several layers of convolution, pooling, batch normalization, dropout,
and fully connected layers. It takes in windows of time series signals as input and outputs a softmax probability
distribution over the different anomaly classes.
"""

import keras
import numpy as np
from keras import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, Dropout, MaxPooling1D, BatchNormalization, Flatten, Dense
from sklearn.metrics import accuracy_score, classification_report


class AnomalyClassifier:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape=(288, 6), classes_num=5):
        """
        Builds the CNN model with the specified input shape and number of classes.

        :param input_shape: Shape of the window, for example (288, 6).
        :param classes_num: Number of classes for classification.
        """
        input_layer = Input(shape=input_shape)

        classifier = Conv1D(filters=32, kernel_size=5, activation='relu')(input_layer)
        classifier = BatchNormalization()(classifier)
        classifier = MaxPooling1D(pool_size=2)(classifier)
        classifier = Dropout(rate=0.2)(classifier)

        classifier = Conv1D(filters=64, kernel_size=5, activation='relu')(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = MaxPooling1D(pool_size=2)(classifier)
        classifier = Dropout(rate=0.2)(classifier)

        classifier = Conv1D(filters=128, kernel_size=5, activation='relu')(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = MaxPooling1D(pool_size=2)(classifier)
        classifier = Dropout(rate=0.2)(classifier)

        classifier = Conv1D(filters=256, kernel_size=5, activation='relu')(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = MaxPooling1D(pool_size=2)(classifier)
        classifier = Dropout(rate=0.2)(classifier)

        classifier = Flatten()(classifier)

        classifier = Dense(units=512, activation='relu')(classifier)
        classifier = Dropout(rate=0.5)(classifier)

        classifier = Dense(units=256, activation='relu')(classifier)
        classifier = Dropout(rate=0.5)(classifier)

        classifier = Dense(units=classes_num, activation='softmax')(classifier)

        classifier = Model(inputs=input_layer, outputs=classifier)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = classifier

    def train_model(self, x_train, y_train, epochs, batch_size, shuffle, patience, verbose):
        """
        Train the built model of the convolutional classifier.

        :param x_train: Training windows X.
        :param y_train: Training labels Y of training windows X.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size to use during training.
        :param shuffle: True/False, determines whether to shuffle windows before every epoch.
        :param patience: False/Integer. False will remove EarlyStopping, an integer sets its patience.
        :param verbose: True/False, whether to display learning status or not.
        """
        fit_callbacks = []
        if patience:
            fit_callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min')]

        y_train_onehot = to_categorical(y_train, num_classes=5)
        self.model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                       validation_split=0.1, callbacks=fit_callbacks, verbose=verbose)

    def classify(self, signal_windows):
        """
        Performs classification on provided signal windows using the trained model.

        :param signal_windows: Windows in the same shape as the training windows.
        :return: An array of classification results.
        """
        classification_results = self.model.predict(signal_windows)
        return classification_results

    def evaluate(self, x_test, y_test):
        """
        Evaluates the trained model's performance on a test set, returning a dictionary of evaluation metrics
        for each class, including precision, recall, and f1-score.

        :param x_test: Testing windows.
        :param y_test: True labels of windows, for example [0,1,2,0,3,4,0,4,3,2,1].
        :return: A dictionary containing the overall accuracy, including list of dictionaries fore every class with
        precision, recall and f1-score.
        """
        x_test_classified = np.argmax(self.classify(x_test), axis=-1)
        report = classification_report(y_test, x_test_classified, output_dict=True)
        num_of_classes = len(np.unique(y_test))

        accuracy = accuracy_score(y_test, x_test_classified)

        dict_classes = []
        for i in range(num_of_classes):
            dict_class = {
                'class': str(i),
                'precision': report[str(i)]['precision'],
                'recall': report[str(i)]['recall'],
                'f1-score': report[str(i)]['f1-score']
            }
            dict_classes.append(dict_class)

        result_dictionary = {
            'accuracy': accuracy,
            'classes': dict_classes
        }

        return result_dictionary

    def save_model(self, name=None, folder='models/'):
        """
        Save trained model.

        :param name: The name of the folder to be saved.
        If none is provided, the default name 'anomaly_classifier' will be used.
        :param folder: The destination folder for the saved model, with a '/' character.
        If none is provided, the default folder 'models/' will be used.
        """
        if name is None:
            filepath = folder + 'anomaly_classifier'
        else:
            filepath = folder + name

        self.model.save(filepath)

    def load_model(self, name=None, folder='models/'):
        """
        Load trained model.

        :param name: The name of the folder to be saved.
        If none is provided, the default name 'anomaly_classifier' will be used.
        :param folder: The destination folder for the saved model, with a '/' character.
        If none is provided, the default folder 'models/' will be used.
        """
        if name is None:
            filepath = folder + 'anomaly_classifier'
        else:
            filepath = folder + name

        self.model = keras.models.load_model(filepath)

    def show_summary(self):
        """
        Display architecture of current model.
        """
        self.model.summary()
