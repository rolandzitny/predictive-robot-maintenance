import numpy as np
from src.utils.utils import temp_detector_data
from src.utils.utils import temp_classifier_data
from src.utils.utils import temp_detector_data2
from src.predictive_maintenance.anomaly_detection import AnomalyDetector
from src.predictive_maintenance.anomaly_classification import AnomalyClassifier


def anomaly_detector_test(load, x_train, y_train, x_test, y_test):
    print('-----------------------| ANOMALY DETECTOR |-----------------------')
    # Prepare model
    input_shape = (288, 6)
    anomaly_detector = AnomalyDetector()
    anomaly_detector.build_model(input_shape=input_shape)

    if load:
        anomaly_detector.load_model()
    else:
        anomaly_detector.train_model(x_train=x_train,
                                     y_train=y_train,
                                     epochs=30,
                                     batch_size=128,
                                     shuffle=True,
                                     patience=False,
                                     verbose=True)
        anomaly_detector.save_model()

    anomaly_detector.show_summary()
    threshold = anomaly_detector.estimate_threshold(x_train=x_train, y_train=y_train)
    print('Threshold:  ', threshold)
    anomaly_detector.set_threshold(threshold)
    evaluation_result = anomaly_detector.evaluate(x_test=x_test, y_test=y_test)

    print('mse_normal: ', evaluation_result['mse_normal'])
    print('mse_anomal: ', evaluation_result['mse_anomal'])
    print('aucroc    : ', evaluation_result['aucroc'])
    print('aucpr     : ', evaluation_result['aucpr'])

    plot_data = np.array([x_train[0], x_train[100], x_test[0], x_test[100]])
    anomaly_detector.reconstruction_plot_evaluation(plot_data, plots_number=4)


def anomaly_classifier_test(load, x_train, y_train, x_test, y_test):
    print('-----------------------|  ANOMALY CLASSIFIER |--------------------')
    anomaly_classifier = AnomalyClassifier()
    anomaly_classifier.build_model(input_shape=(288, 6), classes_num=5)

    if load:
        anomaly_classifier.load_model()
    else:
        anomaly_classifier.train_model(x_train=x_train,
                                       y_train=y_train,
                                       epochs=50,
                                       batch_size=64,
                                       shuffle=True,
                                       patience=False,
                                       verbose=True)

        anomaly_classifier.save_model()

    anomaly_classifier.show_summary()

    evaluation_result = anomaly_classifier.evaluate(x_test=x_test, y_test=y_test)

    print('overall accuracy: ', evaluation_result['accuracy'])
    for anomaly_class in evaluation_result['classes']:
        print("-------------------------------")
        print('class ', (anomaly_class['class']))
        print('precision: ', anomaly_class['precision'])
        print('recall   : ', anomaly_class['recall'])
        print('f1-score : ', anomaly_class['f1-score'])


def main():
    dataset_path = 'stage2_arm1_9-10.csv'
    x_train, y_train, x_test, y_test, y_test_classes = temp_detector_data2(dataset_path)

    anomaly_detector_test(False, x_train, y_train, x_test, y_test)
    anomaly_classifier_test(True, x_test, y_test_classes, x_test, y_test_classes)


if __name__ == '__main__':
    main()



