"""
Functions for showcase.
"""

import numpy as np
import configparser
import matplotlib.pyplot as plt
from src.data_handling.preprocessing import create_signal_windows
from src.predictive_maintenance.anomaly_detector import AnomalyDetector
from src.predictive_maintenance.anomaly_classifier import AnomalyClassifier
from src.predictive_maintenance.segment_comparator import SegmentComparator
from src.predictive_maintenance.maintenance_planner import get_maintenance_plan


def test_segmentation():
    """
    Test segmentation. Created segments are plotted.
    """
    data = np.load('data/testing/segmentation_test_data.npz')
    data = np.array(data['data'])

    config = configparser.ConfigParser()
    config.read('config/configurations.ini')
    pause_duration = int(config.get('segment-comparator', 'min_pause_duration'))
    similarity_threshold = float(config.get('segment-comparator', 'similarity_threshold'))
    segment_comparator = SegmentComparator(min_pause_duration=pause_duration, similarity_threshold=similarity_threshold)
    motion_indices = segment_comparator.get_motions_indices(data)

    plt.plot(data)
    for indc in motion_indices:
        plt.axvline(indc[0], c='r')
        plt.axvline(indc[1], c='r')
    plt.show()


def test_diagnose_robot_health():
    """
    Test diagnose robot health on csv files.
    Data contain arm with robot belt wear and repaired arm.
    This function is just for showcase.

    :return: Diagnosis results as dictionary.
    """

    diagnosis = {
        'anomalies_number': None,
        'anomalies_labels': None,
        'anomalies_labels_count': None,
        'number_of_segments': None,
        'segments_comparison': None
    }

    config = configparser.ConfigParser()
    config.read('config/configurations.ini')
    pause_duration = int(config.get('segment-comparator', 'min_pause_duration'))
    similarity_threshold = float(config.get('segment-comparator', 'similarity_threshold'))

    segment_comparator = SegmentComparator(min_pause_duration=pause_duration, similarity_threshold=similarity_threshold)
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model()
    anomaly_classifier = AnomalyClassifier()
    anomaly_classifier.load_model()

    data = np.load('data/testing/hour_diagnosis_test_data.npz')
    normal_data = data['normal_data']
    abnormal_data = data['abnormal_data']

    try:
        robot_energy_consumption_windows = create_signal_windows(abnormal_data, 288, 0)
        detection_results = anomaly_detector.detect(robot_energy_consumption_windows)
        abnormal_windows = robot_energy_consumption_windows[detection_results == 1]
        classification_results = np.argmax(anomaly_classifier.classify(abnormal_windows), axis=1)
        unique_labels, label_counts = np.unique(classification_results, return_counts=True)
        diagnosis['anomalies_number'] = len(abnormal_windows)
        diagnosis['anomalies_labels'] = np.array(unique_labels)
        diagnosis['anomalies_labels_count'] = np.array(label_counts)
    except:
        pass

    try:
        num_of_segments = len(segment_comparator.get_motions(abnormal_data))
        diagnosis['number_of_segments'] = num_of_segments
        segments_comparison = segment_comparator.compare_signals(abnormal_data, normal_data)
        diagnosis['segments_comparison'] = segments_comparison
    except:
        pass

    return diagnosis


if __name__ == '__main__':
    diagnose_res = test_diagnose_robot_health()
    print("Diagnose results:")
    for key, value in diagnose_res.items():
        print(key, ':', value)

    maintenance_plan = get_maintenance_plan(diagnose_res)
    print("")
    print("Maintenance plan:")
    print(maintenance_plan)
