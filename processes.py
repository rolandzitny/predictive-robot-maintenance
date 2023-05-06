"""
Processes of the diagnostics of the monitored devices.
"""

import os
import numpy as np
import configparser
from datetime import date, timedelta
from src.data_handling.preprocessing import create_signal_windows
from src.data_handling.influxdb_handler import InfluxDBHandler
from src.predictive_maintenance.anomaly_detector import AnomalyDetector
from src.predictive_maintenance.anomaly_classifier import AnomalyClassifier
from src.predictive_maintenance.segment_comparator import SegmentComparator
from src.predictive_maintenance.maintenance_planner import get_maintenance_plan


def diagnose_robot_health(robot_arm_tag):
    """
    Diagnose robot health in last hour.

    :param robot_arm_tag: Robot InfluxDB arm tag.
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

    influxdb_handler = InfluxDBHandler()
    segment_comparator = SegmentComparator(min_pause_duration=pause_duration, similarity_threshold=similarity_threshold)
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model()
    anomaly_classifier = AnomalyClassifier()
    anomaly_classifier.load_model()

    try:
        robot_energy_consumption = influxdb_handler.query_energy_consumption_by_hours(hours=1, arm_tag=robot_arm_tag)
    except:
        # remove historical signal
        if os.path.exists('data/historical_signals/' + str(robot_arm_tag) + '.npy'):
            os.remove('data/historical_signals/' + str(robot_arm_tag) + '.npy')
        return diagnosis

    try:
        robot_energy_consumption_windows = create_signal_windows(robot_energy_consumption, 288, 0)
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
        num_of_segments = len(segment_comparator.get_motions(robot_energy_consumption))
        diagnosis['number_of_segments'] = num_of_segments
        historical_energy_consum = np.load('data/historical_signals/' + str(robot_arm_tag) + '.npy')
        segments_comparison = segment_comparator.compare_signals(robot_energy_consumption, historical_energy_consum)
        diagnosis['segments_comparison'] = segments_comparison
        np.save('data/historical_signals/' + str(robot_arm_tag) + '.npy', robot_energy_consumption)
    except:
        pass

    # TODO save results into database
    maintenance_plan = get_maintenance_plan(diagnosis)
    print(maintenance_plan)
    return diagnosis


def diagnose_robot_health_history(robot_arm_tag):
    """
    Diagnose robot health with historical data.

    :param robot_arm_tag: Robot InfluxDB arm tag.
    :return: Diagnosis results as dictionary.
    """

    diagnosis = {
        'day_comparison': None,
        'week1_comparison': None,
        'week2_comparison': None,
        'week3_comparison': None,
        'week4_comparison': None
    }

    config = configparser.ConfigParser()
    config.read('config/configurations.ini')
    pause_duration = int(config.get('segment-comparator', 'min_pause_duration'))
    similarity_threshold = float(config.get('segment-comparator', 'similarity_threshold'))

    influxdb_handler = InfluxDBHandler()
    segment_comparator = SegmentComparator(min_pause_duration=pause_duration, similarity_threshold=similarity_threshold)

    today_date = date.today().strftime('%Y-%m-%d')
    yestr_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    week1_date = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    week2_date = (date.today() - timedelta(days=14)).strftime('%Y-%m-%d')
    week3_date = (date.today() - timedelta(days=21)).strftime('%Y-%m-%d')
    week4_date = (date.today() - timedelta(days=28)).strftime('%Y-%m-%d')

    try:
        today_ec = influxdb_handler.query_energy_consumption_by_day(today_date, robot_arm_tag)
    except:
        return diagnosis

    try:
        yestr_ec = influxdb_handler.query_energy_consumption_by_day(yestr_date, robot_arm_tag)
        diagnosis['day_comparison'] = segment_comparator.compare_signals(today_ec, yestr_ec)
    except:
        pass

    try:
        week1_ec = influxdb_handler.query_energy_consumption_by_day(week1_date, robot_arm_tag)
        diagnosis['week1_comparison'] = segment_comparator.compare_signals(today_ec, week1_ec)
    except:
        pass

    try:
        week2_ec = influxdb_handler.query_energy_consumption_by_day(week2_date, robot_arm_tag)
        diagnosis['week2_comparison'] = segment_comparator.compare_signals(today_ec, week2_ec)
    except:
        pass

    try:
        week3_ec = influxdb_handler.query_energy_consumption_by_day(week3_date, robot_arm_tag)
        diagnosis['week3_comparison'] = segment_comparator.compare_signals(today_ec, week3_ec)
    except:
        pass

    try:
        week4_ec = influxdb_handler.query_energy_consumption_by_day(week4_date, robot_arm_tag)
        diagnosis['week4_comparison'] = segment_comparator.compare_signals(today_ec, week4_ec)
    except:
        pass

    # TODO save results into database
    return diagnosis
