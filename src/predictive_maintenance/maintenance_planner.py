"""
The maintenance planner serves for the interpretation of the obtained analysis results, above which the severity
levels of the individual results are determined and these subsequently serve to determine the time until the necessary
maintenance, the location of the necessary maintenance and the type of maintenance.
"""


def set_anomaly_severity(anomaly_value):
    """
    Set severities to anomaly results.

    :param anomaly_value: Number of anomalies per one action of robot.
    :return: Severity level.
    """
    if anomaly_value < 0.1:
        return 'None'
    elif anomaly_value < 0.5:
        return 'Low'
    elif anomaly_value < 1.0:
        return 'Medium'
    elif anomaly_value >= 1.0:
        return 'High'


def set_statistics_severity(stat_value):
    """
    Set severities to statistical results.

    :param stat_value: Percentage difference.
    :return: Severity level.
    """
    stat_value = abs(stat_value)
    if stat_value < 0.1:
        return 'None'
    elif stat_value < 0.3:
        return 'Low'
    elif stat_value < 0.5:
        return "Medium"
    elif stat_value >= 0.5:
        return 'High'


def get_severity_levels(diag):
    """
    Set severity levels to dictionary of diagnostic results.

    :param diag: Diagnostic results.
    :return: Dictionary of severity levels.
    """
    severity_results = {
        'anomalies': None,
        'gearbox': None,
        'belts': None,
        'rms': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'std': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'max': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'min': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'ptp': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'cor': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'act': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'mob': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'com': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None},
        'psd': {'signal': None, 'J1': None, 'J2': None, 'J3': None, 'J4': None, 'J5': None, 'J6': None}
    }

    anomalies_number = diag['anomalies_number']
    belt_anomalies_number = diag['anomalies_labels_count'][1]
    fake_anomalies_number = diag['anomalies_labels_count'][0]
    number_of_segments = diag['number_of_segments']
    anomalies_per_action = (anomalies_number - fake_anomalies_number)/number_of_segments
    belt_anomalies_per_action = belt_anomalies_number / number_of_segments

    severity_results['anomalies'] = set_anomaly_severity(anomalies_per_action)
    severity_results['belts'] = set_anomaly_severity(belt_anomalies_per_action)

    severity_results['rms']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['rms']['avg'])
    severity_results['rms']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['rms']['avg'])
    severity_results['rms']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['rms']['avg'])
    severity_results['rms']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['rms']['avg'])
    severity_results['rms']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['rms']['avg'])
    severity_results['rms']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['rms']['avg'])
    severity_results['rms']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['rms']['avg'])

    severity_results['std']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['std']['avg'])
    severity_results['std']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['std']['avg'])
    severity_results['std']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['std']['avg'])
    severity_results['std']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['std']['avg'])
    severity_results['std']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['std']['avg'])
    severity_results['std']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['std']['avg'])
    severity_results['std']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['std']['avg'])

    severity_results['max']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['max']['avg'])
    severity_results['max']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['max']['avg'])
    severity_results['max']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['max']['avg'])
    severity_results['max']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['max']['avg'])
    severity_results['max']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['max']['avg'])
    severity_results['max']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['max']['avg'])
    severity_results['max']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['max']['avg'])

    severity_results['min']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['min']['avg'])
    severity_results['min']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['min']['avg'])
    severity_results['min']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['min']['avg'])
    severity_results['min']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['min']['avg'])
    severity_results['min']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['min']['avg'])
    severity_results['min']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['min']['avg'])
    severity_results['min']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['min']['avg'])

    severity_results['ptp']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['ptp']['avg'])
    severity_results['ptp']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['ptp']['avg'])
    severity_results['ptp']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['ptp']['avg'])
    severity_results['ptp']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['ptp']['avg'])
    severity_results['ptp']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['ptp']['avg'])
    severity_results['ptp']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['ptp']['avg'])
    severity_results['ptp']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['ptp']['avg'])

    severity_results['cor']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['cor']['avg'])
    severity_results['cor']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['cor']['avg'])
    severity_results['cor']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['cor']['avg'])
    severity_results['cor']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['cor']['avg'])
    severity_results['cor']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['cor']['avg'])
    severity_results['cor']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['cor']['avg'])
    severity_results['cor']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['cor']['avg'])

    severity_results['act']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['act']['avg'])
    severity_results['act']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['act']['avg'])
    severity_results['act']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['act']['avg'])
    severity_results['act']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['act']['avg'])
    severity_results['act']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['act']['avg'])
    severity_results['act']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['act']['avg'])
    severity_results['act']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['act']['avg'])

    severity_results['mob']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['mob']['avg'])
    severity_results['mob']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['mob']['avg'])
    severity_results['mob']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['mob']['avg'])
    severity_results['mob']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['mob']['avg'])
    severity_results['mob']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['mob']['avg'])
    severity_results['mob']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['mob']['avg'])
    severity_results['mob']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['mob']['avg'])

    severity_results['com']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['com']['avg'])
    severity_results['com']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['com']['avg'])
    severity_results['com']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['com']['avg'])
    severity_results['com']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['com']['avg'])
    severity_results['com']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['com']['avg'])
    severity_results['com']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['com']['avg'])
    severity_results['com']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['com']['avg'])

    severity_results['psd']['signal'] = set_statistics_severity(diag['segments_comparison']['signal']['psd']['avg'])
    severity_results['psd']['J1'] = set_statistics_severity(diag['segments_comparison']['J1']['psd']['avg'])
    severity_results['psd']['J2'] = set_statistics_severity(diag['segments_comparison']['J2']['psd']['avg'])
    severity_results['psd']['J3'] = set_statistics_severity(diag['segments_comparison']['J3']['psd']['avg'])
    severity_results['psd']['J4'] = set_statistics_severity(diag['segments_comparison']['J4']['psd']['avg'])
    severity_results['psd']['J5'] = set_statistics_severity(diag['segments_comparison']['J5']['psd']['avg'])
    severity_results['psd']['J6'] = set_statistics_severity(diag['segments_comparison']['J6']['psd']['avg'])

    return severity_results


def get_highest_severity(severity_results):
    """
    Find the highest level of severity, its count and locations.

    :param severity_results: Dictionary of severity results.
    :return: Highest severity, its count, locations or type.
    """
    severity_order = {'Low': 1, 'Medium': 2, 'High': 3}
    highest_severity = None
    num_occurrences = 0
    highest_severity_locations = []
    for key, value in severity_results.items():
        if isinstance(value, dict):
            sub_highest_severity, sub_num_occurrences, sub_locations = get_highest_severity(value)
            if sub_highest_severity is not None and (highest_severity is None or severity_order[sub_highest_severity] >
                                                     severity_order[highest_severity]):
                highest_severity = sub_highest_severity
                num_occurrences = sub_num_occurrences
                highest_severity_locations = sub_locations
            elif sub_highest_severity is not None and severity_order[sub_highest_severity] == \
                    severity_order[highest_severity]:
                num_occurrences += sub_num_occurrences
                highest_severity_locations += sub_locations
        else:
            if isinstance(value, str) and value in severity_order:
                if highest_severity is None or severity_order[value] > severity_order[highest_severity]:
                    highest_severity = value
                    num_occurrences = 1
                    highest_severity_locations = [key]
                elif severity_order[value] == severity_order[highest_severity]:
                    num_occurrences += 1
                    highest_severity_locations.append(key)
    return highest_severity, num_occurrences, highest_severity_locations


def get_maintenance_plan(diagnosis_results):
    """
    Return maintenance plan as type of the highest severity level, its count and locations with type of maintenance.

    :param diagnosis_results: Dictionary of diagnostic results.
    :return: Maintenance plan.
    """
    severity_results = get_severity_levels(diagnosis_results)
    print("")
    print('Severity results:')
    print(severity_results)
    maintenance_plan = get_highest_severity(severity_results)
    return maintenance_plan
