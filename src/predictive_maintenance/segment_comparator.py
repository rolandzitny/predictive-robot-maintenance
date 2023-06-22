"""
SegmentComparator is and implementation of a Hidden Markov Models (HMM) for segmentation time series signals into
different segments of motion. These segments can then be matched together using correlation, which allows
various analysis methods to be performed on the signal segments.
"""

import numpy as np
from hmmlearn import hmm
from scipy.stats import entropy
from scipy.signal import welch
from scipy.signal import correlate
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def compare_joint_standard_deviation(matched_segments):
    """
    Compare standard deviation of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    std_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_std = np.std(s1)
        s2_std = np.std(s2)
        percentage_difference = (s1_std - s2_std) / s2_std
        std_diffs.append(percentage_difference)

    avg_diff = np.mean(std_diffs)
    max_diff = std_diffs[np.abs(std_diffs).argmax()]
    min_diff = std_diffs[np.abs(std_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_standard_deviation(matched_segments):
    """
    Compare standard deviation of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    std_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_standard_deviation([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        std_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(std_diffs)
    max_diff = std_diffs[np.abs(std_diffs).argmax()]
    min_diff = std_diffs[np.abs(std_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_rms(matched_segments):
    """
    Compare RMS of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    rms_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_rms = np.sqrt(np.mean(np.square(s1)))
        s2_rms = np.sqrt(np.mean(np.square(s2)))
        percentage_difference = (s1_rms - s2_rms) / s2_rms
        rms_diffs.append(percentage_difference)

    avg_diff = np.mean(rms_diffs)
    max_diff = rms_diffs[np.abs(rms_diffs).argmax()]
    min_diff = rms_diffs[np.abs(rms_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_rms(matched_segments):
    """
    Compare RMS of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    rms_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_rms([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        rms_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(rms_diffs)
    max_diff = rms_diffs[np.abs(rms_diffs).argmax()]
    min_diff = rms_diffs[np.abs(rms_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_max(matched_segments):
    """
    Compare MAX of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    max_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_max = np.max(s1)
        s2_max = np.max(s2)
        percentage_difference = (s1_max - s2_max) / s2_max
        max_diffs.append(percentage_difference)

    avg_diff = np.mean(max_diffs)
    max_diff = max_diffs[np.abs(max_diffs).argmax()]
    min_diff = max_diffs[np.abs(max_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_max(matched_segments):
    """
    Compare MAX of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    max_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_max([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        max_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(max_diffs)
    max_diff = max_diffs[np.abs(max_diffs).argmax()]
    min_diff = max_diffs[np.abs(max_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_min(matched_segments):
    """
    Compare MIN of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    min_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_min = np.min(s1)
        s2_min = np.min(s2)
        percentage_difference = (s1_min - s2_min) / s2_min
        min_diffs.append(percentage_difference)

    avg_diff = np.mean(min_diffs)
    max_diff = min_diffs[np.abs(min_diffs).argmax()]
    min_diff = min_diffs[np.abs(min_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_min(matched_segments):
    """
    Compare MIN of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    min_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_min([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        min_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(min_diffs)
    max_diff = min_diffs[np.abs(min_diffs).argmax()]
    min_diff = min_diffs[np.abs(min_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_peak_to_peak(matched_segments):
    """
    Compare PeakToPeak of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    peak_to_peak = []
    for (s1, s2, _) in matched_segments:
        s1_peak_to_peak = np.min(s1)
        s2_peak_to_peak = np.min(s2)
        percentage_difference = (s1_peak_to_peak - s2_peak_to_peak) / s2_peak_to_peak
        peak_to_peak.append(percentage_difference)

    avg_diff = np.mean(peak_to_peak)
    max_diff = peak_to_peak[np.abs(peak_to_peak).argmax()]
    min_diff = peak_to_peak[np.abs(peak_to_peak).argmin()]
    return avg_diff, max_diff, min_diff


def compare_peak_to_peak(matched_segments):
    """
    Compare PeakToPeak of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    ptp_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_peak_to_peak([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        ptp_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(ptp_diffs)
    max_diff = ptp_diffs[np.abs(ptp_diffs).argmax()]
    min_diff = ptp_diffs[np.abs(ptp_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_entropy(matched_segments):
    """
    Compare Entropy of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    entropy_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_entropy = entropy(np.abs(s1))
        s2_entropy = entropy(np.abs(s2))
        percentage_difference = (s1_entropy - s2_entropy) / s2_entropy
        entropy_diffs.append(percentage_difference)

    avg_diff = np.mean(entropy_diffs)
    max_diff = entropy_diffs[np.abs(entropy_diffs).argmax()]
    min_diff = entropy_diffs[np.abs(entropy_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_entropy(matched_segments):
    """
    Compare Entropy of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    entropy_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_entropy([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        entropy_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(entropy_diffs)
    max_diff = entropy_diffs[np.abs(entropy_diffs).argmax()]
    min_diff = entropy_diffs[np.abs(entropy_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_correlation(matched_segments):
    """
    Compare cross-correlation of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    similarities = []
    for (s1, s2, _) in matched_segments:
        autocorr = np.correlate(s1, s2, mode='same')
        similarity = autocorr.max() / (np.linalg.norm(s1) * np.linalg.norm(s2))
        similarities.append(similarity)

    avg_diff = np.mean(similarities)
    max_diff = similarities[np.abs(similarities).argmax()]
    min_diff = similarities[np.abs(similarities).argmin()]
    return avg_diff, max_diff, min_diff


def compare_correlation(matched_segments):
    """
    Compare cross-correlation of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    corr_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_correlation([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        corr_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(corr_diffs)
    max_diff = corr_diffs[np.abs(corr_diffs).argmax()]
    min_diff = corr_diffs[np.abs(corr_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def compare_joint_hjorth_parameters(matched_segments):
    """
    Compare Hjorth parameters of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    activity_diffs = []
    mobility_diffs = []
    complexity_diffs = []
    for (s1, s2, _) in matched_segments:
        s1_activity = np.var(s1)
        s1_diff = np.diff(s1)
        s1_mobility = np.sqrt(np.var(s1_diff) / s1_activity)
        s1_diff2 = np.diff(s1_diff)
        s1_complexity = np.sqrt(np.var(s1_diff2) / np.var(s1_diff))

        s2_activity = np.var(s2)
        s2_diff = np.diff(s2)
        s2_mobility = np.sqrt(np.var(s2_diff) / s2_activity)
        s2_diff2 = np.diff(s2_diff)
        s2_complexity = np.sqrt(np.var(s2_diff2) / np.var(s2_diff))

        activity_diff = (s1_activity - s2_activity) / s2_activity
        activity_diffs.append(activity_diff)

        mobility_diff = (s1_mobility - s2_mobility) / s2_mobility
        mobility_diffs.append(mobility_diff)

        complexity_diff = (s1_complexity - s2_complexity) / s2_complexity
        complexity_diffs.append(complexity_diff)

    avg_act_diff = np.mean(activity_diffs)
    max_act_diff = activity_diffs[np.abs(activity_diffs).argmax()]
    min_act_diff = activity_diffs[np.abs(activity_diffs).argmin()]

    avg_mob_diff = np.mean(mobility_diffs)
    max_mob_diff = mobility_diffs[np.abs(mobility_diffs).argmax()]
    min_mob_diff = mobility_diffs[np.abs(mobility_diffs).argmin()]

    avg_com_diff = np.mean(complexity_diffs)
    max_com_diff = complexity_diffs[np.abs(complexity_diffs).argmax()]
    min_com_diff = complexity_diffs[np.abs(complexity_diffs).argmin()]

    return avg_act_diff, avg_mob_diff, avg_com_diff, max_act_diff, max_mob_diff, max_com_diff, min_act_diff, \
           min_mob_diff, min_com_diff


def compare_hjorth_parameters(matched_segments):
    """
    Compare Hjorth parameters of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    activity_diffs = []
    mobility_diffs = []
    complexity_diffs = []
    for (s1, s2, _) in matched_segments:
        act_separate_diffs = []
        mob_separate_diffs = []
        com_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            act_diff, mob_diff, com_diff, _, _, _, _, _, _ = compare_joint_hjorth_parameters([(s1_j, s2_j, _)])
            act_separate_diffs.append(act_diff)
            mob_separate_diffs.append(mob_diff)
            com_separate_diffs.append(com_diff)

        activity_diffs.append(np.mean(act_separate_diffs))
        mobility_diffs.append(np.mean(mob_separate_diffs))
        complexity_diffs.append(np.mean(com_separate_diffs))

    avg_act_diff = np.mean(activity_diffs)
    max_act_diff = activity_diffs[np.abs(activity_diffs).argmax()]
    min_act_diff = activity_diffs[np.abs(activity_diffs).argmin()]

    avg_mob_diff = np.mean(mobility_diffs)
    max_mob_diff = mobility_diffs[np.abs(mobility_diffs).argmax()]
    min_mob_diff = mobility_diffs[np.abs(mobility_diffs).argmin()]

    avg_com_diff = np.mean(complexity_diffs)
    max_com_diff = complexity_diffs[np.abs(complexity_diffs).argmax()]
    min_com_diff = complexity_diffs[np.abs(complexity_diffs).argmin()]

    return avg_act_diff, avg_mob_diff, avg_com_diff, max_act_diff, max_mob_diff, max_com_diff, min_act_diff, \
           min_mob_diff, min_com_diff


def compare_joint_psd(matched_segments):
    """
    Compare PSD of energy consumption on segments from one joint.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    sampling_rate = 1 / 0.0035  # every 3.5 millisecond
    psd_diffs = []
    for (s1, s2, _) in matched_segments:
        f, Pxx1 = welch(s1, fs=sampling_rate, nperseg=round(len(s1) * 0.2))
        f, Pxx2 = welch(s2, fs=sampling_rate, nperseg=round(len(s1) * 0.2))
        psd_diff = np.mean((Pxx1 - Pxx2) ** 2) / np.mean(Pxx2 ** 2)
        psd_diffs.append(psd_diff)

    avg_psd_diff = np.mean(psd_diffs)
    max_psd_diff = psd_diffs[np.abs(psd_diffs).argmax()]
    min_psd_diff = psd_diffs[np.abs(psd_diffs).argmin()]

    return avg_psd_diff, max_psd_diff, min_psd_diff


def compare_psd(matched_segments):
    """
    Compare PSD of energy consumption on segments from all joints.

    :param matched_segments: Segments matched using SegmentComparator.match_motion_segments
    :return: Average, max, min percentage difference between segments from present signal and past signal.
    """
    corr_diffs = []
    for (s1, s2, _) in matched_segments:
        std_separate_diffs = []
        for i in range(s1.shape[1]):
            s1_j = s1[:, i]
            s2_j = s2[:, i]
            avg_diff, _, _ = compare_joint_psd([(s1_j, s2_j, _)])
            std_separate_diffs.append(avg_diff)

        corr_diffs.append(np.mean(std_separate_diffs))

    avg_diff = np.mean(corr_diffs)
    max_diff = corr_diffs[np.abs(corr_diffs).argmax()]
    min_diff = corr_diffs[np.abs(corr_diffs).argmin()]
    return avg_diff, max_diff, min_diff


def get_signal_comparison_results(matched_segments):
    """
    Call comparison functions on signal with multiple axis.
    If number of matched segments is bigger than 0 return comparison results. Else return None.
    """
    if len(matched_segments) > 0:
        rms_avg, rms_max, rms_min = compare_rms(matched_segments)
        std_avg, std_max, std_min = compare_standard_deviation(matched_segments)
        max_avg, max_max, max_min = compare_max(matched_segments)
        min_avg, min_max, min_min = compare_min(matched_segments)
        ptp_avg, ptp_max, ptp_min = compare_peak_to_peak(matched_segments)
        cor_avg, cor_max, cor_min = compare_correlation(matched_segments)
        a_avg, m_avg, c_avg, a_max, m_max, c_max, a_min, m_min, c_min = compare_hjorth_parameters(matched_segments)
        psd_avg, psd_max, psd_min = compare_psd(matched_segments)

        comparison_results = {
            'rms': {'avg': round(rms_avg, 4), 'max': round(rms_max, 4), 'min': round(rms_min, 4)},
            'std': {'avg': round(std_avg, 4), 'max': round(std_max, 4), 'min': round(std_min, 4)},
            'max': {'avg': round(max_avg, 4), 'max': round(max_max, 4), 'min': round(max_min, 4)},
            'min': {'avg': round(min_avg, 4), 'max': round(min_max, 4), 'min': round(min_min, 4)},
            'ptp': {'avg': round(ptp_avg, 4), 'max': round(ptp_max, 4), 'min': round(ptp_min, 4)},
            'cor': {'avg': round(1 - cor_avg, 4), 'max': round(1 - cor_min, 4), 'min': round(1 - cor_max, 4)},
            'act': {'avg': round(a_avg, 4), 'max': round(a_max, 4), 'min': round(a_min, 4)},
            'mob': {'avg': round(m_avg, 4), 'max': round(m_max, 4), 'min': round(m_min, 4)},
            'com': {'avg': round(c_avg, 4), 'max': round(c_max, 4), 'min': round(c_min, 4)},
            'psd': {'avg': round(psd_avg, 4), 'max': round(psd_max, 4), 'min': round(psd_min, 4)}
        }

        return comparison_results
    else:
        return None


def get_joint_comparison_results(matched_segments):
    """
    Call comparison functions on signal with one axis.
    If number of matched segments is bigger than 0 return comparison results. Else return None.
    """
    if len(matched_segments) > 0:
        rms_avg, rms_max, rms_min = compare_joint_rms(matched_segments)
        std_avg, std_max, std_min = compare_joint_standard_deviation(matched_segments)
        max_avg, max_max, max_min = compare_joint_max(matched_segments)
        min_avg, min_max, min_min = compare_joint_min(matched_segments)
        ptp_avg, ptp_max, ptp_min = compare_joint_peak_to_peak(matched_segments)
        cor_avg, cor_max, cor_min = compare_joint_correlation(matched_segments)
        a_avg, m_avg, c_avg, a_max, m_max, c_max, a_min, m_min, c_min = compare_joint_hjorth_parameters(
            matched_segments)
        psd_avg, psd_max, psd_min = compare_joint_psd(matched_segments)

        comparison_results = {
            'rms': {'avg': round(rms_avg, 4), 'max': round(rms_max, 4), 'min': round(rms_min, 4)},
            'std': {'avg': round(std_avg, 4), 'max': round(std_max, 4), 'min': round(std_min, 4)},
            'max': {'avg': round(max_avg, 4), 'max': round(max_max, 4), 'min': round(max_min, 4)},
            'min': {'avg': round(min_avg, 4), 'max': round(min_max, 4), 'min': round(min_min, 4)},
            'ptp': {'avg': round(ptp_avg, 4), 'max': round(ptp_max, 4), 'min': round(ptp_min, 4)},
            'cor': {'avg': round(1 - cor_avg, 4), 'max': round(1 - cor_min, 4), 'min': round(1 - cor_max, 4)},
            'act': {'avg': round(a_avg, 4), 'max': round(a_max, 4), 'min': round(a_min, 4)},
            'mob': {'avg': round(m_avg, 4), 'max': round(m_max, 4), 'min': round(m_min, 4)},
            'com': {'avg': round(c_avg, 4), 'max': round(c_max, 4), 'min': round(c_min, 4)},
            'psd': {'avg': round(psd_avg, 4), 'max': round(psd_max, 4), 'min': round(psd_min, 4)}
        }

        return comparison_results
    else:
        return None


class SegmentComparator:
    def __init__(self, min_pause_duration, similarity_threshold):
        """
        SegmentComparator initialization

        :param min_pause_duration: Minimal pause duration between segments in number of samples, e.g. 96 samples.
        :param similarity_threshold: Threshold of similarity for matching segments together, e.g. 0.8 for 80%.
        """
        self.min_pause_duration = min_pause_duration
        self.similarity_threshold = similarity_threshold

    def get_joint_motions_indices(self, joint_signal):
        """
        Using HMM find segment of motion in joint signal. This method is used only on signal with one axis.

        :param joint_signal: Joint signal with shape (samples,).
        :return: Segments indices [(starting index, stopping index), ...].
        """
        np.random.seed(42)  # to ensure the stability of the solution
        signal_squared = np.power(joint_signal, 2)
        signal_diff = np.diff(signal_squared)

        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=3000)
        model.fit(signal_diff.reshape(-1, 1))
        hidden_states = model.predict(signal_diff.reshape(-1, 1))

        pause_to_movement = np.where(np.diff(hidden_states) == 1)[0] + 1
        movement_to_pause = np.where(np.diff(hidden_states) == -1)[0]
        segment_indices = list(zip(pause_to_movement, movement_to_pause))
        # filter segments for there are none with negative length
        filtered_segments = [segment for segment in segment_indices if segment[0] <= segment[1]]

        # merge segments together is pause between them is lower than self.min_pause_duration
        merged_segments = []
        if len(filtered_segments) > 0:
            current_segment = filtered_segments[0]
            for next_segment in filtered_segments[1:]:
                if next_segment[0] - current_segment[1] < self.min_pause_duration:
                    current_segment = (current_segment[0], next_segment[1])
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            merged_segments.append(current_segment)

        segment_indices = merged_segments if len(merged_segments) > 0 else []

        filtered_segment_indices = [segment for segment in segment_indices if (segment[1] - segment[0]) > 10]
        del model  # delete model in case of bug of keeping trained model
        return filtered_segment_indices

    def get_joint_motions(self, joint_signal):
        """
        Use self.get_joint_motions_indices to get segments as signal and not as indices. Signal must just one axis.

        :param joint_signal: Joint signal with shape (samples,).
        :return: Segments [seg1, seg2, seg3, ... ].
        """
        segment_indices = self.get_joint_motions_indices(joint_signal)
        joint_motion_segments = []

        for start_idx, end_idx in segment_indices:
            signal_segment = joint_signal[start_idx:end_idx]
            joint_motion_segments.append(signal_segment)

        return joint_motion_segments

    def get_motions_indices(self, signal):
        """
        Using HMM find segment of motion in signal. This method is used on signal with multiple axis.

        :param signal: Signal with shape (samples, variables).
        :return: Segments indices [(starting index, stopping index), ...].
        """
        np.random.seed(42)
        padded_signal = np.pad(signal, (2, 2), mode='reflect')
        signal_diff = np.diff(padded_signal)

        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=3000)
        model.fit(signal_diff)

        hidden_states = model.predict(signal_diff)
        pause_to_movement = np.where(np.diff(hidden_states) == 1)[0] + 1
        movement_to_pause = np.where(np.diff(hidden_states) == -1)[0]
        segment_indices = list(zip(pause_to_movement, movement_to_pause))
        filtered_segments = [segment for segment in segment_indices if segment[0] <= segment[1]]

        # merge segments together is pause between them is lower than self.min_pause_duration
        merged_segments = []
        if len(filtered_segments) > 0:
            current_segment = filtered_segments[0]
            for next_segment in filtered_segments[1:]:
                if next_segment[0] - current_segment[1] < self.min_pause_duration:
                    current_segment = (current_segment[0], next_segment[1])
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            merged_segments.append(current_segment)

        segment_indices = merged_segments if len(merged_segments) > 0 else []

        filtered_segment_indices = [segment for segment in segment_indices if (segment[1] - segment[0]) > 10]
        del model  # delete model in case of bug of keeping trained model
        return filtered_segment_indices

    def get_motions(self, signal):
        """
        Use self.get_joint_motions_indices to get segments as signal and not as indices.

        :param signal: Signal with shape (samples, variables).
        :return: Segments [seg1, seg2, seg3, ... ].
        """
        segment_indices = self.get_motions_indices(signal)
        motion_segments = []

        for start_idx, end_idx in segment_indices:
            signal_segment = signal[start_idx:end_idx]
            motion_segments.append(signal_segment)

        return motion_segments

    def match_motion_segments(self, present_segments, past_segments):
        """
        Match motion segments together based on results of correlation.
        Segments are after matching transformed into segments of same length with index of the best correlation.

        :param present_segments: Segments from present signal.
        :param past_segments: Segments from past signal.
        :return: Array of matched segments [(seg1, seg2), (seg11, seg22), ...].
        """
        corr_matrix = np.zeros((len(present_segments), len(past_segments)))
        corr_index_matrix = np.zeros((len(present_segments), len(past_segments)))

        for i, seg1 in enumerate(present_segments):
            for j, seg2 in enumerate(past_segments):
                if len(seg1) > len(seg2):
                    corr = correlate(seg1, seg2, mode='valid')
                else:
                    corr = correlate(seg2, seg1, mode='valid')

                norm = np.sqrt(np.sum(seg1 ** 2) * np.sum(seg2 ** 2))
                corr = corr / norm
                max_corr = np.max(corr)
                corr_index = np.argmax(corr)
                corr_index_matrix[i, j] = int(corr_index)
                corr_matrix[i, j] = max_corr

        matched_segments = []
        for i in range(len(corr_matrix)):
            best_corr = 0
            best_corr_shift = 0
            best_corr_idx = 0
            for j in range(len(corr_matrix[i])):
                if (corr_matrix[i][j] > self.similarity_threshold) and (corr_matrix[i][j] > best_corr):
                    best_corr = corr_matrix[i][j]
                    best_corr_shift = corr_index_matrix[i][j]
                    best_corr_idx = j

            if best_corr > self.similarity_threshold:
                present_segment = np.array(present_segments[i])
                past_segment = np.array(past_segments[best_corr_idx])

                if len(present_segment) > len(past_segment):
                    best_corr_shift = int(best_corr_shift)
                    new_present_segment = present_segment[best_corr_shift: best_corr_shift + len(past_segment)]
                    new_past_segment = past_segment[:len(new_present_segment)]
                else:
                    best_corr_shift = int(best_corr_shift)
                    new_past_segment = past_segment[best_corr_shift: best_corr_shift + len(present_segment)]
                    new_present_segment = present_segment[:len(new_past_segment)]

                matched_segments.append((new_present_segment, new_past_segment, best_corr))

        return matched_segments

    def compare_signals(self, present_signal, past_signal):
        """
        Compare present signal to past signal.

        :param present_signal: Signal from present.
        :param past_signal: Signal from past.
        :return: Dictionary of comparison results.
        """
        present_segments = self.get_motions(present_signal)
        past_segments = self.get_motions(past_signal)
        present_joints_segments = []
        past_joints_segments = []
        for i in range(present_signal.shape[1]):
            present_joints_segments.append(self.get_joint_motions(present_signal[:, i]))
            past_joints_segments.append(self.get_joint_motions(past_signal[:, i]))

        signal_matched_segments = self.match_motion_segments(present_segments, past_segments)
        j1_segments = self.match_motion_segments(present_joints_segments[0], past_joints_segments[0])
        j2_segments = self.match_motion_segments(present_joints_segments[1], past_joints_segments[1])
        j3_segments = self.match_motion_segments(present_joints_segments[2], past_joints_segments[2])
        j4_segments = self.match_motion_segments(present_joints_segments[3], past_joints_segments[3])
        j5_segments = self.match_motion_segments(present_joints_segments[4], past_joints_segments[4])
        j6_segments = self.match_motion_segments(present_joints_segments[5], past_joints_segments[5])

        comparison_results = {
            'signal': get_signal_comparison_results(signal_matched_segments),
            'J1': get_joint_comparison_results(j1_segments),
            'J2': get_joint_comparison_results(j2_segments),
            'J3': get_joint_comparison_results(j3_segments),
            'J4': get_joint_comparison_results(j4_segments),
            'J5': get_joint_comparison_results(j5_segments),
            'J6': get_joint_comparison_results(j6_segments)
        }

        return comparison_results
