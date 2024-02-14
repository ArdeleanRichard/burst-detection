# Bakkum DJ, Radivojevic M, Frey U, Franke F, Hierlemann A, Takahashi H. Parameters for burst detection. Front Comput Neurosci. 2014 Jan 13;7:193. doi: 10.3389/fncom.2013.00193. PMID: 24567714; PMCID: PMC3915237.

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class ISIn:
    """
    example usage:
    > nb_detector = ISIn()
    > bursts = nb_detector.burst_detection(spiketime_sec=spike_train, n=10, threshold_msec=50)
    """

    @staticmethod
    def detect_bursts(spike_timestamps_in_ms, n, threshold):
        """
        detect bursts from spike train
        :param spike_timestamps: np.array of spike time in sec scale
        :param n: n for ISIn
        :param threshold: ISIn threshold (in MS scale)
        :return: burst array, burst[i] represents ith burst's start time and end time
        """

        spike_timestamps_in_ms = np.sort(spike_timestamps_in_ms)
        n_spikes = len(spike_timestamps_in_ms)
        burst_idx = np.zeros(n_spikes, dtype=np.int)

        for i in range(n_spikes - n + 1):
            if spike_timestamps_in_ms[i + n - 1] - spike_timestamps_in_ms[i] <= threshold:
                burst_idx[i:i + n] = True

        # extend the train to calculate the difference of burst_idx;
        # so that even if the very first or last spike of the train is in burst, they can be assigned to burst correctly
        extended_idx = np.append(False, burst_idx)
        extended_idx = np.append(extended_idx, False)
        diff = extended_idx[1:] - extended_idx[:-1]

        burst_start_idx = (diff[:-1] == 1)
        burst_end_idx = (diff[1:] == -1)

        burst_start = spike_timestamps_in_ms[burst_start_idx]
        burst_end = spike_timestamps_in_ms[burst_end_idx]
        burst = np.array([burst_start, burst_end]).T

        # burst[i]: ith burst
        # burst[:, 0]: array of all bursts' start time
        # burst[:, 1]: array of all bursts' end time
        return burst
        # return burst_start, burst_end