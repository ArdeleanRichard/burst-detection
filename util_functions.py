import numpy as np

from data_parsing.EpdParser import EpdParser
from data_parsing.SpktweParser import SpktweParser
from detection_methods.CumulativeMovingAverage import CumulativeMovingAverage
from detection_methods.ISIRankThreshold import ISIRankThreshold
from detection_methods.ISIn import ISIn
from detection_methods.MaxInterval import MaxInterval
from detection_methods.PoissonSurprise import PoissonSurprise
from detection_methods.RankSurprise import RankSurprise



def parse_line_csv(line):
    split_line = line.strip("\n").split(",")
    data = []
    for item in split_line:
        data.append(float(item))

    return data


def create_spike_in_burst_booleans(spike_timestamps, burst_begs, burst_ends):
    spikes_in_burst = []
    for spike_ts in spike_timestamps:
        flag = True
        for (beg, end) in zip(burst_begs, burst_ends):
            if beg <= spike_ts <= end:
                spikes_in_burst.append(1)
                flag = False
                break
        if flag == True:
            spikes_in_burst.append(0)

    # print(spike_timestamps_in_ms)
    # print(gt_spikes_in_burst)
    # print(len(spike_timestamps_in_ms))
    # print(len(gt_spikes_in_burst))

    return np.array(spikes_in_burst)


def choose_method_return_burst_beg_end(method, spike_timestamps_in_s):
    spike_timestamps_in_ms = spike_timestamps_in_s * 1000
    if method == 'ISIn':
        bursts = ISIn.detect_bursts(spike_timestamps_in_ms=spike_timestamps_in_ms, n=10, threshold=1000)

        m_bursts_begs = [burst[0] for burst in bursts]
        m_bursts_ends = [burst[1] for burst in bursts]

        return m_bursts_begs, m_bursts_ends

    if method == 'PS':
        bursts = PoissonSurprise.detect_bursts(spike_timestamps_in_s, minBurstLen=3, maxInBurstLen=10, maxBurstIntStart=0.5, maxBurstIntEnd=2.0, surprise=-np.log(0.01))

        m_bursts_begs = [burst[0] for burst in bursts]
        m_bursts_ends = [burst[1] for burst in bursts]

        return m_bursts_begs, m_bursts_ends

    if method == 'RS':
        m_bursts_begs, m_bursts_ends = RankSurprise.detect_bursts(spike_timestamps_in_ms, limit=None, RSalpha=-np.log(0.01))

        return m_bursts_begs, m_bursts_ends

    if method == 'MI':
        bursts = MaxInterval.detect_bursts(spike_timestamps_in_s,
                                 max_begin_ISI=0.17,  # in s
                                 max_end_ISI=0.3,  # in s
                                 min_IBI=0.2,  # in ms
                                 min_burst_duration=0.01,  # in ms
                                 min_spikes_in_burst=3
                                 )

        m_bursts_begs = [burst[0] for burst in bursts]
        m_bursts_ends = [burst[-1] for burst in bursts]

        return m_bursts_begs, m_bursts_ends

    elif method == 'CMA':
        bursts = CumulativeMovingAverage.detect_bursts(spike_timestamps_in_ms, tScale=1., minLen=3, histBins=100)

        m_bursts_begs = [burst[0] for burst in bursts]
        m_bursts_ends = [burst[1] for burst in bursts]

        return m_bursts_begs, m_bursts_ends

    elif method == 'IRT':
        m_bursts_begs, m_bursts_ends = ISIRankThreshold.detect_bursts(spike_timestamps=spike_timestamps_in_s)

        return m_bursts_begs, m_bursts_ends


def get_true_positive_fraction(method_spikes_in_burst, gt_spikes_in_burst):
    if len(gt_spikes_in_burst) == 0:
        return 0
    return np.count_nonzero(np.logical_and(method_spikes_in_burst, gt_spikes_in_burst)) / np.count_nonzero(gt_spikes_in_burst)


def get_false_positive_fraction(method_spikes_in_burst, gt_spikes_in_burst):
    if len(gt_spikes_in_burst) == 0:
        return 0
    return np.count_nonzero(method_spikes_in_burst - np.logical_and(method_spikes_in_burst, gt_spikes_in_burst)) / np.count_nonzero(gt_spikes_in_burst)


def get_false_positive_count(method_spikes_in_burst, gt_spikes_in_burst):
    if len(gt_spikes_in_burst) == 0:
        return np.count_nonzero(method_spikes_in_burst)
    return np.count_nonzero(method_spikes_in_burst - np.logical_and(method_spikes_in_burst, gt_spikes_in_burst))




def load_spktwe(DATASET_PATH, CHANNEL, show=False):
    spktweParser = SpktweParser(DATASET_PATH)
    if show == True:
        spktweParser.assert_correctness()
        spktweParser.plot_spikes_on_channel(channel=CHANNEL, show=True)

    selected_ts = np.squeeze(np.array(spktweParser.get_data_from_channel(spktweParser.timestamps_by_channel, CHANNEL, spktweParser.TIMESTAMP_LENGTH)).astype(int))

    return spktweParser, selected_ts


def load_raw_channel(RAW_PATH, CHANNEL, WAVEFORM_ALIGNMENT, WAVEFORM_LENGTH, thr_multiplier=5):
    rawParser = EpdParser(RAW_PATH, FILTER_BAND=(300, 7000), TRIAL_START=1, STIMULUS_ON=2, STIMULUS_OFF=4, TRIAL_END=8)

    rawParser.load_chosen_channel(CHANNEL - 1)
    rawParser.filter_signal(show=False)

    signal = rawParser.data_channel
    timestamps = rawParser.threshold_signal_by_std_dev(thr_multiplier)

    waveforms = []
    for ts in timestamps:
        # remove artefacts with too low amplitudes
        if not signal[ts] < - 1000:
            waveforms.append(signal[ts - WAVEFORM_ALIGNMENT:ts + (WAVEFORM_LENGTH - WAVEFORM_ALIGNMENT)])

    return rawParser, signal, timestamps, waveforms