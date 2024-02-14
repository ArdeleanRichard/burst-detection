import numpy as np
from matplotlib import pyplot as plt

from common.time_converter import time_converter_by_measurement
from frequency_domain.superlet.apply_slt import generate_spectrogram, plot_spectrogram_and_signal


def convert_timestamps_to_waveform(signal, burst_timestamps, WAVEFORM_ALIGNMENT, WAVEFORM_LENGTH):
    return signal[burst_timestamps[0] - WAVEFORM_ALIGNMENT: burst_timestamps[-1] + (WAVEFORM_LENGTH - WAVEFORM_ALIGNMENT) ]


def plot_burst(waveform, neg_thr, sampling_frequency=32000, measurement='samples', title=''):
    time, time_multiplier = time_converter_by_measurement(waveform.size, sampling_frequency=sampling_frequency, time_measure=measurement)

    plt.title(title)
    plt.plot(time, waveform, label='burst')
    plt.axhline(y=neg_thr, color='r', linestyle='dashed', label='AmpThr')
    plt.xlabel(f"Time ({measurement})")
    plt.ylabel(f"Voltage (mV)")
    plt.legend()
    plt.show()


def plot_some_bursts_using_timestamps(signal, bursts_timestamps, neg_thr,
                                      WAVEFORM_ALIGNMENT, WAVEFORM_LENGTH,
                                      sampling_frequency=32000, spectrogram_freq_range=[300, 7000, 100],
                                      measurement='samples', title='',
                                      show_spectrogram=False,
                                      show_subspike_spectrograms=False):
    for b_ts in bursts_timestamps[4:7]:
        if measurement == 'samples':
            pass
        elif measurement == 'ms':
            b_ts = (b_ts / 1000 * sampling_frequency).astype(int)

        waveform = convert_timestamps_to_waveform(signal, b_ts, WAVEFORM_ALIGNMENT, WAVEFORM_LENGTH)

        peaks = np.array(b_ts)
        peaks = peaks - peaks[0] + WAVEFORM_ALIGNMENT

        plot_burst(waveform, neg_thr, measurement='ms', title=title)

        spectrogram = generate_spectrogram(waveform, ncyc=1.5, ord_min=2,
                                           sampling_frequency=sampling_frequency, fspace=spectrogram_freq_range,
                                           time_measure='ms', show=show_spectrogram)


        intervals = np.append(peaks - WAVEFORM_ALIGNMENT, len(waveform))
        if show_subspike_spectrograms == True:
            for index in range(len(intervals) - 1):
                plot_spectrogram_and_signal(spectrogram[:, intervals[index]:intervals[index+1]],
                                            waveform[intervals[index]:intervals[index+1]],
                                            sampling_frequency=sampling_frequency, fspace=spectrogram_freq_range, label=None, time_measure='ms',
                                            title_sig=f'Subspike', show=False)
                plt.show()
