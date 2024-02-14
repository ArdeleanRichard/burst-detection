def compare_burst_methods_on_synthetic_data():
    from plot_detection_botplots import create_boxplots
    from read_data import read_and_save
    from save_detections import save_detections

    print("*Reading synthetic data")
    read_and_save()
    print("*Running burst detection algorithms")
    save_detections(type="true")
    save_detections(type="false")
    print("*Plotting results")
    create_boxplots("true")
    create_boxplots("false")


def run_burst_detection_on_real_data():
    import numpy as np

    from constants import sampling_freq, RAW_PATH, SPKTWE_PATH

    from plot_bursts import plot_some_bursts_using_timestamps
    from util_functions import load_spktwe, load_raw_channel, choose_method_return_burst_beg_end

    CHANNEL = 10
    spktweParser, timestamps = load_spktwe(SPKTWE_PATH, CHANNEL)
    rawParser, signal, timestamps_found, waveforms = load_raw_channel(RAW_PATH, CHANNEL, spktweParser.WAVEFORM_ALIGNMENT, spktweParser.WAVEFORM_LENGTH, thr_multiplier=4.25)

    timestamps_in_s = timestamps / sampling_freq

    # method = 'ISIn'
    # method = 'IRT'
    method = 'MI'
    # method = 'RS'
    # method = 'PS'
    # method = 'CMA'
    m_bursts_begs, m_bursts_ends = choose_method_return_burst_beg_end(method=method, spike_timestamps_in_s=timestamps_in_s)

    bursts = np.vstack((np.array(m_bursts_begs), np.array(m_bursts_ends))).T

    test = []
    for b_ts in bursts:
        test.append((b_ts * sampling_freq).astype(int))

    plot_some_bursts_using_timestamps(signal, test, spktweParser.negative_thresholds[CHANNEL],
                                      spktweParser.WAVEFORM_ALIGNMENT, spktweParser.WAVEFORM_LENGTH,
                                      measurement='samples', title=method)


if __name__ == '__main__':
    compare_burst_methods_on_synthetic_data()
    # run_burst_detection_on_real_data()