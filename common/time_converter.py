import numpy as np


def time_converter_by_measurement(data_size, sampling_frequency, time_measure):
    # 32000Hz -> time in s = 1/32000 = 0.00003125
    # tims in ms = 0.00003125 * 1000 = 0.03125
    # 100 samples = 3ms
    # 200 samples = 6ms
    if time_measure == 's':
        time_multiplier = 1
    elif time_measure == 'ms':
        time_multiplier = 1000
    elif time_measure == 'samples':
        time_multiplier = sampling_frequency

    time = np.arange(data_size) / sampling_frequency * time_multiplier

    return time, time_multiplier