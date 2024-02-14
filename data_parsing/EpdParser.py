import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

from common.array_processing import split_consecutive
from common.time_converter import time_converter_by_measurement
from data_parsing.AbstractParser import AbstractParser


class EpdParser(AbstractParser):

    def __init__(self, DATASET_PATH,
                 START_AT_STIMULUS_ON=None,
                 TRIAL_START=128, STIMULUS_ON=129, STIMULUS_OFF=130, TRIAL_END=131,
                 FILTER_BAND=None, BAND_TYPE='bandpass'):
        super().__init__()
        self.DATASET_PATH = DATASET_PATH

        self.TRIAL_START = TRIAL_START
        self.STIMULUS_ON = STIMULUS_ON
        self.STIMULUS_OFF = STIMULUS_OFF
        self.TRIAL_END = TRIAL_END

        self.START_AT_STIMULUS_ON = START_AT_STIMULUS_ON

        self.data_in_trials = []
        self.trials = []

        self.parse_epd_file()

        self.FILTER_BAND = FILTER_BAND
        self.BAND_TYPE = BAND_TYPE

        self.load_event_timestamps()
        self.load_event_codes()


    def parse_epd_file(self, show=False):
        """
        Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
        represents the number of spikes in each channel
        """
        string_nr_channels = 'Number of EEG channels:'
        string_nr_channels2 = 'Number of continuous data channels:'

        string_nr_samples = 'Total number of samples:'

        string_filenames = 'List with filenames that hold individual channel samples (32 bit IEEE 754-1985, single precision floating point; amplitudes are measured in uV):'
        string_filenames2 = 'List with filenames that hold individual channel samples (32 bit IEEE 754-1985, single precision floating point):'

        string_channel_names = 'List with labels of EEG channels:'
        string_channel_names2 = 'List with labels of channels:'

        string_sampling_frequency = 'Sampling frequency (Hz):'


        string_event_timestamp_filename = 'File holding event timestamps; timestamp is in samples; (32 bit signed integer file):'
        string_event_codes_filename = 'File holding codes of events corresponding to the event timestamps file; timestamp is in samples; (32 bit signed integer file):'

        for file_name in os.listdir(self.DATASET_PATH):
            full_file_name = self.DATASET_PATH + file_name
            if full_file_name.endswith(".epd"):
                self.EPD_FILE = file_name.split(".")[0]
                file = open(full_file_name, "r")
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                lines = np.array(lines)

                index = self.get_index_line2(lines, string_nr_channels, string_nr_channels2)
                self.NR_CHANNELS = lines[index].astype(int)
                print("NR_CHANNELS:", self.NR_CHANNELS)

                index = self.get_index_line(lines, string_nr_samples)
                self.NR_SAMPLES = lines[index].astype(int)
                print("NR_SAMPLES:", self.NR_SAMPLES)

                index = self.get_index_line(lines, string_sampling_frequency)
                self.SAMPLING_FREQUENCY = lines[index].astype(float)
                print("SAMPLING_FREQUENCY:", self.SAMPLING_FREQUENCY)

                index = self.get_index_line2(lines, string_filenames, string_filenames2)
                self.FILENAMES = lines[index: index + self.NR_CHANNELS]

                index = self.get_index_line2(lines, string_channel_names, string_channel_names2)
                self.CHANNEL_NAMES = lines[index: index + self.NR_CHANNELS]

                index = self.get_index_line(lines, string_event_timestamp_filename)
                self.FILENAME_EVENT_TIMESTAMPS = lines[index]

                index = self.get_index_line(lines, string_event_codes_filename)
                self.FILENAME_EVENT_CODES = lines[index]

                if show==True:
                    print(self.FILENAMES)
                    print(self.CHANNEL_NAMES)
                    print(self.FILENAME_EVENT_TIMESTAMPS)
                    print(self.FILENAME_EVENT_CODES)

    def load_event_timestamps(self):
        self.event_timestamps = np.fromfile(file=self.DATASET_PATH + self.FILENAME_EVENT_TIMESTAMPS, dtype=int)


    def load_event_codes(self):
        self.event_codes = np.fromfile(file=self.DATASET_PATH + self.FILENAME_EVENT_CODES, dtype=int)
        print(np.unique(self.event_codes))



    def load_chosen_channel(self, channel):
        print("-----> Loading:", self.FILENAMES[channel])
        self.data_channel = np.fromfile(file=self.DATASET_PATH + self.FILENAMES[channel], dtype=np.float32)
        print("-----> Finished loading:", self.FILENAMES[channel])

        return self.data_channel


    def split_into_trials(self):
        super().split_event_codes(self.event_codes, self.TRIAL_START, self.STIMULUS_ON, self.STIMULUS_OFF, self.TRIAL_END)
        super().split_event_timestamps_by_codes(self.event_timestamps)

        self.data_in_trials = np.zeros((self.NR_TRIALS, self.NR_CHANNELS, self.trial_timestamp_intervals[0, 1] - self.trial_timestamp_intervals[0, 0]))
        for trial_id, timestamp in enumerate(self.trial_timestamp_intervals):
            self.data_in_trials[trial_id] = self.data_all_channels[:, timestamp[0]:timestamp[1]]


    def load_all_channels(self):
        self.data_all_channels = np.zeros(shape=(self.NR_CHANNELS, self.NR_SAMPLES))
        for chn_id in range(self.NR_CHANNELS):
            data_channel = np.fromfile(file=self.DATASET_PATH + self.FILENAMES[chn_id], dtype=np.float32)

            if self.FILTER_BAND is not None:
                data_channel = self.filter(data_channel)

            self.data_all_channels[chn_id] = data_channel

        self.data_all_channels = self.data_all_channels.T

        return self.data_all_channels

    def split_event_codes(self):
        groups = []
        group = []
        for id, event_code in enumerate(self.event_codes):
            if event_code == self.TRIAL_START:
                group = []
                group.append(id)
            elif len(group) == 1 and event_code == self.STIMULUS_ON:
                group.append(id)
            # codes 1,2,3 for dots, code 50 for monkey, code 4 for mouse
            elif len(group) == 2 and (event_code == 1 or event_code == 2 or event_code == 3 or event_code == 50 or event_code == self.STIMULUS_OFF):
                group.append(id)
            elif len(group) == 3 and event_code == self.TRIAL_END:
                group.append(id)
                groups.append(group)
                group = []

        self.groups = np.array(groups)

        return self.groups


    def split_event_timestamps_by_codes(self):
        groups = self.split_event_codes()

        timestamp_intervals = []
        for group in groups:
            if self.START_AT_STIMULUS_ON is None:
                timestamps_of_interest = [
                    self.event_timestamps[group[0]],
                    self.event_timestamps[group[1]],
                    self.event_timestamps[group[2]],
                    self.event_timestamps[group[-1]]
                ]
            else:
                timestamps_of_interest = [
                    self.event_timestamps[group[1]] - self.START_AT_STIMULUS_ON,
                    self.event_timestamps[group[-1]]
                ]
            timestamp_intervals.append(timestamps_of_interest)

        self.timestamp_intervals = np.array(timestamp_intervals)
        return self.timestamp_intervals


    def read_trial_info(self, info_file, skiprows=10):
        info_table = pd.read_csv(self.DATASET_PATH + info_file, skiprows=skiprows, sep=",", index_col=False)
        return info_table

    def read_trial_info_return_response(self, info_file, skiprows=10):
        info_table = pd.read_csv(self.DATASET_PATH + info_file, skiprows=skiprows, sep=",", index_col=False)
        response_array = info_table["ResponseID"].to_numpy()
        return response_array

    def filter(self, sig):
        sos = signal.butter(1, Wn=self.FILTER_BAND, btype=self.BAND_TYPE, fs=self.SAMPLING_FREQUENCY, output='sos')
        filtered = signal.sosfilt(sos, sig)

        return filtered

    def filter_signal(self, show=False, save_name=None):
        if np.any(self.data_channel < -200):
            self.data_channel[self.data_channel < -200] = -200
        if np.any(self.data_channel > 200):
            self.data_channel[self.data_channel > 100] = 50

        filtered = self.filter(self.data_channel)

        # b, a = signal.butter(3, [300, 7000], 'bandpass', fs=32000, output='ba')
        # filtered = signal.filtfilt(b, a, sig)

        if show == True:
            # t = np.arange(0, len(self.data_channel))
            time, _ = time_converter_by_measurement(self.data_channel.size, self.SAMPLING_FREQUENCY, time_measure='s')
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(time, self.data_channel)
            ax1.set_title('Original signal')
            ax1.set_ylabel('Voltage [mV]')

            ax2.plot(time, filtered)
            ax2.set_title('Bandpass filtered data')
            ax2.set_xlabel('Time [seconds]')
            ax2.set_ylabel('Voltage [mV]')
            plt.tight_layout()
            plt.show()
        return filtered

    def threshold_signal_by_std_dev(self, multiplier=5):
        self.AMP_THR = -1 * multiplier * np.std(self.data_channel)
        all_timestamps = np.where(self.data_channel < self.AMP_THR)[0]

        consecutive_timestamp_groups = split_consecutive(all_timestamps)

        timestamps = []
        for consecutive_ts in consecutive_timestamp_groups:
            timestamps.append(consecutive_ts[0] + np.argmin(self.data_channel[consecutive_ts]))

        return np.array(timestamps)



def filter_sig(sig, band, band_type='bandpass', show=False, save=False, save_name=None):
    sos = signal.butter(1, band, band_type, fs=32000, output='sos')
    filtered = signal.sosfilt(sos, sig)

    # b, a = signal.butter(3, [300, 7000], 'bandpass', fs=32000, output='ba')
    # filtered = signal.filtfilt(b, a, sig)

    if show == True:
        filtered = filtered
        t = range(0, len(filtered))
        plt.figure()

        plt.plot(t, filtered)
        plt.title('Filtered Signal')
        plt.tight_layout()
        if save == True:
            plt.savefig(save_name)
        plt.show()


