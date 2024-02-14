import os

import numpy as np
import matplotlib.pyplot as plt

from data_parsing.AbstractParser import AbstractParser

class SpktweParser(AbstractParser):
    def __init__(self, DATASET_PATH, show=False):
        super().__init__()
        self.DATASET_PATH = DATASET_PATH
        self.show = show

        self.parse_spktwe_file()
        self.get_data()

    def parse_spktwe_file(self):
        """
        Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
        represents the number of spikes in each channel
        @param dir_name: Path to directory that contains the files
        @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
        """
        string_electrodes_per_multitrode = 'Number of electrodes per multitrode (1 = no multitrode; 4 = tetrode; etc.):'
        string_spiking_frequency = 'Spike times sampling frequency [Hz]:'
        string_waveform_frequency = 'Waveform internal sampling frequency [Hz] (can be different than the sampling of the spike times):'
        string_recording_length = 'Recording length (in spike time samples):'
        string_stored_data_channels = 'Number of stored data channels:'
        string_channel_names = 'List with the names of stored channels:'
        string_spike_counts = 'Number of spikes in each stored channel::'
        string_waveform_length = 'Waveform length in samples:'
        string_align = 'Waveform spike align offset - the sample in waveform that is aligned to the spike:'
        string_negative_thresholds = 'List with negative channel thresholds used to extract spikes (multiple thresholds for multitrode):'


        for file_name in os.listdir(self.DATASET_PATH):
            full_file_name = self.DATASET_PATH + file_name
            if full_file_name.endswith(".spktwe"):
                file = open(full_file_name, "r")
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                lines = np.array(lines)

                try:
                    index = self.get_index_line(lines, string_electrodes_per_multitrode)
                    self.NR_ELECTRODES_PER_MULTITRODE = lines[index].astype(int)
                except IndexError:
                    self.NR_ELECTRODES_PER_MULTITRODE = 1

                index = self.get_index_line(lines, string_spiking_frequency)
                self.SPIKING_FREQUENCY = lines[index].astype(float)

                index = self.get_index_line(lines, string_waveform_frequency)
                self.WAVEFORM_FREQUENCY = lines[index].astype(float)

                index = self.get_index_line(lines, string_recording_length)
                self.RECORDING_LENGTH = lines[index].astype(int)

                index = self.get_index_line(lines, string_stored_data_channels)
                self.NR_CHANNELS = lines[index].astype(int)

                index = self.get_index_line(lines, string_channel_names)
                self.stored_chns = lines[index:index + self.NR_CHANNELS]

                index = self.get_index_line(lines, string_spike_counts)
                self.spike_count_per_channels = lines[index: index + self.NR_CHANNELS].astype(int)


                index = self.get_index_line(lines, string_waveform_length)
                self.WAVEFORM_LENGTH = lines[index].astype(int)

                index = self.get_index_line(lines, string_align)
                self.WAVEFORM_ALIGNMENT = lines[index].astype(int)

                try:
                    index = self.get_index_line(lines, string_negative_thresholds)
                    self.negative_thresholds = lines[index: index + self.NR_CHANNELS].astype(float)
                except IndexError:
                    self.negative_thresholds = None


                if self.show == True:
                    print("NR_ELECTRODES_PER_MULTITRODE:", self.NR_ELECTRODES_PER_MULTITRODE)
                    print("CHANNELS:", self.NR_CHANNELS)
                    print(self.stored_chns)
                    print(self.spike_count_per_channels)
                    print("WAVEFORM_LENGTH:", self.WAVEFORM_LENGTH)
                    print("WAVEFORM_ALIGNMENT:", self.WAVEFORM_ALIGNMENT)


    def separate_by_channel(self, data, length):
        """
        Separates a data by spikes_per_channel, knowing that data are put one after another and channel after channel
        :param spikes_per_channel: list of counts - returned by parse_spktwe_file
        :param data: timestamps / waveforms
        :param length: 1 for timestamps and 58 for waveforms
        :return:
        """
        separated_data = []
        sum = 0
        for spikes_in_channel in self.spike_count_per_channels:
            data_on_channel = np.array(data[sum * length: (sum + spikes_in_channel) * length])
            data_on_channel = np.squeeze(np.reshape(data_on_channel, (-1, length)))
            separated_data.append(data_on_channel)
            sum += spikes_in_channel

        return np.array(separated_data)


    def find_waverform_files(self):
        """
        Searches in a folder for certain file formats and returns them
        :param DATASET_PATH: folder that contains files, looks for files that contain the data
        :return: returns the names of the files that contains data
        """
        self.file_timestamp = None
        self.file_waveform = None
        self.file_event_timestamps = None
        self.file_event_codes = None
        for file_name in os.listdir(self.DATASET_PATH):
            if file_name.endswith(".spiket"):
                self.file_timestamp = self.DATASET_PATH + file_name
            if file_name.endswith(".spikew"):
                self.file_waveform = self.DATASET_PATH + file_name
            if file_name.endswith(".eventt"):
                self.file_event_timestamps = self.DATASET_PATH + file_name
            if file_name.endswith(".eventc"):
                self.file_event_codes = self.DATASET_PATH + file_name


    def get_data(self):
        self.find_waverform_files()

        self.timestamps = np.fromfile(file=self.file_timestamp, dtype=int)
        self.waveforms = np.fromfile(file=self.file_waveform, dtype=np.float32)
        self.event_timestamps = np.fromfile(file=self.file_event_timestamps, dtype=int)
        self.event_codes = np.fromfile(file=self.file_event_codes, dtype=int)

        self.timestamps_by_channel = self.separate_by_channel(self.timestamps, self.TIMESTAMP_LENGTH)
        self.waveforms_by_channel = self.separate_by_channel(self.waveforms, self.WAVEFORM_LENGTH)


    def assert_correctness(self):
        print(f"DATASET is in folder: {self.DATASET_PATH}")
        timestamp_file, waveform_file, event_timestamps_file, event_codes_file = self.find_waverform_files()
        print(f"TIMESTAMP file found: {timestamp_file}")
        print(f"WAVEFORM file found: {waveform_file}")
        print("--------------------------------------------")

        print(f"Number of Channels: {self.spike_count_per_channels.shape}")
        print(f"Number of Spikes on all Channels: {np.sum(self.spike_count_per_channels)}")
        print("--------------------------------------------")

        # timestamps = self.FileReader.read_timestamps(timestamp_file)
        timestamps = np.fromfile(file=timestamp_file, dtype=np.int)
        print(f"Timestamps found in file: {timestamps.shape}")
        print(f"Number of spikes in all channels should be equal: {np.sum(self.spike_count_per_channels)}")
        print(f"Assert equality: {len(timestamps) == np.sum(self.spike_count_per_channels)}")
        timestamps_by_channel = self.separate_by_channel(timestamps, self.TIMESTAMP_LENGTH)
        print(f"Spikes per channel parsed from file: {self.spike_count_per_channels}")
        print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_channel))}")
        print(f"Assert equality: {list(self.spike_count_per_channels) == list(map(len, timestamps_by_channel))}")
        print("--------------------------------------------")

        # waveforms = self.FileReader.read_waveforms(waveform_file)
        waveforms = np.fromfile(file=waveform_file, dtype=np.float32)
        print(f"Waveforms found in file: {waveforms.shape}")
        print(f"Waveforms should be Timestamps*{self.WAVEFORM_LENGTH}: {len(timestamps) * self.WAVEFORM_LENGTH}")
        print(f"Assert equality: {len(timestamps) * self.WAVEFORM_LENGTH == len(waveforms)}")
        waveforms_by_channel = self.separate_by_channel(waveforms, self.WAVEFORM_LENGTH)
        print(f"Waveforms per channel: {list(map(len, waveforms_by_channel))}")
        print(f"Spikes per channel parsed from file: {self.spike_count_per_channels}")
        waveform_lens = list(map(len, waveforms_by_channel))
        print(
            f"Waveforms/{self.WAVEFORM_LENGTH} per channel should be equal: {[i // self.WAVEFORM_LENGTH for i in waveform_lens]}")
        print(f"Assert equality: {list(self.spike_count_per_channels) == [i // self.WAVEFORM_LENGTH for i in waveform_lens]}")
        print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
        print("--------------------------------------------")


    def get_data_from_channel(self, data_by_channel, channel, length):
        """
        Selects data by chosen channel
        :param data_by_channel: all the data of a type (all timestamps / all waveforms from all channels)
        :param channel: receives inputs from 1 to NR_CHANNELS, stored in list with start index 0 (so its channel -1)
        :param length: 1 for timestamps and 58 for waveforms
        :return:
        """
        data_on_channel = data_by_channel[channel - 1]
        data_on_channel = np.reshape(data_on_channel, (-1, length))

        return data_on_channel


    def plot_spikes_on_channel(self, channel, show=False):
        waveforms_on_channel = self.get_data_from_channel(self.waveforms_by_channel, channel, self.WAVEFORM_LENGTH)
        plt.figure()
        plt.title(f"Spikes ({len(waveforms_on_channel)}) on channel {channel}")
        time = np.arange(waveforms_on_channel.shape[1])
        for i in range(0, len(waveforms_on_channel)):
            plt.plot(time, waveforms_on_channel[i])

        if show:
            plt.show()

    def plot_all_spikes_by_channel(self):
        for index in range(len(self.waveforms_by_channel)):
            self.plot_spikes_on_channel(index + 1, show=False)
        plt.show()


