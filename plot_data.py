import numpy as np
from matplotlib import pyplot as plt

from constants import PLOT_PATH, DATA_PATH
from util_functions import parse_line_csv
import os


def plot_train(neuron):
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    xticks = range(0, len(neuron), 30*1000)
    axs.plot(neuron)
    axs.set_ylim(bottom=np.amin(neuron) - 0.1, top=np.amax(neuron) * 1.1)
    axs.xaxis.set_ticks(xticks)
    axs.xaxis.set_ticklabels((np.array(list(xticks)) / 1000).astype(int))
    plt.xlabel("Time (s)")
    plt.show()


def plot_trains(examples, save=False):
    fig, axs = plt.subplots(len(examples), 1, figsize=(15, 5), sharex=True)

    names = ["High frequency Bursts", "Long Bursts", "Noisy Bursts", "Non-bursting", "Non-stationary", "Regular Bursts"]
    mapping = [4, 3, 5, 0, 1, 2]
    for id, ex in enumerate(examples):
        xticks = range(0, len(ex), 30 * 1000)
        axs[mapping[id]].plot(ex)
        axs[mapping[id]].set_ylim(bottom=np.amin(ex) - 0.1, top=np.amax(ex) * 1.1)
        axs[mapping[id]].xaxis.set_ticks(xticks)
        axs[mapping[id]].xaxis.set_ticklabels((np.array(list(xticks)) / 1000).astype(int))
        axs[mapping[id]].set_ylabel(names[id], rotation=0, labelpad=60, fontsize=10, fontweight='bold')

    plt.xlabel("Time (s)", fontsize=15)

    if save == True:
        plt.savefig(PLOT_PATH + "/burst_trains.svg")

    plt.show()


if __name__ == "__main__":
    examples = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".spks.csv") and not file.startswith("comp."):
            gt_len_file = file.replace(".spks", ".num.bursts")
            burst_beg_file = file.replace(".spks", ".burst.beg")
            burst_end_file = file.replace(".spks", ".burst.end")

            nr_bursts = []
            burst_begs = []
            burst_ends = []
            if os.path.exists(DATA_PATH + gt_len_file):
                f = open(DATA_PATH + gt_len_file)
                lines = f.readlines()
                for line in lines:
                    nr_bursts.append(int(line))

                f = open(DATA_PATH + burst_beg_file)
                lines = f.readlines()
                for line in lines:
                    burst_begs.append(parse_line_csv(line))

                f = open(DATA_PATH + burst_end_file)
                lines = f.readlines()
                for line in lines:
                    burst_ends.append(parse_line_csv(line))
            else:
                nr_bursts = [0] * 100

            print(file)
            f = open(DATA_PATH + file)
            lines = f.readlines()
            data = []
            for id, line in enumerate(lines):
                spike_timestamps_in_s = np.array(parse_line_csv(line))
                spike_timestamps_in_ms = (spike_timestamps_in_s * 1000).astype(int)
                # print(spike_timestamps_in_ms)

                spike_train = np.zeros(300 * 1000)
                spike_train[spike_timestamps_in_ms] = 1
                # plot_train(spike_train)
                examples.append(spike_train)

                break

    plot_trains(examples, save=True)