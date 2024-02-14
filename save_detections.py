import numpy as np
import pandas as pd
import os

from constants import DATA_PATH
from util_functions import parse_line_csv, create_spike_in_burst_booleans, get_false_positive_fraction, choose_method_return_burst_beg_end, get_true_positive_fraction


def save_detections(type):
    METHODS = ['ISIn', 'IRT', 'MI', 'CMA', 'RS', 'PS']
    for file in os.listdir(DATA_PATH):
        if file.endswith(".spks.csv"):
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
                spike_timestamps_in_s = parse_line_csv(line)

                gt_spikes_in_burst = []
                if os.path.exists(DATA_PATH + gt_len_file):
                    gt_spikes_in_burst = create_spike_in_burst_booleans(spike_timestamps=spike_timestamps_in_s, burst_begs=burst_begs[id], burst_ends=burst_ends[id])

                spike_timestamps_in_s = np.array(spike_timestamps_in_s)

                values = []
                for method in METHODS:
                    bursts_begs, bursts_ends = choose_method_return_burst_beg_end(method=method, spike_timestamps_in_s=spike_timestamps_in_s)
                    method_spikes_in_burst = create_spike_in_burst_booleans(spike_timestamps=spike_timestamps_in_s, burst_begs=bursts_begs, burst_ends=bursts_ends)
                    if type == "true":
                        value = get_true_positive_fraction(method_spikes_in_burst, gt_spikes_in_burst)
                    elif type == "false":
                        value = get_false_positive_fraction(method_spikes_in_burst, gt_spikes_in_burst)
                    values.append(value)

                row = [len(spike_timestamps_in_s), nr_bursts[id]]
                row.extend(values)
                data.append(row)

            columns = ['Timestamps', 'Number of Bursts']
            columns.extend(METHODS)
            df = pd.DataFrame(data, columns=columns)
            if type == "true":
                df.to_csv(DATA_PATH + 'results.fractionTP.' + file.replace(".spks", ""))
            elif type == "false":
                df.to_csv(DATA_PATH + 'results.fractionFP.' + file.replace(".spks", ""))

if __name__ == '__main__':
    save_detections(type="true")
    save_detections(type="false")