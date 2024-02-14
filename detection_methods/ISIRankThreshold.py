import math

import numpy as np
import scipy.stats as ss

# HennigMethod
class ISIRankThreshold:

    @staticmethod
    def detect_bursts(spike_timestamps, cutoff_prob=0.05):
        # spike_timestamps = np.array([2,5,10,12,21,22,24])
        all_isi = np.diff(spike_timestamps)

        if len(all_isi) < 1:
            return [], []
        else:
            isi_rank = ss.rankdata(all_isi) #1 is smallest ISI
            spike_timestamps_length = np.ceil(np.amax(spike_timestamps)).astype(int)
            spike_counts = []

        for i in range(spike_timestamps_length):
            # print(len(spike_timestamps>i))
            # print(len(spike_timestamps<(i+1)))
            spike_counts.append(np.sum(np.logical_and(spike_timestamps>=i,  spike_timestamps<(i+1))))

        hist, x_values = np.histogram(spike_counts, bins=200)
        dist = 1 - np.cumsum(hist/np.sum(hist))
        cutoff_idx = np.sum(dist > cutoff_prob)

        w = 2
        hist_mids = np.convolve(x_values, np.ones(w), 'valid') / w

        theta_c = max([2, math.ceil(hist_mids[cutoff_idx])])
        theta_c_end = int(theta_c * 0.5)

        isi_rel_rank = isi_rank / np.max(isi_rank)

        j = 0
        burst_on = 0
        bc = 0
        dt = 1
        burst_time = []
        burst_end  = []
        burst_dur  = []
        burst_size = []
        burst_beg  = []

        while j < len(all_isi) - theta_c:
            if burst_on == 0 and isi_rel_rank[j] < 0.5:  # burst begins when rank of isi<0.5
                if spike_timestamps[j+theta_c] < spike_timestamps[j]+dt:
                    burst_on = 1
                    burst_time.append(spike_timestamps[j])
                    burst_beg.append(spike_timestamps[j])
                    brc = j
            elif burst_on == 1:
                if spike_timestamps[j+theta_c_end] > spike_timestamps[j]+dt:
                    burst_end.append(spike_timestamps[j])
                    burst_dur.append(spike_timestamps[j]-burst_time[bc])
                    burst_size.append(j-brc)
                    bc = bc+1
                    burst_on = 0
            j = j+1


        if burst_on == 1:
            tmp = spike_timestamps[j]-burst_time[-1]
            burst_end.append(burst_time[-1]+tmp)
            burst_dur.append(spike_timestamps[j]-burst_time[-1])
            burst_size.append(j-brc)
            # bc = bc+1


        burst_time = np.array(burst_time)
        burst_beg  = np.array(burst_beg)
        burst_end  = np.array(burst_end)
        burst_dur  = np.array(burst_dur)
        burst_size = np.array(burst_size)

        N_burst = len(burst_time)
        if N_burst < 1:
            return [], []
        else:
            end = burst_beg + burst_size
            IBI = [burst_time[-1]-burst_end[-N_burst]]
            length = burst_size+1
            mean_isis = burst_dur / (length-1)

        return burst_beg, burst_end