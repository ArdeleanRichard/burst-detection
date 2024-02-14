import numpy as np


class MaxInterval:
    # Maximum beginning ISI	        0.17 s
    # Maximum end ISI	            0.3 s
    # Minimum interburst interval	0.2 s
    # Minimum burst duration	    0.01 s
    # Minimum spikes in a burst	    3
    @staticmethod
    def detect_bursts(timestamps_in_ms,
                     max_begin_ISI=170,      # in ms
                     max_end_ISI=300,        # in ms
                     min_IBI=200,            # in ms
                     min_burst_duration=10,  # in ms
                     min_spikes_in_burst=3   # in ms
                    ):


        lenSpikes = len(timestamps_in_ms)

        allBurstData = []
        inBurst = False
        burstNum = 0

        currentBurst = []

        # Phase 1 - Burst Detection
        # Here a burst is defined as starting when two consecutive spikes have an
        # ISI less than max_begin_ISI apart. The end of the burst is given when two
        # spikes have an ISI greater than max_end_ISI.
        # Find ISIs closer than max_begin_ISI and end with max_end_ISI.
        # The last spike of the previous burst will be used to calculate the IBI.
        # For the first burst, there is no previous IBI.
        n = 1
        while n < lenSpikes:
            ISI = timestamps_in_ms[n] - timestamps_in_ms[n-1] # Calculate ISI

            if inBurst == True: # currently in burst
                if ISI > max_end_ISI: # end the burst
                    currentBurst.append(timestamps_in_ms[n-1]) # store spike in burst
                    burstNum += 1
                    allBurstData.append(currentBurst) # store burst data
                    currentBurst = [] # reset for new burst
                    inBurst = False # no longer in burst
                else:
                    currentBurst.append(timestamps_in_ms[n-1])
            else: # currently not in burst
                if ISI < max_begin_ISI: #possibly found start of new burst
                    currentBurst.append(timestamps_in_ms[n - 1])
                    inBurst = True
            n += 1

        # Calculate IBIs
        IBI = []
        for i in range(1, burstNum):
            prevBurstEnd = allBurstData[i-1][-1]
            currBurstStart = allBurstData[i][0]
            IBI.append(currBurstStart - prevBurstEnd)


        # Phase 2 - Merging of Bursts
        # Here we see if any pair of bursts have an IBI less than min_IBI; if so,
        # we then merge the bursts. We specifically need to check when say three
        # bursts are merged into one.
        tmp = allBurstData.copy()
        allBurstData = []
        for i in range(1, burstNum):
            prevBurst = tmp[i-1]
            currBurst = tmp[i]

            if IBI[i-1] < min_IBI: # IBI is too short to be separate bursts
                prevBurst.extend(currBurst)

            allBurstData.append(prevBurst)

        if burstNum >= 2:
            allBurstData.append(currBurst)

        # Phase 3 - Quality Control
        # Remove small bursts less than min_bursts_duration or having too few
        # spikes less than min_spikes_in_bursts. In this phase we have the
        # possibility of deleting all spikes.
        tooShort = 0
        if burstNum > 1:
            for i in range(0, burstNum):
                currentBurst = allBurstData[i]
                if len(currentBurst) < min_spikes_in_burst:
                    currentBurst = []
                elif (currentBurst[-1] - currentBurst[0]) < min_burst_duration:
                    currentBurst = []
                    tooShort = tooShort + 1
                allBurstData[i] = currentBurst

        if len(allBurstData) != 0:
            tooShort = tooShort / len(allBurstData)
            allBurstData = list(filter(None, allBurstData)) # remove empty lists from quality control
            allBurstData = [np.array(x) for x in allBurstData]
        else:
            allBurstData = []

        # return allBurstData, tooShort
        return allBurstData