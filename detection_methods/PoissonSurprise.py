import numpy as np
import scipy.stats as stat

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.close('all')
plt.ion()


class PoissonSurprise:
    # Minimum surprise value	−log(0.01) ≈ 4.6
    # surprise=3 og -> modified
    # maxBurstIntStart=0.5 for s
    # maxBurstIntEnd=2.0 for s
    @staticmethod
    def detect_bursts(spikeEvs,
                         minBurstLen=2,
                         maxInBurstLen=10,
                         maxBurstIntStart=0.5,
                         maxBurstIntEnd=2.0,
                         surprise=-np.log(0.01)
                         ):
        spikeISI = np.diff(spikeEvs)

        maxSpikeIntStart = np.average(spikeISI) * maxBurstIntStart
        maxSpikeIntEnd = np.average(spikeISI) * maxBurstIntEnd

        avgRate = np.average(1. / spikeISI)

        # print "(Min, Max) = (%.3f, %.3f)" % (np.min(spikeISI), np.max(spikeISI))

        bursts = []
        activeBurst = [-1, -1]

        ix = 0
        while ix < len(spikeISI):
            isi = spikeISI[ix]

            # Initiate burst sequence
            ## Find a spike with ISI < {minSpikeIntStart}
            if activeBurst[0] == -1:
                if isi <= maxSpikeIntStart:
                    activeBurst = [ix, ix]

                    burstExtendForward = 0
                    burstRemoveFront = 0

                    burstExtendForwardSurprise = 0
                else:
                    pass
            else:
                # Test for minimum burst criterion
                ## {minBurstLen} consecutive spikes have ISI of (AVG ISI)*minBurstIntStart

                if isi <= maxSpikeIntEnd:
                    if activeBurst[1] - activeBurst[0] < minBurstLen + 1:
                        if isi <= maxSpikeIntStart:
                            activeBurst[1] = ix

                        else:
                            ix = activeBurst[0]
                            activeBurst = [-1, -1]

                    else:
                        # Test for Poisson Surprise criterion in forward direction
                        evCount = activeBurst[1] - activeBurst[0]
                        evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]

                        # Calculate probability that n or more spikes are found in sample
                        ## survivorFunc(evCount - 1) is equivalent to "n or more"
                        ## survivorFunc = 1 - cumDistFunc
                        ## cumDistFunc = probability of n or fewer events
                        sf_old = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                        temp_ix = ix + 1
                        evCount = temp_ix - activeBurst[0]
                        evTime = spikeEvs[temp_ix] - spikeEvs[activeBurst[0]]

                        sf = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                        # Check up to 10 spikes forward to see if Surprise value increases
                        if sf > burstExtendForwardSurprise and burstExtendForward < maxInBurstLen:
                            activeBurst[1] = ix
                            burstExtendForward = 0
                            burstExtendForwardSurprise = sf

                        elif burstExtendForward < maxInBurstLen:
                            # activeBurst[1] = ix
                            burstExtendForward += 1

                        else:
                            # Test for Poisson Surprise criterion when removing spikes from beginning of burst
                            burstRemoveFront = 0
                            i = 1
                            while burstRemoveFront < maxInBurstLen:

                                sf_old = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                                temp_ix = activeBurst[0] + i
                                evCount = activeBurst[1] - temp_ix
                                evTime = spikeEvs[activeBurst[1]] - spikeEvs[temp_ix]

                                sf = -stat.poisson.logsf(evCount - 1, evTime * avgRate)


                                if sf < sf_old:
                                    burstRemoveFront = 0
                                    i += 1
                                    continue

                                else:
                                    activeBurst[0] = temp_ix
                                    break


                            # Check that the surprise value meets minimum surprise parameter and burst contains enough spikes
                            if -np.log(sf) > surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                                bursts.append(np.array(spikeEvs)[activeBurst])
                            else:
                                ix = activeBurst[0]

                            activeBurst = [-1, -1]


                else:
                    # Check that burst sequence meets minimum burst length
                    if activeBurst[1] - activeBurst[0] > minBurstLen:

                        # Test for Poisson Surprise criterion in backwards direction
                        ## Extend burst sequence backwards

                        burstRemoveFront = 0
                        i = 1
                        while burstRemoveFront < maxInBurstLen:

                            evCount = activeBurst[1] - activeBurst[0]
                            evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]

                            sf_old = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                            temp_ix = activeBurst[0] + i
                            evCount = activeBurst[1] - temp_ix
                            evTime = spikeEvs[activeBurst[1]] - spikeEvs[temp_ix]

                            sf = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                            if sf < sf_old:
                                i += 1
                                continue

                            else:
                                activeBurst[0] = temp_ix
                                break

                        # Check that burst sequence meets Poisson Surprise criterion

                        evCount = activeBurst[1] - activeBurst[0]
                        evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]

                        sf = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                        if sf >= surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                            bursts.append(np.array(spikeEvs[activeBurst]))
                        else:

                            ix = activeBurst[0]
                    else:
                        ix = activeBurst[0]

                    activeBurst = [-1, -1]

            ix += 1

            if ix == len(spikeEvs) - 1:
                evCount = activeBurst[1] - activeBurst[0]
                evTime = spikeEvs[activeBurst[1]] - spikeEvs[activeBurst[0]]
                sf = -stat.poisson.logsf(evCount - 1, evTime * avgRate)

                if sf >= surprise and activeBurst[1] - activeBurst[0] > minBurstLen:
                    bursts.append(np.array(spikeEvs)[activeBurst])

        if len(bursts) > 0:
            burstsFiltered = [bursts[0]]
            for i in range(len(bursts) - 1):
                if burstsFiltered[-1][1] >= bursts[i][0]:
                    burstsFiltered[-1][1] = bursts[i][1]
                else:
                    burstsFiltered.append(bursts[i])

        else:
            burstsFiltered = []

        return np.array(burstsFiltered)
