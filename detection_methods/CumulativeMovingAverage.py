import numpy as np
from scipy import signal, stats

import matplotlib.pyplot as plt

plt.ion()
plt.close('all')


class CumulativeMovingAverage:

    @staticmethod
    def detect_bursts(data,
                                  tScale=1.,
                                  minLen=3,
                                  histBins=100
                                  ):
        ## Calculate inter spike intervals and histogram
        isi = np.diff(data * tScale)
        xN, xBins = np.histogram(isi, bins=histBins)


        ## Calculate cumulative sum and moving average
        cma = np.array([np.sum(xN[:ix]) / (ix + 1) for ix in range(len(xN) + 1)])

        ## Determine peak of moving average curve

        ## Calculate skew of ISI histgoram
        skew = stats.skew(xN)

        ## Determine alpha1 and alpha2 based on published parameter values
        ## Kapucu, et al. 2012. Burst analysis tool for developing neuronal networks
        ## exhibiting highly varying action potential dynamics. Front. Comp Neurosci.
        ## 8(38).

        alph1 = 1.
        alph2 = 1.

        # alph1(alph2)
        # 1.0(0.5) if skew < 1
        # 0.7(0.5) if 1 ≤ skew < 4
        # 0.5(0.9) if 4 ≤ skew < 9
        # 0.3(0.1) if 9 ≤ skew
        if skew >= 9:
            alph1 = 0.3
            alph2 = 0.1
        elif 4 <= skew and  skew < 9:
            alph1 = 0.5
            alph2 = 0.3
        elif skew < 4:
            alph2 = 0.5
            if 1 <= skew:
                alph1 = 0.7
            else:
                # alph1 = 0.75
                alph1 = 1.

        ## Calculate isi threshold from histogram bins
        x1 = np.max(cma) * alph1
        ixBurstThresh = np.where(np.diff(np.sign(cma - x1)) < 0)[0]
        if len(ixBurstThresh) > 0:
            burstThresh = xBins[ixBurstThresh[-1]]
        else:
            burstThresh = xBins[-1]

        x2 = np.max(cma) * alph2
        ixBurstRelThresh = np.where(np.diff(np.sign(cma - x2)) < 0)[0]
        if len(ixBurstRelThresh) > 0:
            burstRelThresh = xBins[ixBurstRelThresh[-1]]
        else:
            burstRelThresh = xBins[-1]

        ## Determine spike events that are in a burst or are burst-related
        burstSpikes = np.where(isi <= burstThresh)[0] + 1
        burstRelSpikes = np.where((isi <= burstRelThresh) & (isi > burstThresh))[0] + 1

        bursts = []
        breaks = np.where(np.diff(burstSpikes) > 1)[0]

        for ix in range(len(breaks)):
            if ix > 0:
                bursts.append([burstSpikes[breaks[ix - 1] + 1], burstSpikes[breaks[ix]]])
            else:
                bursts.append([burstSpikes[0], burstSpikes[breaks[ix]]])

        if len(breaks) > 0 and len(burstSpikes) > 0:
            bursts.append([breaks[-1] + 1, burstSpikes[-1]])

        validBursts = np.where(np.diff(bursts) >= minLen)[0]

        bursts = np.array(bursts)[validBursts]

        burstsFiltered = []

        for ix, burst in enumerate(bursts):
            burstsFiltered.append(np.copy(burst))
            ix = np.where(burstRelSpikes - burst[0] == -1)[0]
            if len(ix) > 0:
                ix = ix[0]
                if (data[burst[0]] - data[burstRelSpikes[ix]]) * tScale <= burstRelThresh:
                    if ix > 0:
                        while burstRelSpikes[ix] - burstRelSpikes[ix - 1] == 1:
                            if ix > 0:
                                ix -= 1
                            else:
                                break

                    burstsFiltered[-1][0] = burstRelSpikes[ix]

            ix = np.where(burstRelSpikes - burst[1] == 1)[0]
            if len(ix) > 0:
                ix = ix[0]
                if (data[burstRelSpikes[ix]] - data[burst[1]]) * tScale <= burstRelThresh:
                    if ix < len(burstRelSpikes) - 1:
                        while burstRelSpikes[ix + 1] - burstRelSpikes[ix] == 1:
                            if ix < len(burstRelSpikes) - 2:
                                ix += 1
                            else:
                                break

                    burstsFiltered[-1][1] = burstRelSpikes[ix]

        if len(burstsFiltered) > 0:
            burstsFinal = [list(burstsFiltered[0])]

            for burst in burstsFiltered:
                if burst[0] <= burstsFinal[-1][1]:
                    burstsFinal[-1][1] = burst[1]
                else:
                    burstsFinal.append(list(burst))

            burstsFinal = np.array(burstsFinal)
        else:
            burstsFinal = []

        return np.array(data)[burstsFinal]


