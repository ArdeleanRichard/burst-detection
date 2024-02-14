import time

import numpy as np
from matplotlib import pyplot as plt

from common.time_converter import time_converter_by_measurement
from frequency_domain.superlet.superlet import SuperletTransform
from visualization.label_map import LABEL_COLOR_MAP


def generate_spectrogram(data, ncyc, ord_min, ord_max=None,
                         sampling_frequency=32000, fspace=(300, 7000, 50),
                         label=None,
                         time_measure='s', show=False, title_sig='Signal', title_spec='Signal Spectrogram',
                         save=False, filename=None):
    # fspace: frequency space (start, end, step)


    start = time.time()
    slt = SuperletTransform(
        inputSize       = len(data),
        samplingRate    = sampling_frequency,
        frequencyRange  = (fspace[0], fspace[1]),
        frequencyBins   = fspace[2],
        baseCycles      = ncyc,
        superletOrders  = (ord_min, ord_min if ord_max is None else ord_max)
    )

    spectrum = slt.transform(data)
    print(f"Time: {time.time() - start}")

    if show == True:
        plot_spectrogram_and_signal(spectrum, data, sampling_frequency, fspace, label=label, time_measure=time_measure,
                                    title_sig=title_sig, title_spec=title_spec, save=save, filename=filename)

    return spectrum


def plot_spectrogram(spectrogram, signal, sampling_frequency=32000, fspace=(300, 7000, 50),
                     label=None, time_measure='s',
                     title='Signal Spectrogram', show=True, cmap='jet'):
    foi = np.linspace(fspace[0], fspace[1])

    plt.title(title)
    time, time_multiplier = time_converter_by_measurement(signal.size, sampling_frequency, time_measure)
    upper_extent = len(signal) / sampling_frequency * time_multiplier

    extent = [0, upper_extent, foi[0], foi[-1]]
    im = plt.imshow(spectrogram, cmap=cmap, aspect="auto", extent=extent, origin='lower')

    plt.colorbar(im, orientation='horizontal', shrink=0.7, pad=0.2, label='amplitude')

    plt.title(title)
    plt.xlabel(f"Time ({time_measure})")
    plt.ylabel("Frequency (Hz)")

    if show == True:
        plt.show()


def plot_spectrogram_and_signal(spectrogram, signal, sampling_frequency=32000, fspace=(300, 7000, 50),
                                label=None, time_measure='s',
                                title_sig='Signal',
                                title_spec='Signal Spectrogram',
                                show=True,
                                save=False,
                                filename=None,
                                cmap='jet'):
    foi = np.linspace(fspace[0], fspace[1])
    time, time_multiplier = time_converter_by_measurement(signal.size, sampling_frequency, time_measure)
    upper_extent = len(signal) / sampling_frequency * time_multiplier
    extent = [0, upper_extent, foi[0], foi[-1]]


    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))

    if label is not None:
        ax1.plot(time, signal, c=LABEL_COLOR_MAP[label])
    else:
        ax1.plot(time, signal)
    ax1.set_title(title_sig)
    ax1.set_ylabel('Voltage (mV)')


    im = ax2.imshow(spectrogram, cmap=cmap, aspect="auto", extent=extent, origin='lower')

    plt.colorbar(im, ax=ax2, orientation='horizontal', shrink=0.7, pad=0.2, label='amplitude')

    ax2.set_title(title_spec)
    ax2.set_xlabel(f"Time ({time_measure})")
    ax2.set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    if save == True:
        plt.savefig(filename)

    if show == True:
        plt.show()
