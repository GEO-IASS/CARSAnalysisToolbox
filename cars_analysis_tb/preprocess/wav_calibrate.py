import numpy as np


def wavenum_calibration(peak_pix, peaks, cal_pix, pump_wl):
    lam0 = pump_wl

    wn0 = 1 / (100 * lam0)
    wn = wn0 + np.asarray(peaks)

    lam = 1 / (100 * wn)

    pix = peak_pix

    p = np.polyfit(pix, lam, 1)

    wavelength = np.polyval(p, cal_pix)

    wavenum = 1 / (100 * wavelength) - wn0

    return wavenum
