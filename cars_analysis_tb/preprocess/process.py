"""
Created on Thu Jun 22 15:27:34 2017
@author: Priyank Shah

Process CARS files
===============================================================================

    process_cars()
    Process .cars files

    process_fl()
    Process _fl_traces.cars background files

    remove_background()
    Subtract the _fl_traces.cars data from the .cars data

    remove_stokespump()
    Remove the separate pump- and stokes- only data from the .cars data

    remove_etaloning()
    Normalisation of the .cars data by division with the etalon data

    N.B:
    The process_cars() and process_fl() functions are written to process .cars
    files that were created after April 2015. Those created before have slightly
    different file formats. Refer to the MATLAB scripts to determine what
    alterations will be needed to process any older .cars files.

===============================================================================
"""

import numpy as np
import struct
from scipy.signal import savgol_filter as sg_filt


class EmptyClass:
    def __init__(self):
        pass


def process_cars(filename):
    if not filename == 'none':
        # Reading in the binary CARS file
        fid = open(filename, 'rb')
        data = fid.read()
        HeaderLength = struct.unpack('>H', data[0:2])[0]
        PointsPerSpectrum = struct.unpack('>H', data[2:4])[0]
        SpectraPerPixel = struct.unpack('>H', data[4:6])[0]
        XPixels = struct.unpack('>H', data[6:8])[0]
        YPixels = struct.unpack('>H', data[8:10])[0]
        ZPixels = struct.unpack('>H', data[10:12])[0]
        MSB = struct.unpack('>H', data[12:14])[0]
        LSB = struct.unpack('>H', data[14:16])[0]
        XYStepSize = struct.unpack('>H', data[16:18])[0] / 1000
        ZStepSize = struct.unpack('>H', data[18:20])[0] / 1000

        ExposureTime = (MSB * 2 ^ 16 + LSB) / 1e5

        s1 = np.full((XPixels, YPixels, ZPixels, PointsPerSpectrum), 0, dtype='float')
        s2 = np.full((XPixels, YPixels, ZPixels, PointsPerSpectrum), 0, dtype='float')

        # Creating and reading in the data variables s1 and s2
        a = 20
        for zi in range(0, ZPixels):
            for yi in range(0, YPixels):
                for xi in range(0, XPixels):
                    s1[xi, yi, zi, :] = np.flipud(struct.unpack('>512H', data[a:a + (2 * PointsPerSpectrum)]))
                    s2[xi, yi, zi, :] = np.flipud(
                        struct.unpack('>512H', data[a + (2 * PointsPerSpectrum):a + 2 * (2 * PointsPerSpectrum)]))

                    a = a + 2 * (2 * PointsPerSpectrum)

        # Output variable for the data extracted above
        cars = EmptyClass()
        cars.HeaderLength = HeaderLength
        cars.PointsPerSpectrum = PointsPerSpectrum
        cars.SpectraPerPixel = SpectraPerPixel
        cars.XPixels = XPixels
        cars.YPixels = YPixels
        cars.ZPixels = ZPixels
        cars.XYStepSize = XYStepSize
        cars.ZStepSize = ZStepSize
        cars.ExposureTime = ExposureTime
        cars.s1 = s1.astype(float)
        cars.s2 = s2.astype(float)

        return cars


def process_fl(filename):
    if not filename == 'none':
        fid = open(filename, 'rb')
        data = fid.read()
        HeaderLength = struct.unpack('>H', data[0:2])[0]
        PointsPerSpectrum = struct.unpack('>H', data[2:4])[0]
        SpectraPerPixel = struct.unpack('>H', data[4:6])[0]
        XPixels = struct.unpack('>H', data[6:8])[0]
        YPixels = struct.unpack('>H', data[8:10])[0]
        ZPixels = struct.unpack('>H', data[10:12])[0]
        MSB = struct.unpack('>H', data[12:14])[0]
        LSB = struct.unpack('>H', data[14:16])[0]
        Average = struct.unpack('>H', data[16:18])[0]
        XYStepSize = struct.unpack('>H', data[18:20])[0] / 1000
        ZStepSize = struct.unpack('>H', data[20:22])[0] / 1000

        ExposureTime = (MSB * 2 ^ 16 + LSB) / 1e5

        s1 = np.full((Average, PointsPerSpectrum), 0, dtype='float')
        s2 = np.full((Average, PointsPerSpectrum), 0, dtype='float')

        a = 22
        for fi in range(0, Average):
            s1[fi, :] = np.flipud(struct.unpack('>512H', data[a:a + (2 * PointsPerSpectrum)]))
            s2[fi, :] = np.flipud(
                struct.unpack('>512H', data[a + (2 * PointsPerSpectrum):a + 2 * (2 * PointsPerSpectrum)]))

            a = a + 2 * (2 * PointsPerSpectrum)

        cars = EmptyClass()
        cars.HeaderLength = HeaderLength
        cars.PointsPerSpectrum = PointsPerSpectrum
        cars.SpectraPerPixel = SpectraPerPixel
        cars.XPixels = XPixels
        cars.YPixels = YPixels
        cars.ZPixels = ZPixels
        cars.XYStepSize = XYStepSize
        cars.ZStepSize = ZStepSize
        cars.ExposureTime = ExposureTime
        cars.s1 = s1.astype(float)
        cars.s2 = s2.astype(float)

        return cars


def norm_array(cars):
    cars_norm = (cars - np.min(np.min(cars))) / (np.max(np.max(cars)) - np.min(np.min(cars)))

    return cars_norm


def norm_cars(cars):
    # cars_norms1 = (cars.s1 - np.min(np.min(cars.s1))) / (np.max(np.max(cars.s1)) - np.min(np.min(cars.s1)))
    # cars_norms2 = (cars.s2 - np.min(np.min(cars.s2))) / (np.max(np.max(cars.s2)) - np.min(np.min(cars.s2)))

    a, b, c, d = cars.s1.shape
    cars_norms1 = np.zeros((a, b, c, d))
    cars_norms2 = np.zeros((a, b, c, d))

    for si in range(512):
        cars_norms1[:,:,:,si] = (cars.s1[:,:,:,si] - np.min(np.squeeze(cars.s1[:,:,:,si]))) / \
                                (np.max(np.squeeze(cars.s1[:,:,:,si])) - np.min(np.squeeze(cars.s1[:,:,:,si])))
        cars_norms2[:, :, :, si] = (cars.s2[:, :, :, si] - np.min(np.squeeze(cars.s2[:, :, :, si]))) / \
                                   (np.max(np.squeeze(cars.s2[:, :, :, si])) - np.min(np.squeeze(cars.s2[:, :, :, si])))

    cars_norm = EmptyClass()
    cars_norm.s1 = cars_norms1
    cars_norm.s2 = cars_norms2

    return cars_norm


def cars_plot(cars):
    # TODO Change so that it returns a plt.plot so that it can be called just as plt.figure() /n cars_plot()

    plot = np.squeeze(np.mean(np.mean(cars, 1), 0))

    return plot


def cars_imshow(cars):
    imshow = np.squeeze(np.mean(cars, 3))

    return imshow


def correct_cars(cars):
    [a, b, c, d] = cars.s1.shape

    corrected1 = np.full((a, b, c, d), 0, dtype='float')
    corrected2 = np.full((a, b, c, d), 0, dtype='float')

    for ai in range(0, a):
        for bi in range(0, b):
            if ai >= bi:
                corrected1[ai, (bi - ai), :, :] = cars.s1[bi, ai, :, :]
                corrected2[ai, (bi - ai), :, :] = cars.s2[bi, ai, :, :]
            elif bi > ai:
                corrected1[ai, (bi - ai), :, :] = cars.s1[bi, ai, :, :]
                corrected2[ai, (bi - ai), :, :] = cars.s2[bi, ai, :, :]

    corrected = EmptyClass()
    corrected.s1 = corrected1
    corrected.s2 = corrected2

    return corrected


def remove_background(cars, cars_fl):
    if not (type(cars) is None.__class__ or type(cars_fl) is None.__class__):
        meanbk1 = np.mean(cars_fl.s1)
        meanbk2 = np.mean(cars_fl.s2)

        [a, b, c, d] = cars.s1.shape

        bkfree1 = np.full((a, b, c, d), 0, dtype='float')
        bkfree2 = np.full((a, b, c, d), 0, dtype='float')

        for ai in range(0, a):
            for bi in range(0, b):
                for ci in range(0, c):
                    bkfree1[ai, bi, ci, :] = np.squeeze(np.squeeze(np.squeeze(cars.s1[ai, bi, ci, :]))) - meanbk1
                    bkfree2[ai, bi, ci, :] = np.squeeze(np.squeeze(np.squeeze(cars.s2[ai, bi, ci, :]))) - meanbk2

        bkfree = EmptyClass()
        bkfree.s1 = bkfree1
        bkfree.s2 = bkfree2

        return bkfree

    else:
        return cars


def process_remove_background(cars_file, cars_fl_file):
    cars = process_cars(cars_file)
    cars_fl = process_fl(cars_fl_file)

    cars_bkfree = remove_background(cars, cars_fl)

    return cars_bkfree


def remove_stokespump(cars, stokes, pump):
    if not (type(cars) is None.__class__ or type(stokes) is None.__class__ or type(pump) is None.__class__):
        pumpstokes1 = np.squeeze(np.mean(np.mean(stokes.s1[:, :, :, :], 1), 0)) + np.squeeze(
            np.mean(np.mean(pump.s1[:, :, :, :], 1), 0))
        pumpstokes2 = np.squeeze(np.mean(np.mean(stokes.s2[:, :, :, :], 1), 0)) + np.squeeze(
            np.mean(np.mean(pump.s2[:, :, :, :], 1), 0))

        [a, b, c, d] = cars.s1.shape

        psfree1 = np.full((a, b, c, d), 0, dtype='float')
        psfree2 = np.full((a, b, c, d), 0, dtype='float')

        pumpstokes1 = sg_filt(pumpstokes1, window_length=5, polyorder=3)
        pumpstokes2 = sg_filt(pumpstokes2, window_length=5, polyorder=3)

        max1 = max(pumpstokes1)
        max2 = max(pumpstokes2)

        for ai in range(0, a):
            for bi in range(0, b):
                for ci in range(0, c):
                    psfree1[ai, bi, ci, :] = np.squeeze(cars.s1[ai, bi, ci, :]) - (max(
                        np.squeeze(cars.s1[ai, bi, ci, :])) / max1) * pumpstokes1
                    psfree2[ai, bi, ci, :] = np.squeeze(cars.s2[ai, bi, ci, :]) - (max(
                        np.squeeze(cars.s2[ai, bi, ci, :])) / max2) * pumpstokes2

        psfree = EmptyClass()
        psfree.s1 = psfree1
        psfree.s2 = psfree2

        return psfree

    else:
        return cars


def remove_etaloning(bkfree, psfree, etalon):
    if type(psfree) is None.__class__:
        psfree = bkfree

    etmn1 = np.squeeze(np.mean(np.mean(etalon.s1[:, :, :, :], 1), 0))
    etmn2 = np.squeeze(np.mean(np.mean(etalon.s2[:, :, :, :], 1), 0))

    [a, b, c, d] = bkfree.s1.shape

    bkfrees1 = bkfree.s1
    bkfrees2 = bkfree.s2
    psfrees1 = psfree.s1
    psfrees2 = psfree.s2

    etalonfree1 = np.full((a, b, c, d), 0, dtype='float')
    etalonfree2 = np.full((a, b, c, d), 0, dtype='float')
    etalonSumfree1 = np.full((a, b, c, d), 0, dtype='float')
    etalonSumfree2 = np.full((a, b, c, d), 0, dtype='float')

    sgetmn1 = sg_filt(etmn1, window_length=7, polyorder=3)
    sgetmn2 = sg_filt(etmn2, window_length=7, polyorder=3)

    for ai in range(0, a):
        for bi in range(0, b):
            for ci in range(0, c):
                etalonfree1[ai, bi, ci, :] = (np.squeeze(bkfrees1[ai, bi, ci, :])) / sgetmn1
                etalonfree2[ai, bi, ci, :] = (np.squeeze(bkfrees2[ai, bi, ci, :])) / sgetmn2

                etalonSumfree1[ai, bi, ci, :] = (np.squeeze(psfrees1[ai, bi, ci, :])) / sgetmn1
                etalonSumfree2[ai, bi, ci, :] = (np.squeeze(psfrees2[ai, bi, ci, :])) / sgetmn2

    etfree = EmptyClass()
    etfree.s1 = etalonfree1
    etfree.s2 = etalonfree2

    etSumfree = EmptyClass()
    etSumfree.s1 = etalonSumfree1
    etSumfree.s2 = etalonSumfree2

    return etfree, etSumfree


def xnr_normalise(etfreecars, etSumfreecars):
    nrb_min = input('Min spectrum number(in pixels): ')
    nrb_max = input('Max spectrum number(in pixels): ')
    [a, b, c, d] = etfreecars.s1.shape

    nrb_sum_im = np.squeeze(np.mean(etSumfreecars.s2[:, :, :, int(nrb_min):int(nrb_max)], 3)) - np.squeeze(
        np.mean(etSumfreecars.s1[:, :, :, int(nrb_min):int(nrb_max)], 3))

    nrb_sum_im = nrb_sum_im + np.ceil(np.abs(np.min(np.min(nrb_sum_im))))
    # nrb_sum_im = np.abs(nrb_sum_im)

    xnr_image = np.sqrt(nrb_sum_im)

    xnrfree1 = np.full((a, b, c, d), 0, dtype='float')
    xnrfree2 = np.full((a, b, c, d), 0, dtype='float')

    if c is 1:
        for ai in range(0, a):
            for bi in range(0, b):
                for ci in range(0, c):
                    xnrfree1[ai, bi, ci, :] = (np.squeeze(etfreecars.s1[ai, bi, ci, :])) / xnr_image[ai, bi]
                    xnrfree2[ai, bi, ci, :] = (np.squeeze(etfreecars.s2[ai, bi, ci, :])) / xnr_image[ai, bi]
    elif c > 1:
        for ai in range(0, a):
            for bi in range(0, b):
                for ci in range(0, c):
                    xnrfree1[ai, bi, ci, :] = (np.squeeze(etfreecars.s1[ai, bi, ci, :])) / xnr_image[ai, bi, ci]
                    xnrfree2[ai, bi, ci, :] = (np.squeeze(etfreecars.s2[ai, bi, ci, :])) / xnr_image[ai, bi, ci]

    xnrfree = EmptyClass()
    xnrfree.s1 = xnrfree1
    xnrfree.s2 = xnrfree2

    return xnrfree, xnr_image

    # TODO Need to implement a mini-GUI to allow user selection of the NRB region
