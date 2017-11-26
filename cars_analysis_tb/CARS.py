"""
Created on Fri Jun 23 19:37:25 2017
@author: Priyank Shah
"""

from .preprocess.denoise import cars_pca
from .preprocess.process import process_cars, process_fl, remove_background, remove_stokespump
from .preprocess.process import remove_etaloning, xnr_normalise, correct_cars
from .HyperspectralViewer.hyperspectral_viewer import main as hyperview
from .MiscTools.tictoc import tic, toc
import pickle

tic()

# TODO Change files around so that you only enter files when they need to be used

def cars_main():
    cars_file = '../2015-08-celegansIV-daf22/data/C Elegans IV daf22 90x90at500nm 150ms a1.cars'
    cars_fl_file = '../2015-08-celegansIV-daf22/data/C Elegans IV daf22 90x90at500nm 150ms a1_fl_traces.cars'
    stokes_file = 'none'
    stokes_fl_file = 'none'
    pump_file = 'none'
    pump_fl_file = 'none'
    etalon_file = '../2015-08-celegansIV-daf22/data/C Elegans IV glass 30x1at500nm 600ms a2.cars'
    etalon_fl_file = '../2015-08-celegansIV-daf22/data/C Elegans IV glass 30x1at500nm 600ms a2_fl_traces.cars'
    etstokes_file = 'none'
    etstokes_fl_file = 'none'
    etpump_file = 'none'
    etpump_fl_file = 'none'

    bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu = do_cars_bk(cars_file, cars_fl_file, stokes_file,
                                                                                  stokes_fl_file, pump_file,
                                                                                  pump_fl_file, etalon_file,
                                                                                  etalon_fl_file, etstokes_file,
                                                                                  etstokes_fl_file, etpump_file,
                                                                                  etpump_fl_file)
    bkfreecars = do_cars_pca(bkfreecars)
    psfreecars, psfreeetal = do_cars_pumpstokes(bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu)
    etfreecars, etSumfreecars = do_cars_et_normalise(bkfreecars, psfreecars, psfreeetal)
    xnrfreecars = do_cars_xnr_normalise(etfreecars, etSumfreecars)

    do_cars_hyper_plot(xnrfreecars)


def remove_bk(cars_file, cars_fl_file):
    cars = process_cars(cars_file)
    cars_fl = process_fl(cars_fl_file)

    cars_bkfree = remove_background(cars, cars_fl)

    return cars_bkfree


def do_cars_bk(cars_file, cars_fl_file='none', stokes_file='none', stokes_fl_file='none', pump_file='none',
               pump_fl_file='none', etalon_file='none', etalon_fl_file='none', etstokes_file='none',
               etstokes_fl_file='none', etpump_file='none', etpump_fl_file='none'):

    # Reading in the cars data files(including stokes and pump only data)
    cars = process_cars(cars_file)
    cars_fl = process_fl(cars_fl_file)
    stokes = process_cars(stokes_file)
    stokes_fl = process_fl(stokes_fl_file)
    pump = process_cars(pump_file)
    pump_fl = process_fl(pump_fl_file)

    # Reading in the etalon data
    etalon = process_cars(etalon_file)
    etalon_fl = process_fl(etalon_fl_file)
    etstokes = process_cars(etstokes_file)
    etstokes_fl = process_fl(etstokes_fl_file)
    etpump = process_cars(etpump_file)
    etpump_fl = process_fl(etpump_fl_file)

    # Removing the dark-field background data from the cars data
    bkfreecars = remove_background(cars, cars_fl)
    bkfreestokes = remove_background(stokes, stokes_fl)
    bkfreepump = remove_background(pump, pump_fl)
    bkfret = remove_background(etalon, etalon_fl)
    bkfretst = remove_background(etstokes, etstokes_fl)
    bkfretpu = remove_background(etpump, etpump_fl)

    # If needed, this function will correct the order of the cars data. Change the relevent
    # function in the file as needed
    # bkfreecars = correct_cars(bkfreecars)

    return bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu


def do_cars_pca(bkfreecars):
    # Performing PCA on the data, preceded by the generalised Anscombe transformation and
    # proceeded by the inverse exact Anscombe transformation
    bkfreecars2 = cars_pca(bkfreecars, 0)

    return bkfreecars2


def do_cars_pumpstokes(bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu):
    # Removing the pump- and Stokes- only readings from the cars and etalon data
    psfreecars = remove_stokespump(bkfreecars, bkfreestokes, bkfreepump)
    psfreeetal = remove_stokespump(bkfret, bkfretst, bkfretpu)

    return psfreecars, psfreeetal


def do_cars_et_normalise(bkfreecars, psfreecars, psfreeetal):
    # Normalising the cars data with the etalon data
    [etfreecars, etSumfreecars] = remove_etaloning(bkfreecars, psfreecars, psfreeetal)

    return etfreecars, etSumfreecars


def do_cars_xnr_normalise(etfreecars, etSumfreecars):
    # Spatially normalise the resultant cars data by Xnr
    xnrfreecars = xnr_normalise(etfreecars, etSumfreecars)

    return xnrfreecars


def do_cars_hyper_plot(xnrfreecars):
    xnrfreecars1 = xnrfreecars.s1
    xnrfreecars2 = xnrfreecars.s2
    diff_cars = xnrfreecars2 - xnrfreecars1

    hyperview(diff_cars)


if __name__ == '__main__':
    cars_main()

toc()

