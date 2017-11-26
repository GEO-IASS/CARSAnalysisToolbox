"""
Created on Mon Sep 25th 2017
@author: Priyank Shah
"""

import os
import CARS
import pickle


def create_variable(cars_file, cars_fl_file='none', stokes_file='none', stokes_fl_file='none', pump_file='none',
               pump_fl_file='none', etalon_file='none', etalon_fl_file='none', etstokes_file='none',
               etstokes_fl_file='none', etpump_file='none', etpump_fl_file='none'):

    bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu = do_cars_bk(cars_file, cars_fl_file='none',
                                                                                  stokes_file='none', stokes_fl_file='none',
                                                                                  pump_file='none', pump_fl_file='none',
                                                                                  etalon_file='none', etalon_fl_file='none',
                                                                                  etstokes_file='none', etstokes_fl_file='none',
                                                                                  etpump_file='none', etpump_fl_file='none')

    bkfreecars = do_cars_pca(bkfreecars)
    psfreecars, psfreeetal = do_cars_pumpstokes(bkfreecars, bkfreestokes, bkfreepump, bkfret, bkfretst, bkfretpu)
    etfreecars, etSumfreecars = do_cars_et_normalise(bkfreecars, psfreecars, psfreeetal)
    xnrfreecars = do_cars_xnr_normalise(etfreecars, etSumfreecars)

    file_name = input('Type the file name required:')
    pickle.dump(xnrfreecars, open(file_name + str('.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
