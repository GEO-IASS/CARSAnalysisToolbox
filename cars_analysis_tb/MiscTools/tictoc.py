"""
Created on Fri Jun 30 14:49:40 2017
@author: Priyank Shah

TIC TOC
===============================================================================

    Two functions, tic() and toc() that perform the same operation as that in
    MATLAB. Just place tic() at the start of the where you want to record the
    time, and toc() where you want to the timer to record to.

===============================================================================
"""
import time


def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
