"""
Created on Sat Jul  1 15:15:00 2017
@author: Priyank Shah
"""

import sys

import numpy as np
# import mainwindow
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow

from ..HyperspectralViewer import mainwindow as mainwindow


class DataViewer(QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, cars, parent=None):
        super(DataViewer, self).__init__(parent)
        self.setupUi(self)

        #        layout1 = QHBoxLayout()
        #        layout1.addWidget(self.image_win)
        #        layout1.addWidget(self.spectrum_win)
        #
        #        layout2 = QHBoxLayout()
        #        layout2.addWidget(self.RESET)
        #        layout2.addWidget(self.update_image)
        #        layout2.addWidget(self.update_spectrum)
        #
        #        layout3 = QVBoxLayout()
        #        layout3.addLayout(layout1)
        #        layout3.addLayout(layout2)
        #        layout3.addStretch(1)
        #
        #        self.setLayout(layout3)

        # Initiallising variables to plot the spectrum and image
        self.cars = cars
        self.carsimage = np.squeeze(np.mean(self.cars, 3))
        self.carsspectrum = np.squeeze(np.mean(np.mean(self.cars, 1), 0))

        # Plotting image and spectrum
        pg.ImageView(view=pg.PlotItem())
        self.image_win.setImage(self.carsimage)
        self.spectrum_win.plot(self.carsspectrum)

        # Creating graphics to select subset of image and spectrum
        #        imroi = pg.ROI([0,0], [10,10], invertible=True, scaleSnap=True)
        #        self.image_win.addItem(imroi)
        #        imroi.addScaleHandle([1,1], [0.5,0.5])
        #        imroi.addTranslateHandle([0,0])

        def spec_region_updated(regionItem):
            self.lo, self.hi = regionItem.getRegion()

        self.pgLRI = pg.LinearRegionItem()
        self.spectrum_win.addItem(self.pgLRI)
        self.pgLRI.sigRegionChanged.connect(spec_region_updated)

        # Connecting the image, spectrum and reset buttons to their functions
        self.update_image.clicked.connect(lambda: self.update_image_func(self.cars, self.lo, self.hi))
        self.update_spectrum.clicked.connect(lambda: self.update_spec_func(self.cars))
        self.RESET.clicked.connect(lambda: self.reset_viewer(self.cars))

    def update_spec_func(self, cars):
        # Obtaining coordinates of ROI graphic in the image plot
        image_coordHandles = self.image_win.roi.getState()
        posimage = image_coordHandles['pos']
        sizeimage = image_coordHandles['size']

        posx = int(posimage[0])
        sizex = int(sizeimage[0])
        posy = int(posimage[1])
        sizey = int(sizeimage[1])
        imageXmin = posx
        imageXmax = posx + sizex
        imageYmin = posy
        imageYmax = posy + sizey

        # Updating the spectrum plot based on the updated coordinates
        carsSpec_update = np.squeeze(np.mean(np.mean(self.cars[imageYmin:imageYmax, imageXmin:imageXmax, :, :], 1), 0))
        self.spectrum_win.clear()
        self.spectrum_win.plot(carsSpec_update)

        def spec_region_updated(regionItem):
            self.lo, self.hi = regionItem.getRegion()

        self.pgLRI = pg.LinearRegionItem()
        self.spectrum_win.addItem(self.pgLRI)
        self.pgLRI.sigRegionChanged.connect(spec_region_updated)

    def update_image_func(self, cars, lo, hi):
        # Obtaining coordinates of LinearRegionItem graph in the spectrum plot
        carsImage_update = np.squeeze(np.mean(self.cars[:, :, :, int(lo):int(hi)], 3))
        self.image_win.clear()
        self.image_win.setImage(carsImage_update)

    def reset_viewer(self, cars):
        # Initiallising variables to plot the spectrum and image
        self.cars = cars
        self.carsimage = np.squeeze(np.mean(self.cars, 3))
        self.carsspectrum = np.squeeze(np.mean(np.mean(self.cars, 1), 0))

        # Plotting image and spectrum
        self.image_win.clear()
        self.image_win.setImage(self.carsimage)
        self.spectrum_win.clear()
        self.spectrum_win.plot(self.carsspectrum)

        def spec_region_updated(regionItem):
            self.lo, self.hi = regionItem.getRegion()

        self.pgLRI = pg.LinearRegionItem()
        self.spectrum_win.addItem(self.pgLRI)
        self.pgLRI.sigRegionChanged.connect(spec_region_updated)

        # Connecting the image, spectrum and reset buttons to their functions
        self.update_image.clicked.connect(lambda: self.update_image_func(self.cars, self.lo, self.hi))
        self.update_spectrum.clicked.connect(lambda: self.update_spec_func(self.cars))
        self.RESET.clicked.connect(lambda: self.reset_viewer(self.cars))


def main(cars):
    app = QApplication(sys.argv)
    form = DataViewer(cars)
    form.setWindowTitle('Hyperspectral Data Viewer')
    form.show()
    sys.exit(app.exec_())


# If the file is run directly and not imported, this runs the main function
if __name__ == '__main__':
    # Initiallising the c.elegans cars data as the default data to display
    cars = np.load('default_carsarray.npy')
    main(cars)
