# -*- coding: utf-8 -*-

#################################################################
# File        : napari_tools.py
# Version     : 0.0.8
# Author      : sebi06
# Date        : 24.08.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################


from __future__ import annotations
try:
    import napari
except ModuleNotFoundError as error:
    print(error.__class__.__name__ + ": " + error.name)

from PyQt5.QtWidgets import (

    QHBoxLayout,
    QVBoxLayout,
    QFileSystemModel,
    QFileDialog,
    QTreeView,
    QDialogButtonBox,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QAbstractItemView,
    QComboBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QGridLayout

)

from PyQt5.QtCore import Qt, QDir, QSortFilterProxyModel
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont
#import czi_metadata as czimd
from czitools import pylibczirw_metadata as czimd
import misc
import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union


class TableWidget(QWidget):

    def __init__(self) -> None:

        super(QWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.mdtable = QTableWidget()
        self.layout.addWidget(self.mdtable)
        self.mdtable.setShowGrid(True)
        self.mdtable.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.mdtable.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)

    def update_metadata(self, md_dict: Dict) -> None:

        # number of rows is set to number of metadata entries
        row_count = len(md_dict)
        col_count = 2
        self.mdtable.setColumnCount(col_count)
        self.mdtable.setRowCount(row_count)

        row = 0

        # update the table with the entries from metadata dictionary
        for key, value in md_dict.items():
            newkey = QTableWidgetItem(key)
            self.mdtable.setItem(row, 0, newkey)
            newvalue = QTableWidgetItem(str(value))
            self.mdtable.setItem(row, 1, newvalue)
            row += 1

        # fit columns to content
        self.mdtable.resizeColumnsToContents()

    def update_style(self) -> None:

        # define font size and type
        fnt = QFont()
        fnt.setPointSize(11)
        fnt.setBold(True)
        fnt.setFamily("Arial")

        # update both header items
        fc = (25, 25, 25)
        item1 = QtWidgets.QTableWidgetItem("Parameter")
        #item1.setForeground(QtGui.QColor(25, 25, 25))
        item1.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(0, item1)

        item2 = QtWidgets.QTableWidgetItem("Value")
        #item2.setForeground(QtGui.QColor(25, 25, 25))
        item2.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(1, item2)


def show(viewer: Any, array: np.ndarray, metadata: czimd.CziMetadata,
         blending: str = "additive",
         contrast: str = "calc",
         gamma: float = 0.85,
         add_mdtable: bool = True,
         name_sliders: bool = False) -> List:
    """ Display the multidimensional array inside the Napari viewer.
    Optionally the CziMetadata class will be used show a table with the metadata.
    Every channel will be added as a new layer to the viewer.

    :param viewer: Napari viewer object
    :param array: multi-dimensional array containing the pixel data
    :param metadata: CziMetadata class
    :param blending: blending mode for viewer
    :param contrast: method to be used to calculate an appropriate display scaling.
    - "calc" : real min & max calculation (might be slow) be calculated (slow)
    - "napari_auto" : let Napari figure out a display scaling. Will look in the center of an image !
    - "from_czi" : use the display scaling from ZEN stored inside the CZI metadata
    :param gamma: gamma value for the Viewer for all layers
    :param add_mdtable: option to show the CziMetadata as a table widget
    :param name_sliders: option to use the dimension letters as slider labels for the viewer
    :return:
    """

    # check if contrast mode
    if contrast not in ["calc", "napari_auto", "from_czi"]:
        print(contrast, "is not valid contrast method. Use napari_auto instead.")
        contrast = "from_czi"

    # create empty list for the napari layers
    napari_layers = []

    # create scalefactor with all ones
    scalefactors = [1.0] * len(array.shape)

    # modify the tuple for the scales for napari
    scalefactors[metadata.dim_order["Z"]] = metadata.scale.ratio["zx"]

    # remove C dimension from scalefactor
    scalefactors_ch = scalefactors.copy()
    del scalefactors_ch[metadata.dim_order["C"]]

    # add Qt widget for metadata
    if add_mdtable:

        # create widget for the metadata
        mdbrowser = TableWidget()
        viewer.window.add_dock_widget(mdbrowser,
                                      name="mdbrowser",
                                      area="right")

        # add the metadata and adapt the table
        mdbrowser.update_metadata(czimd.create_mdict_complete(metadata))
        mdbrowser.update_style()

    # add all channels as individual layers
    if metadata.dims.SizeC is None:
        sizeC = 1
    else:
        sizeC = metadata.dims.SizeC

    # loop over all channels and add them as layers
    for ch in range(sizeC):

        try:
            # get the channel name
            chname = metadata.channelinfo.names[ch]
        except KeyError as e:
            print(e)
            # or use CH1 etc. as string for the name
            chname = "CH" + str(ch + 1)

        # cut out channel
        if metadata.dims.SizeC is not None:
            channel = misc.slicedim(array, ch, metadata.dim_order["C"])
        if metadata.dims.SizeC is None:
            channel = array

        # actually show the image array
        print("Adding Channel  :", chname)
        print("Shape Channel   :", ch, channel.shape)
        print("Scaling Factors :", scalefactors_ch)

        if contrast == "calc":
            # really calculate the min and max values - might be slow
            sc = misc.calc_scaling(channel, corr_min=1.1, corr_max=0.9)
            print("Calculated Display Scaling (min & max)", sc)

            # add channel to napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         contrast_limits=sc,
                                         blending=blending,
                                         gamma=gamma)

        if contrast == "napari_auto":
            # let Napari figure out what the best display scaling is
            # Attention: It will measure in the center of the image !!!

            # add channel to napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         blending=blending,
                                         gamma=gamma)
        if contrast == "from_czi":
            # guess an appropriate scaling from the display setting embedded in the CZI
            lower = np.round(metadata.channelinfo.clims[ch][0] * metadata.maxrange, 0)
            higher = np.round(metadata.channelinfo.clims[ch][1] * metadata.maxrange, 0)

            # simple validity check
            if lower >= higher:
                print("Fancy Display Scaling detected. Use Defaults")
                lower = 0
                higher = np.round(metadata.maxrange * 0.25, 0)

            print("Display Scaling from CZI for CH:", ch, "Min-Max", lower, higher)

            # add channel to Napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         contrast_limits=[lower, higher],
                                         blending=blending,
                                         gamma=gamma)

        # append the current layer
        napari_layers.append(new_layer)

    if name_sliders:

        print("Rename Sliders based on the Dimension String ....")

        # get the label of the sliders (as a tuple) ad rename it
        sliderlabels = rename_sliders(viewer.dims.axis_labels, metadata.dim_order)
        viewer.dims.axis_labels = sliderlabels

    return napari_layers


def rename_sliders(sliders: Tuple, dim_order: Dict) -> Tuple:
    """rename the sliders inside the Napari viewer based on the metadata

    :param sliders: labels of sliders from viewer
    :param dim_order: dictionary indication the dimension string and its
    position inside the array
    :return: tuple with renamed sliders
    """

    # update the labels with the correct dimension strings
    slidernames = ["B", "H", "V", "M", "S", "T", "Z", "Y", "X", "A"]

    # convert to list()
    tmp_sliders = list(sliders)

    for s in slidernames:
        try:
            if dim_order[s] >= 0:

                # assign the dimension labels
                tmp_sliders[dim_order[s]] = s

                # convert back to tuple
                sliders = tuple(tmp_sliders)
        except KeyError:
            print("No", s, "Dimension found")

    return sliders
