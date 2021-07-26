# -*- coding: utf-8 -*-

#################################################################
# File        : napari_tools.py
# Version     : 0.0.5
# Author      : sebi06
# Date        : 19.07.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################


from __future__ import annotations
try:
    import napari
except ModuleNotFoundError as error:
    print(error.__class__.__name__ + ": " + error.msg)

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
from czitools import czi_metadata as czimd
import zarr
import dask
import dask.array as da
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from nptyping import Int, UInt, Float


class TableWidget(QWidget):

    def __init__(self) -> TableWidget:

        super(QWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.mdtable = QTableWidget()
        self.layout.addWidget(self.mdtable)
        self.mdtable.setShowGrid(True)
        self.mdtable.setHorizontalHeaderLabels(['Parameter', 'Value'])
        header = self.mdtable.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)

    def update_metadata(self, md_dict: Dict) -> TableWidget:

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

    def update_style(self) -> TableWidget:

        # define font size and type
        fnt = QFont()
        fnt.setPointSize(11)
        fnt.setBold(True)
        fnt.setFamily('Arial')

        # update both header items
        fc = (25, 25, 25)
        item1 = QtWidgets.QTableWidgetItem('Parameter')
        #item1.setForeground(QtGui.QColor(25, 25, 25))
        item1.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(0, item1)

        item2 = QtWidgets.QTableWidgetItem('Value')
        #item2.setForeground(QtGui.QColor(25, 25, 25))
        item2.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(1, item2)


def show_napari(viewer: Any, array: np.ndarray, metadata: czimd.CziMetadata,
                blending: str = 'additive',
                calc_contrast: bool = False,
                auto_contrast: bool = False,
                gamma: Int = 0.85,
                add_mdtable: bool = True,
                rename_sliders: bool = False):

    # create list for the napari layers
    napari_layers = []

    # create scalefcator with all ones
    scalefactors = [1.0] * len(array.shape)

    # modify the tuple for the scales for napari
    scalefactors[metadata.dim_order['Z']] = metadata.scale.ratio['zx']

    # remove C dimension from scalefactor
    scalefactors_ch = scalefactors.copy()
    del scalefactors_ch[metadata.dim_order['C']]

    # add widget for metadata
    if add_mdtable:

        # create widget for the metadata
        mdbrowser = TableWidget()

        viewer.window.add_dock_widget(mdbrowser,
                                      name='mdbrowser',
                                      area='right')

        # add the metadata and adapt the table display
        mdbrowser.update_metadata(czimd.create_metadata_dict(metadata))
        mdbrowser.update_style()

    # add all channels as layers
    if metadata.dims.SizeC is None:
        sizeC = 1
    else:
        sizeC = metadata.dims.SizeC

    for ch in range(sizeC):

        try:
            # get the channel name
            chname = metadata.channelinfo.names[ch]
        except KeyError as e:
            print(e)
            # or use CH1 etc. as string for the name
            chname = 'CH' + str(ch + 1)

        # cut out channel
        if metadata.dims.SizeC is not None:
            channel = slicedim(array, ch, metadata.dim_order['C'])
        if metadata.dims.SizeC is None:
            channel = array

        # actually show the image array
        print('Adding Channel  :', chname)
        print('Shape Channel   :', ch, channel.shape)
        print('Scaling Factors :', scalefactors_ch)

        if calc_contrast:
            # really calculate the min and max values - might be slow
            sc = calc_scaling(channel, corr_max=0.5)
            print('Display Scaling', sc)

            # add channel to napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         contrast_limits=sc,
                                         blending=blending,
                                         gamma=gamma)

        if not calc_contrast:
            # let napari figure out what the best display scaling is
            # Attention: It will measure in the center of the image
            if not auto_contrast:
                # add channel to napari viewer
                new_layer = viewer.add_image(channel,
                                             name=chname,
                                             scale=scalefactors_ch,
                                             blending=blending,
                                             gamma=gamma)
            if auto_contrast:
                # guess an appropriate scaling from the display setting embedded in the CZI
                lower = np.round(metadata.channelinfo.clims[ch][0] * metadata.maxrange, 0)
                higher = np.round(metadata.channelinfo.clims[ch][1] * metadata.maxrange, 0)

                # add channel to napari viewer
                new_layer = viewer.add_image(channel,
                                             name=chname,
                                             scale=scalefactors_ch,
                                             contrast_limits=[lower, higher],
                                             blending=blending,
                                             gamma=gamma)

        napari_layers.append(new_layer)

    if rename_sliders:

        print('Rename Sliders based on the Dimension String ....')

        # get the label of the sliders (as a tuple) ad rename it
        sliderlabels = napari_rename_sliders(viewer.dims.axis_labels, metadata.dim_order)

        viewer.dims.axis_labels = sliderlabels

    return napari_layers


def napari_rename_sliders(sliders: Tuple, dim_order: Dict) -> Tuple:

    # update the labels with the correct dimension strings
    slidernames = ['B', 'H', 'V', 'M', 'S', 'T', 'Z']

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
            print('No', s, 'Dimension found')

    return sliders


def slicedim(array: Union[np.ndarry, dask.array, zarr.Array],
             dimindex: Int,
             posdim: Int) -> np.ndarray:
    """slice out a specific channel without (!) dropping the dimension
    of the array to conserve the dimorder string
    this should work for Numpy.Array, Dask and ZARR ...

    if posdim == 0:
        array_sliced = array[dimindex:dimindex + 1, ...]
    if posdim == 1:
        array_sliced = array[:, dimindex:dimindex + 1, ...]
    if posdim == 2:
        array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    if posdim == 3:
        array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    if posdim == 4:
        array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    if posdim == 5:
        array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]
    """

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(data: np.ndarray,
                 corr_min: float = 1.0,
                 offset_min: Int = 0,
                 corr_max: Float = 0.85,
                 offset_max: Int = 0) -> List[Int, Int]:
    """Calculate the scaling for better display

    :param data: Calculate min / max scaling
    :type data: Numpy.Array or dask.array or zarr.array
    :param corr_min: correction factor for minvalue, defaults to 1.0
    :type corr_min: float, optional
    :param offset_min: offset for min value, defaults to 0
    :type offset_min: int, optional
    :param corr_max: correction factor for max value, defaults to 0.85
    :type corr_max: float, optional
    :param offset_max: offset for max value, defaults to 0
    :type offset_max: int, optional
    :return: list with [minvalue, maxvalue]
    :rtype: list
    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # use dask.compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data, initial=0), np.max(data, initial=0)

    end = time.time()

    minvalue = np.round((minvalue + offset_min) * corr_min, 0)
    maxvalue = np.round((maxvalue + offset_max) * corr_max, 0)

    print('Scaling:', minvalue, maxvalue)
    print('Calculation of Min-Max [s] : ', end - start)

    return [minvalue, maxvalue]
