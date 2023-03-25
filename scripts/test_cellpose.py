# -*- coding: utf-8 -*-

#################################################################
# File        : test_cellpose.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#################################################################

from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
import os
import sys
from pathlib import Path
import apeer_seg_cellpose as ascp
from dataclasses import dataclass, field


@dataclass
class cp_params:
    filepath: str
    modelbasedir: str = field(init=False)
    seg_nuc: bool = field(init=False)
    chindex_nuc: int = field(init=False)
    nuc_model: str = field(init=False)
    diameter_nuc: int = field(init=False)
    seg_cyto: bool = field(init=False)
    chindex_cyto: int = field(init=False)
    cyto_model: str = field(init=False)
    diameter_cyto: int = field(init=False)
    cellprob_threshold: float = field(init=False)
    tile: bool = field(init=False)
    tile_overlap: float = field(init=False)
    do_area_filter: bool = field(init=False)
    minsize_obj: int = field(init=False)
    maxsize_obj: int = field(init=False)
    do_clear_borders: bool = field(init=False)
    flatten_labels: bool = field(init=False)
    use_gpu: bool = field(init=False)

    def __post_init__(self):

        self.modelbasedir = ".cellpose"
        self.seg_nuc = True
        self.chindex_nuc = 0
        self.nuc_model = "2d_nuclei_fluo"
        self.diameter_nuc = 17
        self.seg_cyto = False
        self.chindex_cyto = 1
        self.cyto_model = "2d_nuclei_cyto"
        self.diameter_cyto = 30
        self.cellprob_threshold = 0.1
        self.tile = True
        self.tile_overlap = 0.1
        self.do_area_filter = True
        self.minsize_obj = 20
        self.maxsize_obj = 100000
        self.do_clear_borders = True
        self.flatten_labels = False
        self.use_gpu = True


def test_cellpose():

    basedir = Path(__file__).resolve().parents[0]
    sys.path.append(basedir)
    filename = r'input/S=1_CH=2.czi'
    filepath = os.path.join(basedir, filename)
    params = cp_params(filepath)

    outputs = ascp.execute(params.filepath,
                           params.modelbasedir,
                           params.seg_nuc,
                           params.chindex_nuc,
                           params.nuc_model,
                           params.diameter_nuc,
                           params.seg_cyto,
                           params.chindex_cyto,
                           params.cyto_model,
                           params.diameter_cyto,
                           params.cellprob_threshold,
                           params.tile,
                           params.tile_overlap,
                           params.do_area_filter,
                           params.minsize_obj,
                           params.maxsize_obj,
                           params.do_clear_borders,
                           params.flatten_labels,
                           params.use_gpu
                           )

    assert (outputs[0] == "A01sm_segCP.czi")
    assert (outputs[1] == "A01sm_obj_nuc.csv")
    assert (outputs[2] == "A01sm_objparams_nuc.csv")
    assert (outputs[3] is None)
    assert (outputs[4] is None)

    # remove results from test run
    for file in outputs:
        if file is not None:
            os.remove(file)


test_cellpose()
