# -*- coding: utf-8 -*-

#################################################################
# File        : test_stardist.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#################################################################

from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
import os
import sys
from pathlib import Path
import apeer_nucseg_stardist as asd
import pandas as pd

basedir = Path(__file__).resolve().parents[0]

def test_stardist_fluo():

    sys.path.append(basedir)

    filename = r'input/A01.czi'
    modeltype = '2d_versatile_fluo'

    # get the CZI filepath
    filepath = os.path.join(basedir, filename)
    print("Filepath:", filepath)
    print("File exists :", os.path.exists(filepath))

    outputs = asd.execute(filename,
                          chindex_nucleus=0,
                          sd_modelbasedir='stardist_models',
                          sd_modelfolder=modeltype,
                          prob_th=0.5,
                          ov_th=0.3,
                          do_area_filter=True,
                          minsize_nuc=200,
                          maxsize_nuc=1000,
                          # blocksize=4096,
                          min_overlap=128,
                          n_tiles=3,
                          norm_pmin=1,
                          norm_pmax=99.8,
                          norm_clip=False,
                          verbose=True,
                          do_clear_borders=True,
                          flatten_labels=True,
                          normalize_whole=True)

    df_obj = pd.read_csv(outputs[0])
    df_params = pd.read_csv(outputs[1])

    assert (outputs[0] == "A01_obj.csv")
    assert (outputs[1] == "A01_objparams.csv")
    assert (outputs[2] == "A01_segSD.czi")

    # remove results from test run
    for file in outputs:
        os.remove(file)


def test_stardist_he():

    sys.path.append(basedir)

    filename = r'input/Tumor_H+E_small2.czi'
    modeltype = "2d_versatile_he"

    # get the CZI filepath
    filepath = os.path.join(basedir, filename)
    print("Filepath:", filepath)
    print("File exists :", os.path.exists(filepath))

    outputs = asd.execute(filename,
                          chindex_nucleus=0,
                          sd_modelbasedir='stardist_models',
                          sd_modelfolder=modeltype,
                          prob_th=0.5,
                          ov_th=0.3,
                          do_area_filter=True,
                          minsize_nuc=200,
                          maxsize_nuc=1000,
                          # blocksize=4096,
                          min_overlap=128,
                          n_tiles=None,
                          norm_pmin=1,
                          norm_pmax=99.8,
                          norm_clip=False,
                          verbose=True,
                          do_clear_borders=True,
                          flatten_labels=True,
                          normalize_whole=True)

    df_obj = pd.read_csv(outputs[0])
    df_params = pd.read_csv(outputs[1])

    assert (outputs[0] == "Tumor_H+E_small2_obj.csv")
    assert (outputs[1] == "Tumor_H+E_small2_objparams.csv")
    assert (outputs[2] == "Tumor_H+E_small2_segSD.czi")

    # remove results from test run
    for file in outputs:
        os.remove(file)


def test_stardist_db():

    sys.path.append(basedir)

    filename = r'input/Nuclei_H3258_3x2_ZSTD_8bit.czi'
    modeltype = '2d_dsb2018_fluo'

    # get the CZI filepath
    filepath = os.path.join(basedir, filename)
    print("Filepath:", filepath)
    print("File exists :", os.path.exists(filepath))

    outputs = asd.execute(filename,
                          chindex_nucleus=0,
                          sd_modelbasedir='stardist_models',
                          sd_modelfolder=modeltype,
                          prob_th=0.5,
                          ov_th=0.3,
                          do_area_filter=True,
                          minsize_nuc=100,
                          maxsize_nuc=5000,
                          # blocksize=4096,
                          min_overlap=128,
                          n_tiles=3,
                          norm_pmin=1,
                          norm_pmax=99.8,
                          norm_clip=False,
                          verbose=False,
                          do_clear_borders=False,
                          flatten_labels=True,
                          normalize_whole=True)

    df_obj = pd.read_csv(outputs[0])
    df_params = pd.read_csv(outputs[1])

    assert (outputs[0] == "Nuclei_H3258_3x2_ZSTD_8bit_obj.csv")
    assert (outputs[1] == "Nuclei_H3258_3x2_ZSTD_8bit_objparams.csv")
    assert (outputs[2] == "Nuclei_H3258_3x2_ZSTD_8bit_segSD.czi")

    # remove results from test run
    for file in outputs:
        os.remove(file)


test_stardist_fluo()
test_stardist_he()
test_stardist_db()
