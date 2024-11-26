# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:49:04 2024


"""
import warnings

warnings.filterwarnings("ignore")
from vinecopulas.vinecopula import *
from vinecopulas.marginals import *
import numpy as np
import unittest
import pandas as pd

# %%
class Testvinecopula(unittest.TestCase):
    def setUp(self):
        datapath = "https://raw.githubusercontent.com/VU-IVM/vinecopula/develop/doc/sample_data.csv"  # path to data
        df = pd.read_csv(datapath)
        self.U = pseudodata(np.array(df)[:, :-1])
        nan = np.nan
        self.M_R = np.array(
            [
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                [3.0, 2.0, 5.0, 4.0, 4.0, nan],
                [2.0, 5.0, 4.0, 0.0, nan, nan],
                [5.0, 4.0, 2.0, nan, nan, nan],
                [4.0, 3.0, nan, nan, nan, nan],
                [1.0, nan, nan, nan, nan, nan],
            ]
        )
        self.P_R = np.array(
            [
                [
                    5.667415060610504,
                    4.809887154028018,
                    4.742610409927173,
                    5.329052634720821,
                    6.365954831464865,
                    nan,
                ],
                [
                    list([-0.6512734168403267, 5.953996539895858]),
                    1.155319323794073,
                    0.1056255559540575,
                    list([0.3900965721838106, 6.344879386256914]),
                    nan,
                    nan,
                ],
                [
                    list([-0.40578880593225847, 5.521243696841605]),
                    1.0500821290068711,
                    0.08990469323285986,
                    nan,
                    nan,
                    nan,
                ],
                [
                    list([0.2172541619061329, 18.453691725212337]),
                    1.587083876681365,
                    nan,
                    nan,
                    nan,
                    nan,
                ],
                [
                    list([0.05131396326384129, 30.722615332724796]),
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                ],
                [nan, nan, nan, nan, nan, nan],
            ],
            dtype=object,
        )
        self.C_R = np.array(
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, nan],
                [15.0, 3.0, 1.0, 15.0, nan, nan],
                [15.0, 4.0, 9.0, nan, nan, nan],
                [15.0, 10.0, nan, nan, nan, nan],
                [15.0, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        )
        self.M_D = np.array(
            [
                [2.0, 0.0, 5.0, 4.0, 5.0, 5.0],
                [0.0, 5.0, 4.0, 5.0, 4.0, nan],
                [5.0, 4.0, 3.0, 3.0, nan, nan],
                [4.0, 3.0, 0.0, nan, nan, nan],
                [3.0, 2.0, nan, nan, nan, nan],
                [1.0, nan, nan, nan, nan, nan],
            ]
        )
        self.P_D = np.array(
            [
                [
                    3.5133699681513915,
                    4.742610409927173,
                    5.329052634720821,
                    4.093431936485096,
                    6.365954831464865,
                    nan,
                ],
                [
                    list([0.7761434164482084, 6.701427960238227]),
                    0.1056255559540575,
                    list([0.3900965721838106, 6.344879386256914]),
                    list([0.16733909434684788, 10.7269854361358]),
                    nan,
                    nan,
                ],
                [
                    0.12978185414142449,
                    0.08990469323285986,
                    list([0.5353314452187353, 16.347989651517267]),
                    nan,
                    nan,
                    nan,
                ],
                [-0.8757255275146373, 1.1429781801124308, nan, nan, nan, nan],
                [
                    list([-0.6825574708843178, 5.342550524934684]),
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                ],
                [nan, nan, nan, nan, nan, nan],
            ],
            dtype=object,
        )
        self.C_D = np.array(
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, nan],
                [15.0, 1.0, 15.0, 15.0, nan, nan],
                [6.0, 9.0, 15.0, nan, nan, nan],
                [10.0, 5.0, nan, nan, nan, nan],
                [15.0, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        )
        self.M_C = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 5.0, 5.0],
                [3.0, 3.0, 3.0, 5.0, 0.0, nan],
                [5.0, 5.0, 5.0, 3.0, nan, nan],
                [1.0, 4.0, 4.0, nan, nan, nan],
                [4.0, 1.0, nan, nan, nan, nan],
                [2.0, nan, nan, nan, nan, nan],
            ]
        )
        self.P_C = np.array(
            [
                [
                    4.742610409927173,
                    5.667415060610504,
                    5.186696320701982,
                    4.809887154028018,
                    5.329052634720821,
                    nan,
                ],
                [
                    1.155319323794073,
                    list([-0.6512734168403267, 5.953996539895858]),
                    1.6047944400112235,
                    1.0501700414768074,
                    nan,
                    nan,
                ],
                [
                    0.10074381405975691,
                    list([0.1637157864473392, 18.148998646970544]),
                    list([0.6321675900633315, 6.631581061709736]),
                    nan,
                    nan,
                    nan,
                ],
                [list([-0.41682583386878064, 5.365613356884372]), 0.07785323645884598, nan, nan, nan, nan],
                [-0.037650974919099515, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ],
            dtype=object,
        )
        self.C_C = np.array(
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, nan],
                [3.0, 15.0, 10.0, 4.0, nan, nan],
                [1.0, 15.0, 15.0, nan, nan, nan],
                [15.0, 6.0, nan, nan, nan, nan],
                [9.0, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ]
        )

    def test_vinecopR(self):
        cops = list(range(1, 16))  # all copulas
        M, P, C = fit_vinecop(self.U, cops, vine="R", printing=False)
        assert np.isclose(self.M_R, M, rtol=0.05, equal_nan=True).all()
        assert np.isclose(self.C_R, C, rtol=0.05, equal_nan=True).all()
        assert self.P_R in P

    def test_vinecopD(self):
        cops = list(range(1, 16))  # all copulas
        M, P, C = fit_vinecop(self.U, cops, vine="D", printing=False)
        assert np.isclose(self.M_D, M, rtol=0.05, equal_nan=True).all()
        assert np.isclose(self.C_D, C, rtol=0.05, equal_nan=True).all()
        assert self.P_D in P

    def test_vinecopC(self):
        cops = list(range(1, 16))  # all copulas
        M, P, C = fit_vinecop(self.U, cops, vine="C", printing=False)
        assert np.isclose(self.M_C, M, rtol=0.05, equal_nan=True).all()
        assert np.isclose(self.C_C, C, rtol=0.05, equal_nan=True).all()
        assert self.P_C in P

    def test_vinecopstructure(self):
        cops = list(range(1, 16))  # all copulas
        P, C = fit_vinecopstructure(self.U, cops, self.M_C)
        assert np.isclose(self.C_C, C, rtol=0.05, equal_nan=True).all()
        assert self.P_C in P

    def test_samplecop(self):
        np.random.seed(10)
        ur_expected = np.array(
            [
                [
                    0.27309758, 0.459142  , 0.30827003, 0.14359465, 0.23025509,
                            0.22479665,
                ],
                [
                    0.87293178, 0.78494858, 0.76770503, 0.93681427, 0.95934152,
                     0.95339335,
                ],
                [
                    0.34006522, 0.18519631, 0.43717586, 0.35082454, 0.33830778,
                     0.29187607,
                ],
                [
                   0.55197026, 0.60908594, 0.58327573, 0.63115738, 0.64226717,
                    0.67413362,
                ],
                [
                    0.62078409, 0.60270035, 0.65357772, 0.61645537, 0.63675281,
                     0.60103895,
                ],
                [0.23607129, 0.28591365, 0.39315093, 0.20390582, 0.2238174 ,
                 0.30070006],
            ]
        )
        ur_result = sample_vinecop(self.M_R, self.P_R, self.C_R, 6)
        assert np.isclose(ur_expected, ur_result, rtol=0.05, equal_nan=True).all()

    def test_vincopconditionalsample(self):
        np.random.seed(10)
        ur_expected = np.array(
            [
                [0.92455016, 0.98003593, 0.92675129, 0.69882321, 0.9, 0.9],
                [0.92623137, 0.94499478, 0.83619152, 0.86607092, 0.9, 0.9],
                [0.9635065, 0.95650604, 0.9643929, 0.83547459, 0.9, 0.9],
                [0.90597386, 0.66753704, 0.94219826, 0.88281989, 0.9, 0.9],
                [0.91975774, 0.93327197, 0.96929214, 0.8292089, 0.9, 0.9],
                [0.9142449, 0.94302835, 0.87124292, 0.82148805, 0.9, 0.9],
            ]
        )
        ur_result = sample_vinecopconditional(self.M_R, self.P_R, self.C_R, 6, [0.9, 0.9])
        assert np.isclose(ur_expected, ur_result, rtol=0.05, equal_nan=True).all()

    def test_samplingorder(self):
        so = samplingorder(self.M_R)
        expected_last_so = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0]
        assert np.isclose(so[-1], expected_last_so, rtol=0.05, equal_nan=True).all()
        assert len(so) == 32

    def test_samplingmatrix(self):
        sorder = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0]
        M, P, C = samplingmatrix(self.M_R, self.C_R, self.P_R, sorder)
        so = list(np.diag(M[::-1])[::-1])
        assert sorder == so

    def test_conditionalvine1(self):
        cops = list(range(1, 16))  # all copulas
        vint = [2, 0]
        MR, PR, CR = fit_conditionalvine(
            self.U, vint, cops, vine="R", condition=1, printing=False
        )
        soR = list(np.diag(MR[::-1])[::-1])
        MD, PD, CD = fit_conditionalvine(
            self.U, vint, cops, vine="D", condition=1, printing=False
        )
        soD = list(np.diag(MD[::-1])[::-1])
        MC, PC, CC = fit_conditionalvine(
            self.U, vint, cops, vine="C", condition=1, printing=False
        )
        soC = list(np.diag(MC[::-1])[::-1])
        assert 2 in soR[-2:]
        assert 0 in soR[-2:]
        assert 2 in soD[-2:]
        assert 0 in soD[-2:]
        assert 2 in soC[-2:]
        assert 0 in soC[-2:]

    def test_conditionalvine2(self):
        cops = list(range(1, 16))  # all copulas
        vint = [2, 0]
        MR, PR, CR = fit_conditionalvine(
            self.U, vint, cops, vine="R", condition=2, printing=False
        )
        soR = list(np.diag(MR[::-1])[::-1])
        MD, PD, CD = fit_conditionalvine(
            self.U, vint, cops, vine="D", condition=2, printing=False
        )
        soD = list(np.diag(MD[::-1])[::-1])
        MC, PC, CC = fit_conditionalvine(
            self.U, vint, cops, vine="C", condition=2, printing=False
        )
        soC = list(np.diag(MC[::-1])[::-1])
        assert 2 in soR[:2]
        assert 0 in soR[:2]
        assert 2 in soD[:2]
        assert 0 in soD[:2]
        assert 2 in soC[:2]
        assert 0 in soC[:2]

# %%
if __name__ == "__main__":
    unittest.main()
