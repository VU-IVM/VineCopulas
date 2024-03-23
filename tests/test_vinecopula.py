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
                    1.149978331028409,
                    0.1022309367943847,
                    list([0.3900965721838106, 6.344879386256914]),
                    nan,
                    nan,
                ],
                [
                    -2.4112565844686684,
                    1.0500821290068711,
                    -0.4228848850327248,
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
                    list([0.05936656554842536, 34.95313492125299]),
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
                [10.0, 4.0, 10.0, nan, nan, nan],
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
                    0.1022309367943847,
                    list([0.3900965721838106, 6.344879386256914]),
                    list([0.16733909434684788, 10.7269854361358]),
                    nan,
                    nan,
                ],
                [
                    1.0658076533567236,
                    -0.4228848850327248,
                    list([0.5353314452187353, 16.347989651517267]),
                    nan,
                    nan,
                    nan,
                ],
                [-0.8343481518269869, 1.134845652371381, nan, nan, nan, nan],
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
                [4.0, 10.0, 15.0, nan, nan, nan],
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
                    1.149978331028409,
                    list([-0.6512734168403267, 5.953996539895858]),
                    list([0.2631200499275453, 15.01549411095887]),
                    1.04760234620062,
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
                [-2.547925820399213, 0.06548122629152132, nan, nan, nan, nan],
                [1.0160030224745213, nan, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan, nan],
            ],
            dtype=object,
        )
        self.C_C = np.array(
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, nan],
                [3.0, 15.0, 15.0, 4.0, nan, nan],
                [1.0, 15.0, 15.0, nan, nan, nan],
                [10.0, 1.0, nan, nan, nan, nan],
                [14.0, nan, nan, nan, nan, nan],
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
                    0.27275282,
                    0.46328193,
                    0.30775242,
                    0.14396727,
                    0.22998533,
                    0.22479665,
                ],
                [
                    0.87391783,
                    0.79113556,
                    0.76771778,
                    0.93706168,
                    0.95969633,
                    0.95339335,
                ],
                [
                    0.33985906,
                    0.17280276,
                    0.43307602,
                    0.35275917,
                    0.33811724,
                    0.29187607,
                ],
                [
                    0.55183273,
                    0.61436764,
                    0.57938101,
                    0.63053866,
                    0.64248016,
                    0.67413362,
                ],
                [
                    0.62093734,
                    0.60059869,
                    0.64921244,
                    0.61719035,
                    0.63688995,
                    0.60103895,
                ],
                [0.23588692, 0.28182591, 0.39815398, 0.20405075, 0.2237578, 0.30070006],
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
