# Imports
from pathlib import Path
from vinecopula.bivariate import *
import matplotlib.pyplot as plt
import numpy as np

# Constants
TEST_DIR = Path(__file__).parent


# Tests
def test_random_cop_1():
    cop = 1
    par = 0.9
    n = 100
    np.random.seed(10)
    ur = random(cop, par, n)
    vr = np.loadtxt("tests/testdata/test_random_cop_1_data.csv", delimiter=",")
    assert np.array_equal(vr, ur)

    # Used for making this test (saving data and manual checking) # TO DO: remove
    # np.savetxt("testdata/test_random_cop_1_data.csv", X=ur, delimiter=",")
    # plt.scatter(ur[:, 0], ur[:, 1], alpha=0.5, label="Random") # check if results are reasonable
