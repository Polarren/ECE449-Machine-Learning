import hw1
import hw1_utils as utils
import matplotlib.pyplot as plt

def test_1():
    X, Y = utils.load_reg_data()
    w = hw1.logistic(X,Y)
    return w

# test_1()
plot = hw1.logistic_vs_ols()

