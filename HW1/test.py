import hw1
import hw1_utils as utils

def test_1():
    X, Y = utils.load_reg_data
    w = hw1.linear_gd(X,Y)
    
