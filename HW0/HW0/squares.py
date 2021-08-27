import numpy
import torch

## you only need to complete the two functions.
## do not print anything when submit your code to the Gradescope, or it may fail the test.
## do not include your code for Question 3 in your submission
def numpy_squares(k):
    """return (1, 4, 9, ..., k^2) as a numpy array"""
    if k < 0: 
    	return []
    return numpy.array([i**2 for i in range(1,k+1)])
    # your code here
    pass

def torch_squares(k):
    """return (1, 4, 9, ..., k^2) as a torch array"""
    # your code here
    if k < 0: 
    	return []
    return torch.Tensor([i*i for i in range(1,k+1) ])
    pass

