import numpy as np

def average(arr, window):
    """Average an array arr over a certain window size
    """
    remainder = len(arr) % window
    avg = np.mean(arr[:-remainder].reshape(-1, window), axis=1)
    return avg
    
def pooling1d(arr, window, pad_size=0, stride=1, mode='min'):
    """Perform 1d pooling on an array arr
    """
    # Padding
    if mode == 'min':
        A = np.full(len(arr) + 2 * pad_size, np.inf)
    elif mode == 'max':
        A = np.full(len(arr) + 2 * pad_size, -np.inf)
    A[pad_size:len(arr) + pad_size] = arr
    
    # Window view of data
    from numpy.lib.stride_tricks import as_strided
    output_shape = ((len(A) - window)//stride + 1,)
    A_w = as_strided(A, shape = output_shape + (window,), 
                        strides = (stride*A.strides) + A.strides)
    
    if mode == 'max':
        return A_w.max(axis=1)
    elif mode == 'min':
        return A_w.min(axis=1)
    elif mode == 'avg' or mode == 'mean':
        return A_w.mean(axis=1)
    else:
        raise NotImplementedError("Mode {} is not implemented".format(mode))
        
def reg(x, y, n):
    """Regression, finding coefficients beta
    """
    
    xb = np.c_[np.ones((len(x),1))]
    for i in range(1,n+1):
        xb = np.c_[xb, x**i]

    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
    return beta.flatten()[::-1]
