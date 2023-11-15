from calculate_key_quantities import key_quantities
import scipy.linalg
from model import load_model_params
import numpy as np
from scipy.stats.stats import pearsonr


def linear_fit(X, y):
    A = np.c_[X,  np.ones(X.shape[0])]
    C,residual,_,_ = scipy.linalg.lstsq(A, y)
    return C, residual

def generalization_gap_estimation(unit_sparse_rate_average):

    C, residual = linear_fit(unit_sparse_rate_average[:, 0:2], unit_sparse_rate_average[:, 2])
    print("The fitting SSR is %.4f"%(residual))
    a, b, z0 = C
    print("Fitting function is %.3f*x + %.3f*y + %.3f"%(a, b, z0))

    g = a * unit_sparse_rate_average[:, 0] + b * unit_sparse_rate_average[:, 1] + z0

    print("The true generalization gap is: ", unit_sparse_rate_average[:, 2])
    print("The fitted generalization gap is: ", g)
    r_pearson, p_pearson = pearsonr(g, unit_sparse_rate_average[:, 2])
    print("Pearson coefficient is %.4f"%r_pearson)
