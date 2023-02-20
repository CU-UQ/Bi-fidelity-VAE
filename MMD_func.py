"""
@author: NUOJIN
"""
import torch
import numpy as np

def norm_matrix(matrix_1, matrix_2):
    norm_square_1 = np.sum(np.square(matrix_1), axis = 1)
    norm_square_1 = np.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = np.sum(np.square(matrix_2), axis = 1)
    norm_square_2 = np.reshape(norm_square_2, (-1,1))
    
    inner_matrix = np.matmul(matrix_1, np.transpose(matrix_2))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + np.transpose(norm_square_2)
    
    return np.maximum(0, norm_diff)
    
def inner_matrix(matrix_1, matrix_2):
    return np.matmul(matrix_1, np.transpose(matrix_2))

def kernel_rational_quadratic(matrix_1, matrix_2, para):
    matrix = norm_matrix(matrix_1, matrix_2)
    return np.sum([(1 + matrix/(2.0 *sigma))**(-sigma) for sigma in para],axis=2)

def MMD_U(P,Q,para):
    P,Q = P.detach().numpy(),Q.detach().numpy()
    pm,n1 = P.shape
    qm,n2 = Q.shape
    
    KPP = kernel_rational_quadratic(P,P,para)
    KPQ = kernel_rational_quadratic(P,Q,para)
    KQQ = kernel_rational_quadratic(Q,Q,para)
    mmd = (KPP.sum()-np.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-np.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
    return mmd