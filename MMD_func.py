#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:48:46 2022

@author: nokicheng
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

def kernel_RBF(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    K =  np.exp(-matrix/ (sigma**2))
    return K

def kernel_rational_quadratic(matrix_1, matrix_2, para):
    matrix = norm_matrix(matrix_1, matrix_2)
    return np.sum([(1 + matrix/(2.0 *sigma))**(-sigma) for sigma in para],axis=2)

def kernel_poly(matrix_1, matrix_2, para):
    a = 1/para[0]
    b = 1
    d = 3
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (a * matrix + b) ** d
    return K 

def MMD(P,Q,para):
    pm,n1 = P.shape
    qm,n2 = Q.shape
    if n1 != n2:
        print('Data Dimension not match')
        return
    mmd = pm**(-2)*kernel_RBF(P,P,para).sum()-2/(pm*qm)*kernel_RBF(P,Q,para).sum()+qm**(-2)*kernel_RBF(Q,Q,para).sum()
    return mmd

def MMD_U1(P,Q,para):
    pm,n1 = P.shape
    qm,n2 = Q.shape
    if n1 != n2:
        print('Data Dimension not match')
        return
    KPP = kernel_RBF(P,P,para)
    KPQ = kernel_RBF(P,Q,para)
    KQQ = kernel_RBF(Q,Q,para)
    mmd = (KPP.sum()-np.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-np.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
    return mmd

def MMD_U(P,Q,para):
    P,Q = P.detach().numpy(),Q.detach().numpy()
    pm,n1 = P.shape
    qm,n2 = Q.shape
    
    KPP = kernel_rational_quadratic(P,P,para)
    KPQ = kernel_rational_quadratic(P,Q,para)
    KQQ = kernel_rational_quadratic(Q,Q,para)
    mmd = (KPP.sum()-np.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-np.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
    return mmd

# =============================================================================
### Followings are pyTorch version
# def norm_matrix(matrix_1, matrix_2):
#     norm_square_1 = torch.sum(torch.square(matrix_1), axis = 1)
#     norm_square_1 = torch.reshape(norm_square_1, (-1,1))
#     
#     norm_square_2 = torch.sum(torch.square(matrix_2), axis = 1)
#     norm_square_2 = torch.reshape(norm_square_2, (-1,1))
#     
#     inner_matrix = torch.matmul(matrix_1, torch.transpose(matrix_2,0,1))
#     
#     norm_diff = -2 * inner_matrix + norm_square_1 + torch.transpose(norm_square_2,0,1)
#     
#     return torch.maximum(torch.FloatTensor([0]), norm_diff)
#
# def kernel_rational_quadratic(matrix_1, matrix_2, para):
#     matrix = norm_matrix(matrix_1, matrix_2)
#     res = torch.zeros(size=matrix.shape)
#     for sigma in para:
#         res += (1 + matrix/(2.0 *sigma))**(-sigma)
#     return res
# 
# def kernel_poly(matrix_1, matrix_2, para):
#     a = 1/para[0]
#     b = 1
#     d = 3
#     matrix = inner_matrix(matrix_1, matrix_2)
#     K = (a * matrix + b) ** d
#     return K 
# 
# def MMD(P,Q,para):
#     pm,n1 = P.shape
#     qm,n2 = Q.shape
#     if n1 != n2:
#         print('Data Dimension not match')
#         return
#     mmd = pm**(-2)*kernel_RBF(P,P,para).sum()-2/(pm*qm)*kernel_RBF(P,Q,para).sum()+qm**(-2)*kernel_RBF(Q,Q,para).sum()
#     return mmd
# 
# def MMD_U1(P,Q,para):
#     pm,n1 = P.shape
#     qm,n2 = Q.shape
#     if n1 != n2:
#         print('Data Dimension not match')
#         return
#     KPP = kernel_RBF(P,P,para)
#     KPQ = kernel_RBF(P,Q,para)
#     KQQ = kernel_RBF(Q,Q,para)
#     mmd = (KPP.sum()-np.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-np.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
#     return mmd
# 
# def MMD_U(P,Q,para):
#     pm,n1 = P.shape
#     qm,n2 = Q.shape
#     if n1 != n2:
#         print('Data Dimension not match')
#         return
#     KPP = kernel_rational_quadratic(P,P,para)
#     KPQ = kernel_rational_quadratic(P,Q,para)
#     KQQ = kernel_rational_quadratic(Q,Q,para)
#     mmd = (KPP.sum()-torch.diag(KPP).sum())/(pm*pm-pm)+(KQQ.sum()-torch.diag(KQQ).sum())/(qm*qm-qm)-2*KPQ.sum()/(pm*qm)
#     return mmd.item()
# =============================================================================
