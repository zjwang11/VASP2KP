# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:44:07 2023
Last modified on Wedn Dec 6 14:45:00 2023

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Calculate the numerical kp Hamiltonian as well as Zeeman's coupling
"""

import numpy as np
import sympy as sp
import sys

from ._transform_matrix import get_transform_matrix
from ._transform_matrix import CoRepresentationInequivalenceERROR

###############################################################################
# whether to reduce 'repr_matrix' to 'band_rep_matrix_list' then calculation
flag_reduce_Symmetry = True

# whether take the conjugate
Symmetry_conjugate_flag = False
###############################################################################

###############################################################################
# define ERROR class(type)
class SymmetryERROR(Exception):
    '''
    if the keys of the parameter Symmetry are not correct, raise this ERROR
    '''
    pass

class VASPMomentumMatrixERROR(Exception):
    '''
    if the generalized momentum matrices' names got by vasp2mat are not 'Pix', 'Piy' and 'Piz', raise this ERROR.
    '''
    pass

class VASPSpinMatrixERROR(Exception):
    '''
    if the spin matrices's names are not 'Pix', 'Piy' and 'Piz', raise this ERROR.
    '''
    pass
###############################################################################



###############################################################################
def direct_sum(A,B):
    '''
    Calculate the direct sum of two matrices

    Parameters
    ----------
    A : np.array/sp.Matrix
        input matrix
    B : np.array/sp.Matrix
        input matrix

    Returns
    -------
    direct_sum_result : sp.Matrix
        The direct sum of A and B

    '''
    # obtain dimension
    try:
        m, n = A.shape
        
    except:
        m = 1
        n = 1
        
        while True:
            try:
                A = A[0]
                
            except:
                break
            
        A = sp.Matrix([A])
        
    try:
        p, q = B.shape
        
    except:
        p = 1
        q = 1
        
        while True:
            try:
                B = B[0]
                
            except:
                break
            
        B = sp.Matrix([B])
    
    direct_sum_dim = (m + p, n + q)
    
    direct_sum_result = sp.zeros(*direct_sum_dim)
    
    direct_sum_result[:m, :n] = A
    direct_sum_result[m:, n:] = B
    
    return direct_sum_result
###############################################################################



###############################################################################
def bandreplist2rep(band_rep_list):
    '''
    Calculate the standard corepresentation based on 'band_repr_matrix_list'

    Parameters
    ----------
    band_rep_list : list
        The value of Symmetry[i]['band_repr_matrix_list'].

    Returns
    -------
    rep
        The standard corepresentation.

    '''
    band_rep_num = len(band_rep_list)
    
    if band_rep_num == 1:
        
        return band_rep_list[0]
    
    else:
        rep = direct_sum(band_rep_list[0],band_rep_list[1])

        for i in range(2,band_rep_num):
            rep = direct_sum(rep, band_rep_list[i])
        
        return rep
###############################################################################



###############################################################################
def get_std_Symmetry(Symmetry):
    '''
    Transform 'Symmetry' to the standard form of input of kdotp-generator

    Parameters
    ----------
    Symmetry : dictionary
        User's input.

    Returns
    -------
    Symmetry_copy : dictionary
        the standard form of input of kdotp-generator.

    '''
    Symmetry_copy = Symmetry.copy()
    
    for i in Symmetry_copy.keys():
        Symmetry_copy[i] = Symmetry[i].copy()
    
    for i in Symmetry_copy.keys():
        
        try:
            del Symmetry_copy[i]["repr_matrix"]
            
        except:
            pass
        
        Symmetry_copy[i]["repr_matrix"] = bandreplist2rep(Symmetry_copy[i]['band_repr_matrix_list'])
        
        del Symmetry_copy[i]['band_repr_matrix_list']
        
    return Symmetry_copy
###############################################################################



###############################################################################
def get_block_dim(A):
    '''
    Find the dimensions of each block of the block diagonal matrix

    Parameters
    ----------
    A : sympy.Matrix/numpy.array
        The matrix to solve.

    Returns
    -------
    diagonal_block_dimensions : list
        The list comprised of each dimension.

    '''

    rows, cols = A.shape
    
    diagonal_block_dimensions = []
    
    j=0
    j0 = 0
    dim = 0
    
    while True:
        if A[j:,j:].is_zero_matrix:
            
            for l in range(rows-j):
                diagonal_block_dimensions.append(1)
                
            break
        
        i = rows-1
        
        while i>=j:
            
            if A[j,i] != 0:
                break
            
            else:
                i-=1
        
        
        
        k = rows-1
        
        while k>=j:
            
            if A[k,j] != 0:
                break
            
            else:
                k-=1
        
        if i<j and k<j:
            
            try:
                dim += 1
                
            except:
                dim += 1
            
            diagonal_block_dimensions.append(1)
            j0 = dim
            j = dim
            
        else:
            dim = max(i,k)+1
        
            if dim>=rows:
                break
            
            tmp_flag = 0
            
            for l in range(dim):
                
                if not A[l,dim:].is_zero_matrix:
                    tmp_flag = 1
                    
                    break
                
                if not A[dim:,l].is_zero_matrix:
                    tmp_flag = 1
                    
                    break
            
            if tmp_flag == 0:
                diagonal_block_dimensions.append(dim-j0)
                j0 = dim
                
            j=dim
            
            
    diagonal_block_dimensions.append(rows-sum(diagonal_block_dimensions))
    
    # remove redundant elements
    if diagonal_block_dimensions[-1] == 0:
        del diagonal_block_dimensions[-1]
    
    #sys.exit()
    return diagonal_block_dimensions
###############################################################################



###############################################################################
def get_max_dim(dim_list):
    '''
    calculate the shared dimensions

    Parameters
    ----------
    dim_list : list
        The list comprised by the block dimensions of all matrices

    Returns
    -------
    max_dim_list : list
        the shared dimensions

    '''
    max_dim_list = dim_list[0]
    dim_list = dim_list[1:]
    
    if dim_list == []:
        return max_dim_list
    
    l_max = len(max_dim_list)
    
    for i in dim_list:
        len_i = len(i)
        
        if l_max<len_i:

            for j in range(len_i-l_max):
                max_dim_list.append(0)
                
            l_max = len_i

        l_dim_res = i[0]-max_dim_list[0]
        l_dim_record = i[0]
        l_dim_record_list = []
        
        j=1
        k=1
        
        
        while True:
            
            if l_dim_res == 0:
                l_dim_record_list.append(l_dim_record)
                
                try:
                    l_dim_res = i[j]-max_dim_list[k]
                    
                except:
                    break
                
                l_dim_record = i[j]
                j += 1
                k += 1
                
                continue
            
            if l_dim_res > 0:
                l_dim_res -= max_dim_list[k]
                k += 1
                
            else:
                l_dim_res += i[j]
                l_dim_record += i[j]
                j += 1
        
        max_dim_list = l_dim_record_list
        l_max = len(max_dim_list)
                
    return max_dim_list
###############################################################################



###############################################################################
def get_matrix_per_block(A,dim_list):
    '''
    Decompose matrix A into diagonal blocks

    Parameters
    ----------
    A : sympy.Matrix/numpy.array
        DESCRIPTION.
    dim_list : list
        the list of dimensions of each block.

    Returns
    -------
    part_matrix_list : list
        A list comprised by all the diagnal blocks.

    '''
    
    j=0
    part_matrix_list = []
    
    for i in dim_list:
        part_matrix = A[j:j+i,j:j+i]
        part_matrix_list.append(part_matrix)
        j += i
        
    return part_matrix_list
###############################################################################



###############################################################################
def get_corep_VASP_reduce(operator,dim_list):
    '''
    reduce the VASP corepresentation according to the dimension list

    Parameters
    ----------
    operator : dict
        the dictionary of all the corepresentation matrices obtained by VASP and vasp_kp.
    dim_list : list
        DESCRIPTION.

    Returns
    -------
    operator_ : dict
        the reduced dictionary of all the corepresentation matrices obtained by VASP and vasp_kp.
        (set elements outside the diagonal blocks to 0)

    '''
    operator_ = operator.copy()
    
    for i in operator.keys():
        operator_[i] = bandreplist2rep(get_matrix_per_block(operator[i], dim_list))
    
    return operator_
###############################################################################    


    
###############################################################################
def get_corep_dim_reduce_old(Symmetry):
    '''
    reduce 'repr_matrix' to 'band_rep_matrix_list'
    "old": this function are not used in the main code

    Parameters
    ----------
    Symmetry : dictionary
        user's input. O(3) matrices, corepresentation matrices and whether complex conjugate contains
        of each generators.

    Returns
    -------
    new_Symmetry : dictionary
        update 'Symmetry'
    max_dim_list : list
        the list of the shared dimensions of each blocks

    '''
    # deep copy
    new_Symmetry = Symmetry.copy()
    
    for i in new_Symmetry.keys():
        new_Symmetry[i] = Symmetry[i].copy()
    
    dim_each_list = []
    
    for i in new_Symmetry.keys():
        dim_list = get_block_dim(new_Symmetry[i]['repr_matrix'])
        dim_each_list.append(dim_list)
        
    max_dim_list = get_max_dim(dim_each_list)
    
    for i in new_Symmetry.keys():
        new_Symmetry[i]['band_repr_matrix_list'] = get_matrix_per_block(new_Symmetry[i]['repr_matrix'], max_dim_list)
        
        del new_Symmetry[i]['repr_matrix']
        
    return new_Symmetry, max_dim_list
###############################################################################



###############################################################################
def get_corep_dim_reduce(Symmetry,other_dim_list=[]):
    '''
    reduce 'repr_matrix' to 'band_rep_matrix_list'

    Parameters
    ----------
    Symmetry : dictionary
        user's input. O(3) matrices, corepresentation matrices and whether complex conjugate contains
        of each generators.

    Returns
    -------
    new_Symmetry : dictionary
        update 'Symmetry'
    max_dim_list : list
        the list of the shared dimensions of each blocks

    '''
    #deep copy
    new_Symmetry = Symmetry.copy()
    

    # reduce the calculation time
    sum_rep = sp.zeros(*(list(new_Symmetry.values())[0]['repr_matrix'].shape))
    
    for i in new_Symmetry.keys():
        new_Symmetry[i] = Symmetry[i].copy()
        sum_rep += abs(new_Symmetry[i]['repr_matrix'])

        
    dim_list = get_block_dim(sum_rep)
    
    if len(other_dim_list) != 0:
        max_dim_list_input = other_dim_list.copy()
        max_dim_list_input.append(dim_list)

        max_dim_list = get_max_dim(max_dim_list_input)
    else:
        max_dim_list = dim_list
    
    for i in new_Symmetry.keys():
        new_Symmetry[i]['band_repr_matrix_list'] = get_matrix_per_block(new_Symmetry[i]['repr_matrix'], max_dim_list)
        
        del new_Symmetry[i]['repr_matrix']
        
    return new_Symmetry, max_dim_list
###############################################################################



###############################################################################
def get_numeric_kp(pi_list,sigma_list,U,eigen_energy,band_interest_set,order,gfactor = 1,acc = 0):
    """
    Löwdin partitioning to one/two/three order

    Parameters
    ----------
    pi_list : list
        List of momentum matrix elements.
        pi_list[0]: pix; pi_list[1]: piy; pi_list[2]: piz
    sigma_list : list
        List of spin matrix elements.
        If gfactor=1, sigma_list[0]: sigx; sigma_list[1]: sigy; sigma_list[2]: sigz
    U : np.Matrix
        the similarity transformation matrix between VASP corepresenation and standard corepresentation
    eigen_energy : list
        list of eigen energy got by DFT Calculation
    band_interest_set : list
        id list of the bands of interest
    order : integer
        the order of kp Hamiltonian we intend to generate
    gfactor : integer, optional
        Whether to solve the Zeeman‘s coupling. The default is 1.
    acc : integer, optional(one of [0,1])
        user's input. The default is 0.
        If acc == 0, the code will not be accalerated.
        If acc == 1, the code will be accelerated by numba.njit.
    
    Returns
    -------
    dict
        the standard kp hamiltonian matrix of each order as well as the standard Zeeman's coupling if g_chooose is set to be 1

    """
    
    if acc == 1:
        
        try:
            import numba
            
        except:
            acc = 0
    
    #band_num =  band_end - band_start + 1 # total number of bands we are interested in
    band_num = len(band_interest_set)
    band_num_all = len(eigen_energy) # total number of bands got by DFT(VASP) Calculation
    
    # Unit conversion constant
    hbor=5.2917721067e-1
    const_kp_1=3.809982208629016*2/(hbor)
    const_kp_2=3.809982208629016
    const_kp_3=0.5*(3.809982208629016*2)**2/(hbor**2)
    
    # set the levi_civita
    if gfactor == 1:
        
        levi_civita = np.zeros((3, 3, 3), dtype=int)
        levi_civita[0, 1, 2] = 1
        levi_civita[2, 0, 1] = 1
        levi_civita[1, 2, 0] = 1
        levi_civita[1, 0, 2] = -1
        levi_civita[0, 2, 1] = -1
        levi_civita[2, 1, 0] = -1
    
    ###########################################################################
    # get the kp Hamiltonian of the specific order under the vasp basis
    # linear term
    #linear_V2=[const_kp_1*i[band_start-1:band_end,band_start-1:band_end] for i in pi_list]
    linear_V2=[const_kp_1*i[band_interest_set-1][:,band_interest_set-1] for i in pi_list]

    # Löwdin partitioning to two order 
    if order>=2 or gfactor==1:
        
        if acc == 0:
            
            quadratic_V2_part = np.zeros((band_num_all, band_num, band_num, 3, 3),dtype=complex)
            quadratic_V2_diag = np.zeros((band_num, band_num, 3, 3),dtype=complex)
            
            if gfactor == 1:
                G_part_V2 = np.zeros((band_num_all, band_num, band_num, 3),dtype=complex)
            
            for ii in range(3):
                
                for jj in range(3):
                    
                    for n_m1 in range(band_num):
                        #m1 = n_m1 + band_start - 1
                        m1 = band_interest_set[n_m1] - 1
                        
                        for n_m2 in range(band_num):
                            #m2 = n_m2 + band_start - 1
                            m2 = band_interest_set[n_m2] - 1
                            
                            for l in range(band_num_all):
                                
                                #if band_start-1 - 0.1 < l < band_end-1 + 0.1:
                                if l in (band_interest_set-1):
                                    
                                    continue
                                
                                else:
                                    s1 = (1 / (eigen_energy[m1] - eigen_energy[l]) + 1 / (eigen_energy[m2] - eigen_energy[l]))
                                    s2_2 = (pi_list[ii][m1, l] * pi_list[jj][l, m2])
                                    quadratic_V2_part[l, n_m1, n_m2, ii, jj] += s1 * s2_2 * const_kp_3
                                    
                                    if gfactor == 1:
                                        
                                        for kk in range(3):
                                            G_part_V2[l, n_m1, n_m2, kk] += s1 * s2_2 * levi_civita[ii, jj, kk] * const_kp_3 * (-1j) / (3.809982208629016 * 2)
            
                                if ii == jj and m1 == m2:
                                    quadratic_V2_diag[n_m1, n_m2, ii, jj] = const_kp_2
        
        elif acc == 1:
            
            # create decorator
            @numba.njit(fastmath = True)
            def acc_order2_kp_Zeeman(eigen_energy,band_interest_set,pi_list,gfactor):
                
                hbor=5.2917721067e-1
                const_kp_2=3.809982208629016
                const_kp_3=0.5*(3.809982208629016*2)**2/(hbor**2)
                #band_num =  band_end - band_start + 1
                band_num = len(band_interest_set)
                band_num_all = len(eigen_energy)
                quadratic_V2_part = np.zeros((band_num_all, band_num, band_num, 3, 3),dtype=np.complex128)
                quadratic_V2_diag = np.zeros((band_num, band_num, 3, 3),dtype=np.complex128)
                
                if gfactor == 1:
                    
                    levi_civita = np.zeros((3, 3, 3), dtype=np.int16)
                    levi_civita[0, 1, 2] = 1
                    levi_civita[2, 0, 1] = 1
                    levi_civita[1, 2, 0] = 1
                    levi_civita[1, 0, 2] = -1
                    levi_civita[0, 2, 1] = -1
                    levi_civita[2, 1, 0] = -1
                    
                if gfactor == 1:
                    G_part_V2 = np.zeros((band_num_all, band_num, band_num, 3),dtype=np.complex128)
                
                for ii in range(3):
                    
                    for jj in range(3):
                        
                        for n_m1 in range(band_num):
                            #m1 = n_m1 + band_start - 1
                            m1 = band_interest_set[n_m1] - 1
                            
                            for n_m2 in range(band_num):
                                #m2 = n_m2 + band_start - 1
                                m2 = band_interest_set[n_m2] - 1
                                
                                for l in range(band_num_all):
                                    
                                    if l in (band_interest_set-1):
                                        continue
                                    
                                    else:
                                        s1 = (1 / (eigen_energy[m1] - eigen_energy[l]) + 1 / (eigen_energy[m2] - eigen_energy[l]))
                                        s2_2 = (pi_list[ii][m1, l] * pi_list[jj][l, m2])
                                        quadratic_V2_part[l, n_m1, n_m2, ii, jj] += s1 * s2_2 * const_kp_3
                                        
                                        if gfactor == 1:
                                            
                                            for kk in range(3):
                                                G_part_V2[l, n_m1, n_m2, kk] += s1 * s2_2 * levi_civita[ii, jj, kk] * const_kp_3 * (-1j) / (3.809982208629016 * 2)
                
                                    if ii == jj and m1 == m2:
                                        quadratic_V2_diag[n_m1, n_m2, ii, jj] = const_kp_2
                                        
                return quadratic_V2_part,quadratic_V2_diag,G_part_V2
            
            quadratic_V2_part,quadratic_V2_diag,G_part_V2 = acc_order2_kp_Zeeman(np.array(eigen_energy,dtype = float),band_interest_set,np.array(pi_list,dtype = np.complex128),gfactor)
            
    # Löwdin partitioning to three order
    if order == 3:

        if acc == 0:
            
            cubic_lm_V2_part = np.zeros((band_num_all, band_num, band_num, band_num, 3, 3, 3),dtype=complex)
            cubic_ll_V2_part = np.zeros((band_num_all, band_num_all, band_num, band_num, 3, 3, 3),dtype=complex)
            
            for ii in range(3):
                
                for jj in range(3):
                    
                    for kk in range(3):
                        
                        for n_m0 in range(1, band_num + 1):
                            #m0 = n_m0 + band_start - 1
                            m0 = band_interest_set[n_m0-1]
                            
                            for n_m1 in range(1, band_num + 1):
                                #m1 = n_m1 + band_start - 1
                                m1 = band_interest_set[n_m1-1]
                                
                                for n_m2 in range(1, band_num + 1):
                                    #m2 = n_m2 + band_start - 1
                                    m2 = band_interest_set[n_m2-1]
                                    
                                    for l0 in range(band_num_all):
                                        
                                        if l0 in (band_interest_set-1):
                                            
                                            continue
                                        
                                        cubic_lm_V2_part[l0, n_m2 - 1, n_m0 - 1, n_m1 - 1, ii, jj, kk] = \
                                            -0.5 * (
                                                (pi_list[ii][m0 - 1, l0] * pi_list[jj][l0, m2 - 1] * pi_list[kk][m2 - 1, m1 - 1]) /
                                                ((eigen_energy[m1 - 1] - eigen_energy[l0]) * (eigen_energy[m2 - 1] - eigen_energy[l0])) +
                                                (pi_list[ii][m0 - 1, m2 - 1] * pi_list[jj][m2 - 1, l0] * pi_list[kk][l0, m1 - 1]) /
                                                ((eigen_energy[m0 - 1] - eigen_energy[l0]) * (eigen_energy[m2 - 1] - eigen_energy[l0]))
                                            )
            
                                    for l0 in range(band_num_all):
                                        
                                        #if band_start-1 - 0.1 < l0 < band_end-1 + 0.1:
                                        if l0 in (band_interest_set-1):
                                            continue
                                        
                                        for l1 in range(band_num_all):
                                            
                                            if l1 in (band_interest_set-1):
                                                continue
                                            
                                            cubic_ll_V2_part[l0, l1, n_m0 - 1, n_m1 - 1, ii, jj, kk] = \
                                                0.5 * (
                                                    pi_list[ii][m0 - 1, l0] * pi_list[jj][l0, l1] * pi_list[kk][l1, m1 - 1] *
                                                    (((eigen_energy[m0 - 1] - eigen_energy[l0]) * (eigen_energy[m0 - 1] - eigen_energy[l1])) ** -1 +
                                                    ((eigen_energy[m1 - 1] - eigen_energy[l0]) * (eigen_energy[m1 - 1] - eigen_energy[l1])) ** -1)
                                                )
            
            cubic_V2 = (np.sum(np.sum(cubic_lm_V2_part, axis=0), axis=0) +
                        np.sum(np.sum(cubic_ll_V2_part, axis=0), axis=0)) * const_kp_1**3
            
            # symmetrize
            
            cubic_V2_symm = np.zeros(cubic_V2.shape,dtype = complex)
            
            for ii in range(3):
                
                for jj in range(3):
                    
                    for kk in range(3):
                        
                        cubic_V2_symm[:, :, ii, jj, kk] = (
                            cubic_V2[:, :, ii, jj, kk] + cubic_V2[:, :, ii, kk, jj] + cubic_V2[:, :, jj, ii, kk] +
                            cubic_V2[:, :, kk, jj, ii] + cubic_V2[:, :, jj, kk, ii] + cubic_V2[:, :, kk, ii, jj]
                        ) / 6

        elif acc == 1:
            
            # create decorator
            @numba.njit(fastmath = True)
            def acc_order3_kp(eigen_energy,band_interest_set,pi_list):
                
                hbor=5.2917721067e-1
                const_kp_1=3.809982208629016*2/(hbor)
                #band_num =  band_end - band_start + 1
                band_num = len(band_interest_set)
                band_num_all = len(eigen_energy)
                cubic_lm_V2_part = np.zeros((band_num_all, band_num, band_num, band_num, 3, 3, 3),dtype=np.complex128)
                cubic_ll_V2_part = np.zeros((band_num_all, band_num_all, band_num, band_num, 3, 3, 3),dtype=np.complex128)
                
                for ii in range(3):
                    
                    for jj in range(3):
                        
                        for kk in range(3):
                            
                            for n_m0 in range(1, band_num + 1):
                                m0 = band_interest_set[n_m0-1]
                                
                                for n_m1 in range(1, band_num + 1):
                                    m1 = band_interest_set[n_m1-1]
                                    
                                    for n_m2 in range(1, band_num + 1):
                                        m2 = band_interest_set[n_m2-1]
                                        
                                        for l0 in range(band_num_all):
                                            
                                            if l0 in (band_interest_set-1):
                                                continue
                                            
                                            cubic_lm_V2_part[l0, n_m2 - 1, n_m0 - 1, n_m1 - 1, ii, jj, kk] = \
                                                -0.5 * (
                                                    (pi_list[ii][m0 - 1, l0] * pi_list[jj][l0, m2 - 1] * pi_list[kk][m2 - 1, m1 - 1]) /
                                                    ((eigen_energy[m1 - 1] - eigen_energy[l0]) * (eigen_energy[m2 - 1] - eigen_energy[l0])) +
                                                    (pi_list[ii][m0 - 1, m2 - 1] * pi_list[jj][m2 - 1, l0] * pi_list[kk][l0, m1 - 1]) /
                                                    ((eigen_energy[m0 - 1] - eigen_energy[l0]) * (eigen_energy[m2 - 1] - eigen_energy[l0]))
                                                )
                
                                        for l0 in range(band_num_all):
                                            
                                            if l0 in (band_interest_set-1):
                                                continue
                                            
                                            for l1 in range(band_num_all):
                                                
                                                if l1 in (band_interest_set-1):
                                                    continue
                                                
                                                cubic_ll_V2_part[l0, l1, n_m0 - 1, n_m1 - 1, ii, jj, kk] = \
                                                    0.5 * (
                                                        pi_list[ii][m0 - 1, l0] * pi_list[jj][l0, l1] * pi_list[kk][l1, m1 - 1] *
                                                        (((eigen_energy[m0 - 1] - eigen_energy[l0]) * (eigen_energy[m0 - 1] - eigen_energy[l1])) ** -1 +
                                                        ((eigen_energy[m1 - 1] - eigen_energy[l0]) * (eigen_energy[m1 - 1] - eigen_energy[l1])) ** -1)
                                                    )
                
                cubic_V2 = (np.sum(np.sum(cubic_lm_V2_part, axis=0), axis=0) +
                            np.sum(np.sum(cubic_ll_V2_part, axis=0), axis=0)) * const_kp_1**3
                
                # symmetrize
                
                cubic_V2_symm = np.zeros(cubic_V2.shape,dtype = np.complex128)
                
                for ii in range(3):
                    
                    for jj in range(3):
                        
                        for kk in range(3):
                            
                            cubic_V2_symm[:, :, ii, jj, kk] = (
                                cubic_V2[:, :, ii, jj, kk] + cubic_V2[:, :, ii, kk, jj] + cubic_V2[:, :, jj, ii, kk] +
                                cubic_V2[:, :, kk, jj, ii] + cubic_V2[:, :, jj, kk, ii] + cubic_V2[:, :, kk, ii, jj]
                            ) / 6
                            
                return cubic_V2_symm

            cubic_V2_symm = acc_order3_kp(np.array(eigen_energy,dtype = float),band_interest_set,np.array(pi_list,dtype = np.complex128))
                    
    if gfactor==1: 
        #calculate Zeeman's coupling
        
        G_orbit_V2 = np.sum(G_part_V2, axis=0)
        
        G_4c4 = np.zeros((band_num, band_num, 3), dtype=complex)
    
        for i in range(3):
            # add spin contribution
            G_4c4[:, :, i] = G_orbit_V2[:, :, i] + sigma_list[i]
            
        
    if order>=2:
        quadratic_V2_nosymm = np.sum(quadratic_V2_part, axis=0) + quadratic_V2_diag
        quadratic_V2_symm = np.zeros_like(quadratic_V2_nosymm)

        #symmertrize
        for ii in range(3):
            
            for jj in range(3):
                
                quadratic_V2_symm[:, :, ii, jj] = (quadratic_V2_nosymm[:, :, ii, jj] + quadratic_V2_nosymm[:, :, jj, ii]) / 2
    
    
    ###########################################################################
    #transform the kp Hamiltonian under the VASP basis to the standard form
    eigen_trans = U.conj().T@np.diag(np.array(eigen_energy)[band_interest_set-1])@U
    
    
    if order>=1 or gfactor == 1:
        linear_V2 = [U.conj().T@i@U for i in linear_V2]
    
    if order>=2:
        
        for i in range(3):
            
            for j in range(3):
                quadratic_V2_symm[:,:,i,j] = U.conj().T@quadratic_V2_symm[:,:,i,j] @U
    
    if order>=3:
        
        for i in range(3):
            
            for j in range(3):
                
                for k in range(3):
                    cubic_V2_symm[:,:,i,j,k] = U.conj().T@cubic_V2_symm[:,:,i,j,k]@U
    
    #transform the Zeeman's coupling under the VASP basis to the standard form
    if gfactor == 1:
        
        for i in range(3):
            G_4c4[:,:,i]=U.conj().T@G_4c4[:,:,i]@U
    
    ###########################################################################
    # return the value
    if order == 0:
        if gfactor==0:
            return {}
        
        else:
            return {"B":G_4c4}
        
    if order==1:
        
        if gfactor==0:
            return {"1":eigen_trans,"k":linear_V2}
        
        else:
            return {"1":eigen_trans,"k":linear_V2,"B":G_4c4}
        
    if order==2:
        
        if gfactor==0:
            return {"1":eigen_trans,"k":linear_V2,"k^2":quadratic_V2_symm}
        
        else:
            return {"1":eigen_trans,"k":linear_V2,"k^2":quadratic_V2_symm,"B":G_4c4}
        
    if order==3:
        
        if gfactor==0:
            return {"1":eigen_trans,"k":linear_V2,"k^2":quadratic_V2_symm,"k^3":cubic_V2_symm}
        
        else:
            return {"1":eigen_trans,"k":linear_V2,"k^2":quadratic_V2_symm,"k^3":cubic_V2_symm,"B":G_4c4}
###############################################################################        
        
        

###############################################################################        
def get_U_pi_sigma(data,operator,Symmetry,repr_split=True,dim_list = [],gfactor=1,log=0):
    '''
    find the similarity transformation matrix between VASP corep and standard corep, and transform the data to the list

    Parameters
    ----------
    data : dictionary
        the momentum matrices and the spin matrices under the VASP basis.
    operator : dictionary
        the corep matrices of generators under the VASP basis.
    Symmetry : dictionary
        the corep matrices and the rotation matrices under the standard basis.
    gfactor : int, optional
        Whether to solve the Zeeman's coupling. The default is 1.
    log : integer, optional
        Whether to input log file. The default is 0.

    Returns
    -------
    pi_list : list
        the list of generalized momentum matrices, [pix,piy,piz].
    sigma_list : list
        the list of spin matrices, [sigx,sigy,sigz].
    U : numpy.matrix
        the unitary similarity matrix between the corep under VASP basis and the standard corep.

    '''
    
    flag = True #if the information of unitaryizing process will be printed to console when log = 0
    
    band_num = list(operator.values())[0].shape[0]
    
    if dim_list == []:
        dim_list = list(operator.values())[0].shape[0]
        
    save_stdout = sys.stdout
    
    if log != 0:
        logging = open("find_similarity_transformation_matrix.log","w")
        #logging.write('ERRORs in the optimization process (unitaryizing the similarity transformation matrix)\n')
        
    else:
        if flag:
            # set the output stream to empty
            
            # judge what the system is
            import platform
            system = platform.system()
            
            # define the empty output stream
            if system == "Windows":
                logging = open('nul', 'w')
                
            elif system == "Linux":
                logging = open('/dev/null', 'w')
                
            elif system == "Darwin": # mac
                logging = open('/dev/null', 'w')
                
            else:
                logging = open('/dev/null', 'w')
                
        else:
            logging = save_stdout
            
    sys.stdout = logging
    
    print("==================================================")
    print("Begin finding similarity transformation")
    
    global flag_reduce_Symmetry
    
    if flag_reduce_Symmetry:
        
        if 'repr_matrix' not in Symmetry[list(Symmetry.keys())[0]].keys():
            flag_reduce_Symmetry = False
            
        else:
            
            Symmetry, dim_list = get_corep_dim_reduce(Symmetry,[dim_list])
            operator = get_corep_VASP_reduce(operator,dim_list)
        
            print("Dimension composition:",dim_list)

    if 'repr_matrix' in Symmetry[list(Symmetry.keys())[0]].keys():
        # try to find the similarity transformation matrix between VASP corep and standard corep
        trace_ERROR = False
        try:
            
            # reduce the tolerence gradually
            try:
                U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-4)
                
            except (CoRepresentationInequivalenceERROR):
                # trace not equal
                raise CoRepresentationInequivalenceERROR("The standard corepresentation is not equivalent to VASP corepresentation!(traces ERROR)")
                
                if not trace_ERROR:
                    trace_ERROR = True
                    
            except:
                
                try:
                    U = get_transform_matrix(operator,Symmetry,band_num,tol = 5e-4)
                
                except:
                    
                    try:
                        U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-3)
                    
                    except:
                        
                        try:
                            U = get_transform_matrix(operator,Symmetry,band_num,tol = 5e-3)
                            
                        except:
                            U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-2)
            
            
        # take the conjugation
        except:
            
            if Symmetry_conjugate_flag:
                
                raise ValueError('Fail to find unitary U!')
                
                sys.exit()
                
            print("\n\n Begin conjudge")
            
            for i in Symmetry.keys():
                Symmetry[i]['repr_matrix']=sp.conjugate(Symmetry[i]['repr_matrix'])
            
            # reduce the tolerence gradually
            try:
                U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-4)
                
            except (CoRepresentationInequivalenceERROR):
                # trace not equal
                if trace_ERROR:
                    raise CoRepresentationInequivalenceERROR("The standard corepresentation is not equivalent to VASP corepresentation!(traces ERROR)")
                    sys.exit()
                    
                else:
                    raise ValueError('Fail to find unitary U!')
                
            except:
                
                try:
                    U = get_transform_matrix(operator,Symmetry,band_num,tol = 5e-4)
                
                except:
                    
                    try:
                        U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-3)
                    
                    except:
                        
                        try:
                            U = get_transform_matrix(operator,Symmetry,band_num,tol = 5e-3)
                            
                        except:
                            try:
                                U = get_transform_matrix(operator,Symmetry,band_num,tol = 1e-2)
                            except:
                                print("ERROR: Fail to find unitary U!")
                                sys.stdout = save_stdout
                                print("ERROR: Fail to find the unitary transformation U!")
                                print('''Please check: 1. Whether the standard representation matrices in "mat2kp.in" are correct.''')
                                print('''              2. Whether WAVECAR is not empty and LWAVE=.TRUE. is set in INCAR when running vasp2mat.''')
                                print('''              3. Whether rot_n and rot_tau in "INCAR.mat" in INCAR.mat are correct when running vasp2mat.''')
                                if repr_split:
                                    print('''              4. If the above points are all checked but mat2kp still fails, or you do not use representation matrices from BCS Server, you can set "repr_split = False" in "mat2kp.in" and run mat2kp again.''')
                                sys.exit()
                    
                    
    elif 'band_repr_matrix_list' in Symmetry[list(Symmetry.keys())[0]].keys():
        
        corepr_num = len(Symmetry[list(Symmetry.keys())[0]]['band_repr_matrix_list'])
        key_list = list(Symmetry.keys())
        dimension_tot = 0
        operator_ = dict()
        
        for i in range(corepr_num):
            operator_.clear()
            
            if i != 0:
                print('\n\n\n')
                
            print("------------------------------------------------")
            print("Begin finding the similarity transformation matrix of the part "+str(i+1))
            
            for id_key in range(len(key_list)):
                key = key_list[id_key]
                
                try:
                    del Symmetry[key]['repr_matrix']
                    
                except:
                    pass
                
                else:
                    pass
                
                finally:
                    try:
                        ele = Symmetry[key]['band_repr_matrix_list'][i]
                        
                    except:
                        ele = Symmetry[key]['band_repr_matrix_list']
                    
                    if type(ele) != sp.matrices.dense.MutableDenseMatrix:
                        
                        try:
                            ele[0]
                            
                        except:
                            ele = [ele]
                            
                        else:
                            pass
                        
                        finally:
                            ele = sp.Matrix(ele)
                            
                    Symmetry[key]['repr_matrix'] = ele
                    dimension = Symmetry[key]['repr_matrix'].shape[0]
                
                dimension_tot_ = dimension_tot + dimension
                operator_i = operator[key][dimension_tot:dimension_tot_,dimension_tot:dimension_tot_]
                operator_[key] = operator_i
                
            try:
                
                # reduce the tolerence gradually
                try:
                    #print('flag1')
                    U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 1e-4)
                    #print('flag2')
                    
                except (CoRepresentationInequivalenceERROR):
                    # trace not equal
                    raise CoRepresentationInequivalenceERROR("The standard corepresentation is not equivalent to VASP corepresentation!(traces ERROR)")
                
                except:
                    
                    try:
                        U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 5e-4)
                    
                    except:
                        
                        try:
                            U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 1e-3)
                        
                        except:
                            
                            try:
                                U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 5e-3)
                                
                            except:
                                U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 1e-2)
                
                
            # take the conjugation
            except:
                
                #sys.exit()
                print("\n\n Begin conjudge")
                
                for k in Symmetry.keys():
                    Symmetry[k]['repr_matrix']=sp.conjugate(Symmetry[k]['repr_matrix'])
                    Symmetry[k]['band_repr_matrix_list'][i]=sp.conjugate(Symmetry[k]['band_repr_matrix_list'][i])
                
                # reduce the tolerence gradually
                try:
                    U_ = get_transform_matrix(operator_,Symmetry,band_num,tol = 1e-4)
                    
                except (CoRepresentationInequivalenceERROR):
                    # trace not equal
                    raise CoRepresentationInequivalenceERROR("The standard corepresentation is not equivalent to VASP corepresentation!(traces ERROR)")
                
                except:
                    
                    try:
                        U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 5e-4)
                    
                    except:
                        
                        try:
                            U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 1e-3)
                        
                        except:
                            
                            try:
                                U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 5e-3)
                                
                            except:
                                try:
                                    U_ = get_transform_matrix(operator_,Symmetry,dimension,tol = 1e-2)
                                except:
                                    print("ERROR: Fail to find unitary U!")
                                    sys.stdout = save_stdout
                                    print("ERROR: Fail to find the unitary transformation U!")
                                    print('''Please check: 1. Whether the standard representation matrices in "mat2kp.in" are correct.''')
                                    print('''              2. Whether WAVECAR is not empty and LWAVE=.TRUE. is set in INCAR when running vasp2mat.''')
                                    print('''              3. Whether rot_n and rot_tau in "INCAR.mat" in INCAR.mat are correct when running vasp2mat.''')
                                    sys.exit()
                                    
            print("End finding the similarity transformation matrix of the part "+str(i+1))
            print("------------------------------------------------")
            print('\n-----------  Result of Unitary similarity transformation matrix  ----------')
            print("Unitary similarity transformation matrix for part "+str(i+1)+":")
            
            print(sp.Matrix(U_))
            
            print("\n")
            print("ERROR:")
            print("ERROR for unitaryizing:", end=' ')
            #print(abs(U.conj().T@U-np.eye(band_num)).max())
            U__ = np.array(U_,dtype = np.complex128)
            
            ERROR = sum(abs(U__.conj().T@U__-np.eye(dimension))).sum()
            print(ERROR)
            
            #print(operator_.keys())
            for j in Symmetry.keys():
                
                print("ERROR for "+j+":", end=' ')
                
                if Symmetry[j]['repr_has_cc']:    
                    #print(abs(U.conj().T@operator[i]@U.conj()-np.array(Symmetry[i]['repr_matrix'].evalf())).max())
                    try:
                        ERROR = sum(abs(U__.conj().T@operator_[j]@U__.conj()-np.array(Symmetry[j]['band_repr_matrix_list'][i].evalf()))).sum()
                        
                    except:
                        
                        try:
                            ERROR = sum(abs(U__.conj().T@operator_[j]@U__.conj()-np.array(Symmetry[j]['band_repr_matrix_list'][i]))).sum()
                        
                        except:
                            ERROR = sum(abs(U__.conj().T@operator_[j]@U__.conj()-np.array(Symmetry[j]['band_repr_matrix_list'][i]))).evalf()
                            ERROR = abs(ERROR)
                            
                    print(ERROR)
                    
                else:
                    #print(abs(U.conj().T@operator[i]@U-np.array(Symmetry[i]['repr_matrix'].evalf())).max())
                    try:
                        ERROR = sum(abs(U__.conj().T@operator_[j]@U__-np.array(Symmetry[j]['band_repr_matrix_list'][i].evalf()))).sum()
                        
                    except:
                        
                        try:
                            ERROR = sum(abs(U__.conj().T@operator_[j]@U__-np.array(Symmetry[j]['band_repr_matrix_list'][i]))).sum()
                        
                        except:
                            ERROR = sum(abs(U__.conj().T@operator_[j]@U__-np.array(Symmetry[j]['band_repr_matrix_list'][i]))).evalf()
                            ERROR = abs(ERROR)
                            
                    print(ERROR)
                
                
                
            if i==0: 
                U = U_
                
            else:
                U = direct_sum(U, U_)
                
            dimension_tot = dimension_tot_
            
        Symmetry = get_std_Symmetry(Symmetry)
        
    
    else:
        sys.stdout = save_stdout
        print("ERROR: There must be 'band_repr_matrix_list' or 'repr_matrix' keys!")
        #raise SymmetryERROR("There must be 'band_repr_matrix_list' or 'repr_matrix' keys!")
        sys.exit()
        
        
                
    print("End finding similarity transformation")
    print("==================================================")
    
    # calculate the ERROR
    print('\n\n\n')
    print("==================================================")
    print('\n==========  Result of Unitary similarity transformation matrix  ==========')
    print("Unitary similarity transformation matrix:")
    print(sp.Matrix(U))
    print("\n")
    print("ERROR:")
    print("ERROR for unitaryizing:", end=' ')
    #print(abs(U.conj().T@U-np.eye(band_num)).max())
    U = np.array(U,dtype = np.complex128)
    
    ERROR = sum(abs(U.conj().T@U-np.eye(band_num))).sum()
    print(ERROR)
    
    for i in Symmetry.keys():
        
        print("ERROR for "+i+":", end=' ')
        
        if Symmetry[i]['repr_has_cc']:    
            #print(abs(U.conj().T@operator[i]@U.conj()-np.array(Symmetry[i]['repr_matrix'].evalf())).max())
            try:
                ERROR = sum(abs(U.conj().T@operator[i]@U.conj()-np.array(Symmetry[i]['repr_matrix'].evalf()))).sum()
                
            except:
                
                try:
                    ERROR = sum(abs(U.conj().T@operator[i]@U.conj()-np.array(Symmetry[i]['repr_matrix']))).sum()
                
                except:
                    ERROR = sum(abs(U.conj().T@operator[i]@U.conj()-np.array(Symmetry[i]['repr_matrix']))).evalf()
                    ERROR = abs(ERROR)
                    
            print(ERROR)
            
        else:
            #print(abs(U.conj().T@operator[i]@U-np.array(Symmetry[i]['repr_matrix'].evalf())).max())
            try:
                ERROR = sum(abs(U.conj().T@operator[i]@U-np.array(Symmetry[i]['repr_matrix'].evalf()))).sum()
                
            except:
                
                try:
                    ERROR = sum(abs(U.conj().T@operator[i]@U-np.array(Symmetry[i]['repr_matrix']))).sum()
                
                except:
                    ERROR = sum(abs(U.conj().T@operator[i]@U-np.array(Symmetry[i]['repr_matrix']))).evalf()
                    ERROR = abs(ERROR)
                    
            print(ERROR)
    
    print('\n\n\n')
    
    
    if log != 0:
        
        if flag:
            logging.close()
            
        else:
            pass
        
    sys.stdout = save_stdout
    
    
    
    # put the input generalized momentum matrices into a list
    if 'Piz' in data.keys():
        pi_list=[data['Pix'],data['Piy'],data['Piz']]
        
    else:
        
        try: 
            pi_list=[data['pix'],data['piy'],data['piz']]
            
        except:
            #raise VASPMomentumMatrixERROR("The generalized momentum matrices' names got by vasp2mat should be 'Pix', 'Piy', 'Piz'! The vat_name in the input file INCAR.mat of vasp2mat should be 'Pi'!")
            print("ERROR: The generalized momentum matrices' names got by vasp2mat should be 'Pix', 'Piy', 'Piz'! The vat_name in the input file INCAR.mat of vasp2mat should be 'Pi'!")
            sys.exit()

    if gfactor == 1:
        # put the input spin matrices into a list
        if 'sigz' in data.keys():
            sigma_list=[data['sigx'],data['sigy'],data['sigz']]
            
        else:
            
            try:
                sigma_list=[data['Sigx'],data['Sigy'],data['Sigz']]
                
            except:
                print("ERROR: The spin matrices' names got by vasp2mat should be 'sigx', 'sigy', 'sigz'! The 'vat_name' in the input file INCAR.mat of vasp2mat should be 'sig'!")
                #raise VASPSpinMatrixERROR("The spin matrices' names got by vasp2mat should be 'sigx', 'sigy', 'sigz'! The 'vat_name' in the input file INCAR.mat of vasp2mat should be 'sig'!")
                
                sys.exit()
        
    else:    
        sigma_list=[]
    
    return pi_list,sigma_list,U,Symmetry
###############################################################################



###############################################################################
if __name__ == "__main__":
    A = sp.Matrix([[2, 0, 0, 0, 2, 0, 0, 0], [0, 2, 0, 0, 0, 2, 0, 0], [0, 0, 2, 0, 0, 0, 2, 0], [0, 0, 0, 2, 0, 0, 0, 2], [2, 0, 0, 0, 2, 0, 0, 0], [0, 2, 0, 0, 0, 2, 0, 0], [0, 0, 2, 0, 0, 0, 2, 0], [0, 0, 0, 2, 0, 0, 0, 2]])
    get_block_dim(A)
if False:
    from sympy import I
    Symmetry = {

    'C3z' : {
        'rotation_matrix': sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2,0],
                                      [sp.sqrt(3)/2, -sp.Rational(1,2), 0],
                                      [0, 0, 1]]),
        'repr_matrix': sp.Matrix([[sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0,0,0,0],
                                  [0,sp.Rational(1,2)+I*sp.sqrt(3)/2,0,0,0,0],
                                  [0,0,sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0,0],
                                  [0,0,0,sp.Rational(1,2)+I*sp.sqrt(3)/2,0,0],
                                  [0,0,0,0,-1,0],
                                  [0,0,0,0,0,-1]]),              #GM8 8 7
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[sp.Rational(1,2)-I*sp.sqrt(3)/2,0],[0,sp.Rational(1,2)+I*sp.sqrt(3)/2]]),
                            sp.Matrix([[sp.Rational(1,2)-I*sp.sqrt(3)/2,0],[0,sp.Rational(1,2)+I*sp.sqrt(3)/2]]),
                            -sp.eye(2)]
    },



    'C2z' : {
        'rotation_matrix': sp.Matrix([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]]),
        'repr_matrix': sp.Matrix([[-I,0, 0,0,0,0],
                                  [0,I,0,0,0,0],
                                  [0,0,-I,0, 0,0],
                                  [0,0,0,I,0,0],
                                  [0,0,0,0,-I,0],
                                  [0,0,0,0,0,I]]),    #GM8 8 7
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[-I,0],[0,I]]),
                            sp.Matrix([[-I,0],[0,I]]),
                            sp.Matrix([[-I,0],[0,I]])]
    },


    'Mx' : {
        'rotation_matrix': sp.Matrix([[-1,0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]),
        'repr_matrix': sp.Matrix([[0,-sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0,0,0],
                                  [sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0,0,0,0],
                                  [0,0,0,-sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0],
                                  [0,0,sp.Rational(1,2)-I*sp.sqrt(3)/2,0,0,0],
                                  [0,0,0,0,0,1],
                                  [0,0,0,0,-1,0]]),
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[0,-sp.Rational(1,2)-I*sp.sqrt(3)/2],[sp.Rational(1,2)-I*sp.sqrt(3)/2,0]]),
                            sp.Matrix([[0,-sp.Rational(1,2)-I*sp.sqrt(3)/2],[sp.Rational(1,2)-I*sp.sqrt(3)/2,0]]),
                            sp.Matrix([[0,1],[-1,0]])]
    },




    'T' : {
        'rotation_matrix': sp.eye(3),
        'repr_matrix': sp.Matrix([[0,1, 0,0,0,0],
                                  [-1,0,0,0,0,0],
                                  [0,0,0,1, 0,0],
                                  [0,0,-1,0,0,0],
                                  [0,0,0,0,0,1],
                                  [0,0,0,0,-1,0]]),
        'repr_has_cc': True,
        'band_repr_matrix_list':[sp.Matrix([[0,1],[-1,0]]),
                            sp.Matrix([[0,1],[-1,0]]),
                            sp.Matrix([[0,1],[-1,0]])]
    }
    }


    for i in Symmetry.keys():
        print(bandreplist2rep(Symmetry[i]['band_repr_matrix_list'])-Symmetry[i]['repr_matrix'])
        
        
    Symmetry = {

    'C3z' : {
        'rotation_matrix': sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2,0],
                                      [sp.sqrt(3)/2, -sp.Rational(1,2), 0],
                                      [0, 0, 1]]),
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[sp.Rational(1,2)-I*sp.sqrt(3)/2,0],[0,sp.Rational(1,2)+I*sp.sqrt(3)/2]]),
                            sp.Matrix([[sp.Rational(1,2)-I*sp.sqrt(3)/2,0],[0,sp.Rational(1,2)+I*sp.sqrt(3)/2]]),
                            -sp.eye(2)]
    },



    'C2z' : {
        'rotation_matrix': sp.Matrix([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]]),
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[-I,0],[0,I]]),
                            sp.Matrix([[-I,0],[0,I]]),
                            sp.Matrix([[-I,0],[0,I]])]
    },


    'Mx' : {
        'rotation_matrix': sp.Matrix([[-1,0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]),
        'repr_has_cc': False,
        'band_repr_matrix_list':[sp.Matrix([[0,-sp.Rational(1,2)-I*sp.sqrt(3)/2],[sp.Rational(1,2)-I*sp.sqrt(3)/2,0]]),
                            sp.Matrix([[0,-sp.Rational(1,2)-I*sp.sqrt(3)/2],[sp.Rational(1,2)-I*sp.sqrt(3)/2,0]]),
                            sp.Matrix([[0,1],[-1,0]])]
    },




    'T' : {
        'rotation_matrix': sp.eye(3),
        'repr_has_cc': True,
        'band_repr_matrix_list':[sp.Matrix([[0,1],[-1,0]]),
                            sp.Matrix([[0,1],[-1,0]]),
                            sp.Matrix([[0,1],[-1,0]])]
    }
    }


    
    def deepcopy_Symmetry(Symmetry):
        Symmetry_copy = Symmetry.copy()
        for i in Symmetry_copy.keys():
            Symmetry_copy[i] = Symmetry[i].copy()
        return Symmetry_copy
        

    for i in Symmetry.keys():
        if 'repr_matrix' not in Symmetry[i].keys():
            Symmetry[i]['repr_matrix'] = bandreplist2rep(Symmetry[i]['band_repr_matrix_list'])
            #del Symmetry[i]['band_repr_matrix_list']
            


    Symmetry_2 = Symmetry.copy()
    Symmetry_2_ = deepcopy_Symmetry(Symmetry)
    for i in Symmetry.keys():

        del Symmetry[i]['band_repr_matrix_list']
    
    from VASP2KP import get_O3_matrix
        
    '''
    A = Matrix([
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 3, 8]
    ])
    
    B = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 8]
    ])
    
    C = Matrix([
        [1, 3, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 8]
    ])
    
    print(get_block_dim(A))
    
    print(get_block_dim(B))
    
    print(get_max_dim([[1,2,3,4,5,1],[3,3,4,1,4,1],[1,2,3,4,4,1,1],[3,3,4,1,3,1,1]]))
    
    
    get_corep_dim_reduce([A,B,C])
    '''
    
    
    Symmetry = {
    
    'C2z' : {
        'rotation_matrix': get_O3_matrix(180,[0,0,1],1),
        'repr_matrix': sp.Matrix([[-I,0,0,0,0,0,0,0],
                                  [0,I,0,0,0,0,0,0],
                                  [0,0,-I,0,0,0,0,0],
                                  [0,0,0,I,0,0,0,0],
                                  [0,0,0,0,-I,0,0,0],
                                  [0,0,0,0,0,I,0,0],
                                  [0,0,0,0,0,0,-I,0],
                                  [0,0,0,0,0,0,0,I]]),
        'repr_has_cc': False  
    },
    
    
    
    'C2y' : {
        'rotation_matrix': get_O3_matrix(180,[0,1,0],1),
        'repr_matrix': sp.Matrix([[0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0,0,0],
                                  [sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0,0,0,0],
                                  [0,0,0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0],
                                  [0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0,0],
                                  [0,0,0,0,0,sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0],
                                  [0,0,0,0,-sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0,0],
                                  [0,0,0,0,0,0,0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2],
                                  [0,0,0,0,0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0]]),
        'repr_has_cc': False  
    },
    
    
    'C4z' : {
        'rotation_matrix': get_O3_matrix(90,[0,0,1],1),
        'repr_matrix': sp.Matrix([[sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0,0,0,0,0,0],
                                  [0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0,0,0],
                                  [0,0,sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0,0,0,0],
                                  [0,0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0],                              
                                  [0,0,0,0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0],
                                  [0,0,0,0,0,-sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0],                              
                                  [0,0,0,0,0,0,sp.sqrt(2)/2-I*sp.sqrt(2)/2,0],
                                  [0,0,0,0,0,0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2]]),
        'repr_has_cc': False  
    },
    
    
    
    
    'P' : {
        'rotation_matrix': get_O3_matrix(0,[0,0,1],-1),
        'repr_matrix': sp.Matrix([[-1,0, 0,0,0,0,0,0],
                                  [0,-1,0,0,0,0,0,0],
                                  [0,0,-1,0,0,0,0,0],
                                  [0,0,0,-1,0,0,0,0],
                                  [0,0,0,0,-1,0,0,0],
                                  [0,0,0,0,0,-1,0,0],
                                  [0,0,0,0,0,0,-1,0],
                                  [0,0,0,0,0,0,0,-1]]),
        'repr_has_cc': False  
    },
    
    'T' : {
        'rotation_matrix': sp.eye(3),
        'repr_matrix': sp.Matrix([[0,-1,0,0,0,0,0,0],
                                  [1,0,0,0,0,0,0,0],
                                  [0,0,0,-1,0,0,0,0],
                                  [0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,-1,0,0],
                                  [0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,-1],
                                  [0,0,0,0,0,0,1,0]]),
        'repr_has_cc': True  
    }
    }
    
    Symmetry2 = {
    
    'C2z' : {
        'rotation_matrix': get_O3_matrix(180,[0,0,1],1),
        'repr_has_cc': False  ,
        'band_repr_matrix_list': [sp.Matrix([[-I,0],[0,I]]),sp.Matrix([[-I,0],[0,I]]),sp.Matrix([[-I,0],[0,I]]),sp.Matrix([[-I,0],[0,I]])]
    },
    
    
    
    'C2y' : {
        'rotation_matrix': get_O3_matrix(180,[0,1,0],1),
        'repr_has_cc': False  ,
        'band_repr_matrix_list': [sp.Matrix([[0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2],[sp.sqrt(2)/2+I*sp.sqrt(2)/2,0]]),
                                  sp.Matrix([[0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2],[sp.sqrt(2)/2+I*sp.sqrt(2)/2,0]]),
                                  sp.Matrix([[0,sp.sqrt(2)/2-I*sp.sqrt(2)/2],[-sp.sqrt(2)/2-I*sp.sqrt(2)/2,0]]),
                                  sp.Matrix([[0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2],[sp.sqrt(2)/2+I*sp.sqrt(2)/2,0]])]
    },
    
    
    'C4z' : {
        'rotation_matrix': get_O3_matrix(90,[0,0,1],1),
        'repr_has_cc': False  ,
        'band_repr_matrix_list': [sp.Matrix([[sp.sqrt(2)/2-I*sp.sqrt(2)/2,0],[0,sp.sqrt(2)/2+I*sp.sqrt(2)/2]]),
                                  sp.Matrix([[sp.sqrt(2)/2-I*sp.sqrt(2)/2,0],[0,sp.sqrt(2)/2+I*sp.sqrt(2)/2]]),
                                  sp.Matrix([[-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0],[0,-sp.sqrt(2)/2-I*sp.sqrt(2)/2]]),
                                  sp.Matrix([[sp.sqrt(2)/2-I*sp.sqrt(2)/2,0],[0,sp.sqrt(2)/2+I*sp.sqrt(2)/2]])]
    },
    
    
    
    
    'P' : {
        'rotation_matrix': get_O3_matrix(0,[0,0,1],-1),
        'repr_has_cc': False  ,
        'band_repr_matrix_list': [-sp.eye(2),-sp.eye(2),-sp.eye(2),-sp.eye(2)]
    },
    
    'T' : {
        'rotation_matrix': sp.eye(3),
        'repr_has_cc': True  ,
        'band_repr_matrix_list': [sp.Matrix([[0,-1],[1,0]]),sp.Matrix([[0,-1],[1,0]]),sp.Matrix([[0,-1],[1,0]]),sp.Matrix([[0,-1],[1,0]])]
    }
    }
    
    print(get_corep_dim_reduce(Symmetry))
    
    print(get_corep_dim_reduce(Symmetry)==Symmetry2)
    
    
    
    
    
    
    
