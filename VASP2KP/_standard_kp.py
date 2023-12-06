# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:44:07 2023
Last modified on Wedn Dec 6 14:45:00 2023 

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Construct the standard kp Hamiltonian as well as calculate the parameters
"""

import numpy as np
import kdotp_generator as kp
import sympy as sp
import os
import sys
import subprocess
import warnings

#warnings.filterwarnings("ignore", category=sp.SymPyDeprecationWarning)

###############################################################################
# Hidden parameter region
error_flag = 0 # 0,1 or 2

tol = {0:1e-4,  1:1e-3,  2:1e-2,  3:1e-1,  "Zeeman":1e-3}
#tol = {0:0,  1:0,  2:0,  3:0,  "Zeeman":0}

print_each_order_flag = False

###############################################################################



###############################################################################
# define Error class(type)
class NumericalStandardKpMatrixError(Exception):
    '''
    if the absolute value of the element of the numerical standard Hamiltonian whose value should be zero is larger than tol, 
        raise this error
    '''
    pass

class Error_FlagChooseError(Exception):
    '''
    if the parameter error_flag is not in [0,1,2]: raise this error
    '''
    pass


class suppress_stdout_stderr(object):
    '''A context manager for doing a "deep suppression" of stdout and stderr in Python, i.e. will suppress all print, even if the print originates in acompiled C/Fortran sub-function.This will not suppress raised exceptions, since exceptions are printedto stderr just before a script exits, and after the context manager hasexited (at least, I think that is why it lets exceptions through).'''
    def __init__(self):
        super(suppress_stdout_stderr,self).__init__()
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))
        
    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        
    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
###############################################################################



###############################################################################
def get_coeff(eql,coe_sym):
    '''
    solve the equations set to get the undetermined parameters of kp Hamiltonian or Zeeman's coupling

    Parameters
    ----------
    eql : list
        the equation list, the elements' type is sympy.logic.boolalg.BooleanFalse.
    coe_sym : tuple
        the tuple of the object.

    Returns
    -------
    coe : list
        the value of the undetermined parameters.
    residuals : float
        the error of the linear least squard method.

    '''
    
    # transform the equation set to the matrix
    A, B = sp.linear_eq_to_matrix(eql, coe_sym)
    #coe, residuals, rank, singular_values = np.linalg.lstsq(np.array(A,dtype=complex), np.array(B,dtype=complex), rcond=None)
    
    A_array=np.array(A,dtype=np.complex128)
    B_array=np.array(B,dtype=np.complex128)
    
    # seperate the real parts and the imaginary parts
    # real part
    A_r=A_array.real
    B_r=B_array.real
    
    # imaginary part
    A_i=A_array.imag
    B_i=B_array.imag
    
    # construct the new parameters matrix and value vector
    A_matrix=np.vstack([A_r,A_i])
    B_matrix=np.vstack([B_r,B_i])
    #print(A_matrix)
    #print(B_matrix)
    
    # linear least squares method to get the parameters as well as error
    coe, residuals, rank, singular_values = np.linalg.lstsq(A_matrix, B_matrix, rcond=None)
    
    return coe, residuals
###############################################################################



###############################################################################
def get_std_kp_0_order(kpmodel,eigen_trans,print_flag=2,log=0):
    '''
    get the part of standard kp hamiltonian without any the wave vector k.

    Parameters
    ----------
    kpmodel : list
        The list of all symmetry allowed kp Hamiltonian of the zeroth order, constructed by invariant theory.
    eigen_trans : numpy.array
        The standard zeroth order kp Hamiltonian matrix contructed by VASP and vasp_song.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    coe_0, list
        the list of the values of the undetermined parameters.
    coe_0_sym, tuple
        the tuple of all the symbol variables of the undetermined parameters. The type of the element is 'sympy.core.symbol.Symbol'.
    sum_result_kpmodel_0, sympy.matrices.dense.MutableDenseMatrix
        the part of the analytical standard kp hamiltonian: terms without any the wave vector k.
    error, float
        the error of the linear least squares method.
    all_dif, float
        the sum of the absolute values of the numerical zero matrix elements.

    '''
    
    
    # define symbol variables of the undetermined paramters
    coe_0_num = len(kpmodel)
    coe_0_sym = sp.symbols('a1:{}'.format(coe_0_num+1))
    
    #dimension = kpmodel[0].shape[0]
    dimension = eigen_trans.shape[0]

    
    sum_kpmodel_0=sp.zeros(dimension)
    wrong_flag = 0
    
    for i in range(len(kpmodel)):
        sum_kpmodel_0 = sum_kpmodel_0 + coe_0_sym[i]*kpmodel[i]
        
    zeros_k = eigen_trans
        
    ###########################################################################
    # construct the linear equation set
    eql=[]
    
    all_dif = 0
    
    for i in range(dimension):
        
        for j in range(dimension):
            
            ele=sp.Eq(sum_kpmodel_0[i,j]-zeros_k[i,j],0)
            
            ###############################################################
            # calculate the errors of the matrix elements whose theoretical value is zero
            if (ele is sp.false) or (ele is sp.true):
                
                # calculate error
                dif = abs(zeros_k[i,j])
                all_dif += dif
                
                if dif > tol[0]:
                    
                    if error_flag == 0:
                        pass
                    
                    elif error_flag == 1:
                        # raise an error
                        raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                        
                        return 
                    
                    elif error_flag == 2:
                        
                        if log == 1:
                            
                            if wrong_flag == 0:
                                error_file = open("Matrix_error.log","a")
                                error_file.write('\n\n==========  Error of 0-order kp model  ==========\n')
                                error_file.write("VASP_element \t\t Difference\n")
                                error_file.write(str(zeros_k[i,j])+'\t'+str(dif)+'\n')
                                wrong_flag = 1
                                
                            else:
                                error_file.write(str(zeros_k[i,j])+'\t'+str(dif)+'\n')
                                
                        else:
                            sys.stdout.flush()
                            tmp_stdout = sys.stdout
                            sys.stdout = sys.__stdout__
                            
                            if wrong_flag == 0:
                                print('\n\n==========  Error of 0-order kp model  ==========')
                                print("VASP_element \t\t Difference")
                                print(str(zeros_k[i,j])+'\t'+str(dif))
                                wrong_flag = 1
                            
                            else:
                                print(str(zeros_k[i,j])+'\t'+str(dif))
                            
                            sys.stdout.flush()
                            sys.stdout = tmp_stdout
                continue
            
            eql.append(ele)
            
    if wrong_flag == 1:
        
        if log == 1:
            
            error_file.write('---------- VASP matrix ----------\n')
            error_file.write(str(sp.Matrix(zeros_k))+'\n')
            error_file.write('---------- kp matrix ----------\n')
            error_file.write(str(sum_kpmodel_0)+'\n')
        
        else:
            sys.stdout.flush()
            tmp_stdout = sys.stdout
            sys.stdout = sys.__stdout__
            
            print('---------- VASP matrix ----------')
            print(sp.Matrix(zeros_k))
            print('---------- kp matrix ----------')
            print(sum_kpmodel_0)
            
            sys.stdout.flush()
            sys.stdout = tmp_stdout
    
    if log == 1 and wrong_flag == 1:
        error_file.close()
        
    if len(eql) == 0:
        return [],[],0,0,all_dif
    
    coe_0, residuals=get_coeff(eql,coe_0_sym)
    error = residuals[0]
    
    ###########################################################################
    # output the result
    if print_each_order_flag and print_flag != 0:
        print("0-order kp Hamiltonian:")
        print(sum_kpmodel_0)
        print("Parameters:")
        
        for i in range(coe_0_num):
            coe_0[i][0] = round(coe_0[i][0].real,4)
            print(coe_0_sym[i],"=",coe_0[i][0].real,";")
            
        error_print = "{:.2e}".format(error)
        all_dif_print = "{:.2e}".format(all_dif)
        
        print("Error of the linear least square method:",error_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
        
    sum_result_kpmodel_0=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_result_kpmodel_0 = sum_result_kpmodel_0 + coe_0[i,0]*kpmodel[i]

    
    return coe_0,coe_0_sym,sum_result_kpmodel_0,error,all_dif
###############################################################################



###############################################################################
def get_std_kp_1_order(kpmodel,linear_V2,print_flag=2,log=0):
    '''
    get the part of the analytical standard kp hamiltonian: terms whose vector k are of 1 order

    Parameters
    ----------
    kpmodel : list
        The list of all symmetry allowed kp Hamiltonian of the zeroth order, constructed by invariant theory.
    linear_V2 : numpy.array
        The standard first order kp Hamiltonian matrix contructed by VASP and vasp_song.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    coe_1, list
        the list of the values of the undetermined parameters.
    coe_1_sym, tuple
        the tuple of all the symbol variables of the undetermined parameters. The type of the element is 'sympy.core.symbol.Symbol'.
    sum_result_kpmodel_1, sympy.matrices.dense.MutableDenseMatrix
        the part of the analytical standard kp hamiltonian: terms whose vector k are of 1 order.
    error, float
        the error of the linear least squares method.
    all_dif, float
        the sum of the absolute values of the numerical zero matrix elements.

    '''
    
    # define the symbol variables(constant)
    kx,ky,kz = sp.symbols("kx ky kz")
    id_k=[kx,ky,kz]
    char_k = "kx ky kz".split(' ')
    
    # define symbol variables of the undetermined paramters
    coe_1_num = len(kpmodel)
    coe_1_sym = sp.symbols('b1:{}'.format(coe_1_num+1))
    
    #dimension = kpmodel[0].shape[0]
    dimension = linear_V2[0].shape[0]
    
    sum_kpmodel_1=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_kpmodel_1 = sum_kpmodel_1 + coe_1_sym[i]*kpmodel[i]
        
    wrong_flag = 0
    
    ###########################################################################
    # construct the linear equation set
    eql = []
    all_dif = 0
    
    for ii in range(3):
        
        tmp_sum_kpmodel_1 = sum_kpmodel_1.copy()
        tmp_sum_kpmodel_1 = tmp_sum_kpmodel_1.subs(id_k[ii], 1)
        wrong_flag_per = 0
        
        for l in range(3):
            tmp_sum_kpmodel_1 = tmp_sum_kpmodel_1.subs(id_k[l], 0)
            
        linear_V2_part = linear_V2[ii][:,:]
        
        for i in range(dimension):
            
            for j in range(dimension):
                
                ele=sp.Eq(tmp_sum_kpmodel_1[i,j]-linear_V2_part[i,j],0)
                
                ###############################################################
                # calculate the errors of the matrix elements whose theoretical value is zero
                if (ele is sp.false) or (ele is sp.true):
                    
                    dif = abs(linear_V2_part[i,j])
                    all_dif += dif
                    
                    if dif > tol[1]:
                        
                        if error_flag == 0:
                            pass
                        
                        elif error_flag == 1:
                            raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                            
                            return 
                        
                        elif error_flag == 2:
                            
                            if log == 1:
                                
                                if wrong_flag == 0:
                                    error_file = open("Matrix_error.log","a")
                                    error_file.write('\n\n==========  Error of 1-order kp model  ==========\n')
                                    error_file.write('========== {} ==========\n'.format(char_k[ii]))
                                    error_file.write("VASP_element \t\t Difference\n")
                                    error_file.write(str(linear_V2_part)+'\t'+str(dif)+'\n')
                                    wrong_flag = 1
                                    wrong_flag_per = 1
                                    
                                elif wrong_flag_per == 0:
                                    error_file.write('\n========== {} ==========\n'.format(char_k[ii]))
                                    error_file.write("VASP_element \t\t Difference\n")
                                    error_file.write(str(linear_V2_part[i,j])+'\t'+str(dif)+'\n')
                                    wrong_flag_per = 1
                                    
                                else:
                                    error_file.write(str(linear_V2_part[i,j])+'\t'+str(dif)+'\n')
                                    
                            else:
                                sys.stdout.flush()
                                tmp_stdout = sys.stdout
                                sys.stdout = sys.__stdout__
                                
                                if wrong_flag == 0:
                                    print('\n\n==========  Error of 1-order kp model  ==========')
                                    print('========== {} =========='.format(char_k[ii]))
                                    print("VASP_element \t\t Difference")
                                    print(str(linear_V2_part[i,j])+'\t'+str(dif))
                                    wrong_flag = 1
                                    wrong_flag_per = 1
                                
                                elif wrong_flag_per == 0:
                                    print('\n========== {} =========='.format(char_k[ii]))
                                    print("VASP_element \t\t Difference")
                                    print(str(linear_V2_part[i,j])+'\t'+str(dif))
                                    wrong_flag_per = 1
                                
                                else:
                                    print(str(linear_V2_part[i,j])+'\t'+str(dif))
                                
                                sys.stdout.flush()
                                sys.stdout = tmp_stdout
                                
                    continue
                
                eql.append(ele)
                
        if wrong_flag_per == 1:
            
            if log == 1:
                error_file.write('---------- VASP matrix for {} ----------\n'.format(char_k[ii]))
                error_file.write(str(sp.Matrix(linear_V2_part))+'\n')
                error_file.write('---------- kp matrix for {} ----------\n'.format(char_k[ii]))
                error_file.write(str(tmp_sum_kpmodel_1))
                
            else:
                sys.stdout.flush()
                tmp_stdout = sys.stdout
                sys.stdout = sys.__stdout__
                
                print('---------- VASP matrix for {} ----------'.format(char_k[ii]))
                print(sp.Matrix(linear_V2_part))
                print('---------- kp matrix for {} ----------'.format(char_k[ii]))
                print(tmp_sum_kpmodel_1)
                
                sys.stdout.flush()
                sys.stdout = tmp_stdout
    
    if log == 1 and wrong_flag == 1:
        error_file.close()
        
    if len(eql) == 0:
        return [],[],0,0,all_dif
    
    coe_1, residuals=get_coeff(eql,coe_1_sym)
    error = residuals[0]
    
    ###########################################################################
    # output the result
    if print_each_order_flag and print_flag != 0:
        print("1-order kp Hamiltonian:")
        print(sum_kpmodel_1)
        print("Parameters:")
        
        for i in range(coe_1_num):
            coe_1[i][0] = round(coe_1[i][0].real,4)
            print(coe_1_sym[i],"=",coe_1[i][0].real,";")
            
        error_print = "{:.2e}".format(error)
        all_dif_print = "{:.2e}".format(all_dif)
            
        print("Error of the linear least square method:",error_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
        
    sum_result_kpmodel_1=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_result_kpmodel_1 = sum_result_kpmodel_1 + coe_1[i,0]*kpmodel[i]
        
        
    return coe_1,coe_1_sym,sum_result_kpmodel_1,error,all_dif
###############################################################################



###############################################################################
def get_std_kp_2_order(kpmodel,quadratic_V2_symm,print_flag=2,log=0):
    '''
    get the part of the analytical standard kp hamiltonian: terms whose vector k are of 2 order

    Parameters
    ----------
    kpmodel : list
        The list of all symmetry allowed kp Hamiltonian of the zeroth order, constructed by invariant theory.
    quadratic_V2_symm : numpy.array
        The standard second order kp Hamiltonian matrix contructed by VASP and vasp_song.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    coe_2, list
        the list of the values of the undetermined parameters.
    coe_2_sym, tuple
        the tuple of all the symbol variables of the undetermined parameters. The type of the element is 'sympy.core.symbol.Symbol'.
    sum_result_kpmodel_2, sympy.matrices.dense.MutableDenseMatrix
        the part of the analytical standard kp hamiltonian: terms whose vector k are of 2 order.
    error, float
        the error of the linear least squares method.
    all_dif, float
        the sum of the absolute values of the numerical zero matrix elements.

    '''
    
    
    # define the symbol variables(constant)
    kx,ky,kz = sp.symbols("kx ky kz")
    id_k=[kx,ky,kz]
    char_k = "kx ky kz".split(' ')
    
    # define symbol variables of the undetermined paramters
    coe_2_num = len(kpmodel)
    coe_2_sym = sp.symbols('c1:{}'.format(coe_2_num+1))
    
    #dimension = kpmodel[0].shape[0]
    dimension = quadratic_V2_symm.shape[0]
    
    
    sum_kpmodel_2=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_kpmodel_2 = sum_kpmodel_2 + coe_2_sym[i]*kpmodel[i]

    wrong_flag = 0
    
    ###########################################################################
    # construct the linear equation set
    eql=[]
    all_dif = 0
    
    for ii in range(3):
        
        for jj in range(ii,3):
            
            tmp_sum_kpmodel_2 = sum_kpmodel_2.copy()
            tmp_sum_kpmodel_2 = tmp_sum_kpmodel_2.subs(id_k[ii]*id_k[jj], 1)
            wrong_flag_per = 0
            
            for l in range(3):
                tmp_sum_kpmodel_2 = tmp_sum_kpmodel_2.subs(id_k[l], 0)
                
            quadratic_V2_part=quadratic_V2_symm[:,:,ii,jj]
            
            if ii!=jj:
                quadratic_V2_part=quadratic_V2_part*2
            
            for i in range(dimension):
                
                for j in range(dimension):
                    
                    ele=sp.Eq(tmp_sum_kpmodel_2[i,j]-quadratic_V2_part[i,j],0)
                    
                    ###############################################################
                    # calculate the errors of the matrix elements whose theoretical value is zero
                    if (ele is sp.false) or (ele is sp.true):
                        #print(abs(tmp_sum_kpmodel_2[i,j]-quadratic_V2_part[i,j]).evalf())
                        
                        dif = abs(quadratic_V2_part[i,j])
                        all_dif += dif
                        
                        if dif > tol[2]:
                            
                            if error_flag == 0:
                                pass
                            
                            elif error_flag == 1:
                                raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                                
                                return 
                            
                            elif error_flag == 2:
                                
                                if log == 1:
                                    
                                    if wrong_flag == 0:
                                        error_file = open("Matrix_error.log","a")
                                        error_file.write('\n\n==========  Error of 2-order kp model  ==========\n')
                                        error_file.write('========== {}{} ==========\n'.format(char_k[ii],char_k[jj]))
                                        error_file.write("VASP_element \t\t Difference\n")
                                        error_file.write(str(quadratic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                        wrong_flag = 1
                                        wrong_flag_per = 1
                                        
                                    elif wrong_flag_per == 0:
                                        error_file.write('\n========== {}{} ==========\n'.format(char_k[ii],char_k[jj]))
                                        error_file.write("VASP_element \t\t Difference\n")
                                        error_file.write(str(quadratic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                        wrong_flag_per = 1
                                        
                                    else:
                                        error_file.write(str(quadratic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                        
                                else:
                                    tmp_stdout = sys.stdout
                                    sys.stdout.flush()
                                    sys.stdout = sys.__stdout__
                                    
                                    if wrong_flag == 0:
                                        print('\n\n==========  Error of 2-order kp model  ==========')
                                        print('========== {}{} =========='.format(char_k[ii],char_k[jj]))
                                        print("VASP_element \t\t Difference")
                                        print(str(quadratic_V2_part[i,j])+'\t'+str(dif))
                                        wrong_flag = 1
                                        wrong_flag_per = 1
                                    
                                    elif wrong_flag_per == 0:
                                        print('\n========== {}{} =========='.format(char_k[ii],char_k[jj]))
                                        print("VASP_element \t\t Difference")
                                        print(str(quadratic_V2_part[i,j])+'\t'+str(dif))
                                        wrong_flag_per = 1
                                    
                                    else:
                                        print(str(quadratic_V2_part[i,j])+'\t'+str(dif))
                                    
                                    sys.stdout.flush()
                                    sys.stdout = tmp_stdout
                        continue
                    
                    eql.append(ele)
                    
            if wrong_flag_per == 1:
                
                if log == 1:
                    error_file.write('---------- VASP matrix for {}{} ----------\n'.format(char_k[ii],char_k[jj]))
                    error_file.write(str(sp.Matrix(quadratic_V2_part))+'\n')
                    error_file.write('---------- kp matrix for {}{} ----------\n'.format(char_k[ii],char_k[jj]))
                    error_file.write(str(tmp_sum_kpmodel_2)+'\n')
                
                else:
                    sys.stdout.flush()
                    tmp_stdout = sys.stdout
                    sys.stdout = sys.__stdout__
                    
                    print('---------- VASP matrix for {}{} ----------'.format(char_k[ii],char_k[jj]))
                    print(sp.Matrix(quadratic_V2_part))
                    print('---------- kp matrix for {}{} ----------'.format(char_k[ii],char_k[jj]))
                    print(tmp_sum_kpmodel_2)
                    
                    sys.stdout.flush()
                    sys.stdout = tmp_stdout
                    
    if log == 1 and wrong_flag == 1:
        error_file.close()
        
    if len(eql) == 0:
        return [],[],0,0,all_dif
    
    coe_2, residuals=get_coeff(eql,coe_2_sym)
    #print(coe_2)
    #print(residuals)
    error = residuals[0]
    
    ###########################################################################
    # output the result
    if print_each_order_flag and print_flag != 0:
        print("2-order kp Hamiltonian:")
        print(sum_kpmodel_2)
        print("Parameters:")
        
        for i in range(coe_2_num):
            coe_2[i][0] = round(coe_2[i][0].real,4)
            print(coe_2_sym[i],"=",coe_2[i][0].real,";")
            
        error_print = "{:.2e}".format(error)
        all_dif_print = "{:.2e}".format(all_dif)
        
        print("Error of the linear least square method:",error_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
    
    sum_result_kpmodel_2=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_result_kpmodel_2 = sum_result_kpmodel_2 + coe_2[i,0]*kpmodel[i]
    
    return coe_2,coe_2_sym,sum_result_kpmodel_2,error,all_dif
###############################################################################



###############################################################################
def get_std_kp_3_order(kpmodel,cubic_V2_symm,print_flag=2,log=0):
    '''
    get the part of the analytical standard kp hamiltonian: terms whose vector k are of 3 order

    Parameters
    ----------
    kpmodel : list
        The list of all symmetry allowed kp Hamiltonian of the zeroth order, constructed by invariant theory.
    cubic_V2_symm : numpy.array
        The standard third order kp Hamiltonian matrix contructed by VASP and vasp_song.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    coe_2, list
        the list of the values of the undetermined parameters.
    coe_2_sym, tuple
        the tuple of all the symbol variables of the undetermined parameters. The type of the element is 'sympy.core.symbol.Symbol'.
    sum_result_kpmodel_2, sympy.matrices.dense.MutableDenseMatrix
        the part of the analytical standard kp hamiltonian: terms whose vector k are of 2 order.
    error, float
        the error of the linear least squares method.
    all_dif, float
        the sum of the absolute values of the numerical zero matrix elements.

    '''

    
    # define the symbol variables(constant)
    kx,ky,kz = sp.symbols("kx ky kz")
    id_k=[kx,ky,kz]
    char_k = "kx ky kz".split(' ')
    
    #dimension = kpmodel[0].shape[0]
    dimension = cubic_V2_symm.shape[0]
    
    # define symbol variables of the undetermined paramters
    coe_3_num = len(kpmodel)
    coe_3_sym = sp.symbols('d1:{}'.format(coe_3_num+1))
    
    sum_kpmodel_3=sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_kpmodel_3 = sum_kpmodel_3 + coe_3_sym[i]*kpmodel[i]

    wrong_flag = 0
    
    ###########################################################################
    # construct the linear equation set
    eql = []
    all_dif = 0
    
    for ii in range(3):
        
        for jj in range(ii,3):
            
            for kk in range(jj,3):
                
                tmp_sum_kpmodel_3 = sum_kpmodel_3.copy()
                tmp_sum_kpmodel_3 = tmp_sum_kpmodel_3.subs(id_k[ii]*id_k[jj]*id_k[kk], 1)
                wrong_flag_per = 0
                
                for l in range(3):
                    tmp_sum_kpmodel_3 = tmp_sum_kpmodel_3.subs(id_k[l], 0)
                
                cubic_V2_part=cubic_V2_symm[:,:,ii,jj,kk]
                
                if ii != jj:
                    
                    if ii != kk:
                        
                        if jj != kk:
                            
                            cubic_V2_part = cubic_V2_part*6
                            
                        else:
                            cubic_V2_part = cubic_V2_part*3
                            
                    else:
                        cubic_V2_part = cubic_V2_part*3
                        
                else:
                    if ii != kk:
                        cubic_V2_part = cubic_V2_part*3
                        
                for i in range(dimension):
                    
                    for j in range(dimension):
                        
                        ele=sp.Eq(tmp_sum_kpmodel_3[i,j]-cubic_V2_part[i,j],0)
                        
                        ###############################################################
                        # calculate the errors of the matrix elements whose theoretical value is zero
                        if (ele is sp.false) or (ele is sp.true):
                            #print(abs(tmp_sum_kpmodel_3[i,j]-cubic_V2_part[i,j]).evalf())
                            dif = abs(cubic_V2_part[i,j])
                            all_dif += dif
                            
                            if dif > tol[3]:
                                
                                if error_flag == 0:
                                    pass
                                
                                elif error_flag == 1:
                                    raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                                    
                                    return 
                                
                                elif error_flag == 2:
                                    
                                    if log == 1:
                                        
                                        if wrong_flag == 0:
                                            error_file = open("Matrix_error.log","a")
                                            error_file.write('\n\n==========  Error of 3-order kp model  ==========\n')
                                            error_file.write('========== {}{}{} ==========\n'.format(char_k[ii],char_k[jj],char_k[kk]))
                                            error_file.write("VASP_element \t\t Difference\n")
                                            error_file.write(str(cubic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                            wrong_flag = 1
                                            wrong_flag_per = 1
                                            
                                        elif wrong_flag_per == 0:
                                            error_file.write('\n========== {}{}{} ==========\n'.format(char_k[ii],char_k[jj],char_k[kk]))
                                            error_file.write("VASP_element \t\t Difference\n")
                                            error_file.write(str(cubic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                            wrong_flag_per = 1
                                            
                                        else:
                                            error_file.write(str(cubic_V2_part[i,j])+'\t'+str(dif)+'\n')
                                            
                                    else:
                                        tmp_stdout = sys.stdout
                                        sys.stdout.flush()
                                        sys.stdout = sys.__stdout__
                                        
                                        if wrong_flag == 0:
                                            print('\n\n==========  Error of 3-order kp model  ==========')
                                            print('========== {}{}{} =========='.format(char_k[ii],char_k[jj],char_k[kk]))
                                            print("VASP_element \t\t Difference")
                                            print(str(cubic_V2_part[i,j])+'\t'+str(dif))
                                            wrong_flag = 1
                                            wrong_flag_per = 1
                                        
                                        elif wrong_flag_per == 0:
                                            print('\n========== {}{}{} =========='.format(char_k[ii],char_k[jj],char_k[kk]))
                                            print("VASP_element \t\t Difference")
                                            print(str(cubic_V2_part[i,j])+'\t'+str(dif))
                                            wrong_flag_per = 1
                                        
                                        else:
                                            print(str(cubic_V2_part[i,j])+'\t'+str(dif))
                                        
                                        sys.stdout.flush()
                                        sys.stdout = tmp_stdout
                                        
                            continue
                        
                        eql.append(ele)
                        
                if wrong_flag_per == 1:
                    
                    if log == 1:
                        error_file.write('---------- VASP matrix for {}{}{} ----------\n'.format(char_k[ii],char_k[jj],char_k[kk]))
                        error_file.write(str(sp.Matrix(cubic_V2_part))+'\n')
                        error_file.write('---------- kp matrix for {}{}{} ----------\n'.format(char_k[ii],char_k[jj],char_k[kk]))
                        error_file.write(str(tmp_sum_kpmodel_3)+'\n')
                    
                    else:
                        sys.stdout.flush()
                        tmp_stdout = sys.stdout
                        sys.stdout = sys.__stdout__
                        
                        print('---------- VASP matrix for {}{}{} ----------'.format(char_k[ii],char_k[jj],char_k[kk]))
                        print(sp.Matrix(cubic_V2_part))
                        print('---------- kp matrix for {}{}{} ----------'.format(char_k[ii],char_k[jj],char_k[kk]))
                        print(tmp_sum_kpmodel_3)
                        
                        sys.stdout.flush()
                        sys.stdout = tmp_stdout
    
    if log == 1 and wrong_flag == 1:
        error_file.close()
        
    if len(eql) == 0:
        return [],[],0,0,all_dif
    
    coe_3, residuals=get_coeff(eql,coe_3_sym)
    error = residuals[0]
    
    ###########################################################################
    # output the result
    if print_each_order_flag and print_flag != 0:
        print("3-order kp Hamiltonian:")
        print(sum_kpmodel_3)
        print("Parameters:")
        
        for i in range(coe_3_num):
            coe_3[i][0] = round(coe_3[i][0].real,4)
            print(coe_3_sym[i],"=",coe_3[i][0].real,";")
            
        error_print = "{:.2e}".format(error)
        all_dif_print = "{:.2e}".format(all_dif)
            
        print("Error of the linear least square method:",error_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
    
    sum_result_kpmodel_3 = sp.zeros(dimension)
    
    for i in range(len(kpmodel)):
        sum_result_kpmodel_3 = sum_result_kpmodel_3 + coe_3[i,0]*kpmodel[i]
    
    return coe_3,coe_3_sym,sum_result_kpmodel_3,error,all_dif
###############################################################################



###############################################################################        
def get_Zeeman(Symmetry,msg_num,kvec,G_4c4,order = 0,print_flag = 2,log = 0):
    '''
    construct the standard Zeeman's coupling

    Parameters
    ----------
    Symmetry : dictionary
        a dictionary according to the user's input, the rotation matrices, the standard corepresentation matrices of all generators
    msg_num : decimal or integer, optional
        user's input, magnetic space group number. You can set it to None directly.
    kvec : list, optional
        user's input, the high symmetry point coordinate. You can set it to None directly.
    G_4c4 : numpy.array
        the numerical standard Zeeman's coupling obtained by VASP
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    Error_FlagChooseError
        If error_flag is not in [0,1,2]: raise this error.
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    sum_result_kpmodel_g : sympy.matrices.dense.MutableDenseMatrix
        the effective Zeeman's coupling.
    Zeeman_coe : a dictionary
        keys: parameters' name, TYPE: sympy.core.symbol.Symbol
            You can use list(Zeeman_coe.keys()) to get the list of keys and then get the variable name.
        values: a list, 
            The zeroth element is the value of the parameters.
            The first element is the matrix of the independent Zeeman's coupling. 

    '''
    
    if error_flag not in [0,1,2]:
        raise Error_FlagChooseError("The parameter error_flag in _standard_kp must be in [0,1,2]!")
        
        sys.exit(0)
    
    # define the symbol variables(constant)
    Bx,By,Bz = sp.symbols("Bx By Bz")
    variable_list = sp.symbols("Bx By Bz")
    Bx_,By_,Bz_= sp.symbols("B_x B_y B_z",commutative = False)
    id_B=[Bx,By,Bz]
    char_B = "Bx By Bz".split(' ')
    
    if print_flag == 2 and order !=0:
        texfile_name = "kp_Hamiltonian-Zeeman's_coupling"
        
        texfile = open(texfile_name+".tex","a")
    
    if print_flag == 1:
        print('\n\n\n')
    
    # input of kdotp-generator 
    sym_ops = list(Symmetry.values())
    save_stdout = sys.stdout
    
    if log != 0:
        logging = open("independent_Zeeman's_coupling(without numerical parameters).log","w")
        logging.write('\nPrint Zeeman\'s coupling results:')
        sys.stdout = logging
        
    else:
        # set the output stream to empty
        
        # judge what the system is
        import platform
        system = platform.system()
        
        if system == "Windows":
            logging = open('nul', 'w')
            
        elif system == "Linux":
            logging = open('/dev/null', 'w')
            
        elif system == "Darwin": # mac
            logging = open('/dev/null', 'w')
            
        else:
            logging = open('/dev/null', 'w')
            
        sys.stdout = logging
        
    #dimension=sym_ops[0]['repr_matrix'].shape[0]
    dimension = G_4c4.shape[0]
    #result_Zeeman_no_coe = sp.zeros(dimension)

    if print_flag == 2:
        outfile = open("g-factors.out","w")
        
    if print_flag != 0:
        print('\n==========  Result of Zeeman\'s coupling  ==========')
    
    # fixing the bug in kdotp-generator
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        np.complex
        warnings.resetwarnings()
        
    except:
        warnings.resetwarnings()
        np.complex = np.complex128
        
    # use kdotp-generator to generate the analytical Zeeman's coupling based on invariant theory
    
    kpmodel, irrep_expr_basis, irrep_repr_basis = kp.symmetric_hamiltonian(
                sym_ops,
                kp_variable = 'B',
                order = [1],
                repr_basis = 'pauli',
                msg_num = msg_num,
                kvec = kvec
            )   
    
    if print_flag == 2:
        sys.stdout = outfile
        
    else:
        sys.stdout = save_stdout
     
    # removal of duplicates
    if False:
        kpmodel_delete = []
        k = sp.symbols('k')
        
        for i in range(len(kpmodel)):
            
            if len(kpmodel_delete) == 0:
                kpmodel_delete.append(kpmodel[i])
                
            else:
                add_flag = 1
                
                for j in kpmodel_delete:
                    
                    equation = sp.Eq(kpmodel[i], k * j)
                    flag = sp.solve(equation, k)
                    
                    if flag:
                        
                        if flag[k]!=0:
                            
                            try:
                                float(flag[k])
                                add_flag = 0
                                #print(i,j)
                                break
                                
                            except:
                                pass
                            
                if add_flag == 1:
                    kpmodel_delete.append(kpmodel[i])
    
    # define symbol variables of the undetermined paramters
    coe_g_num = len(kpmodel) # the number of independent Zeeman's coupling terms
    coe_g_sym = sp.symbols('g1:{}'.format(coe_g_num+1))
    
    dimension = G_4c4[0].shape[0]
    
    sum_kpmodel_g=sp.zeros(dimension)
    
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        sum_kpmodel_g_tex = sp.matrices.immutable.ImmutableDenseMatrix(sp.zeros(dimension))
        warnings.resetwarnings()
        
    except:
        warnings.resetwarnings()
        sum_kpmodel_g_tex = sp.zeros(dimension)
    
    Zeeman_coe=dict()
    
    
    # Put the variables after the coefficients in expression
    variable_basis = []
    
    for i in range(3):
            
        variable_basis.append(variable_list[i])
            
    variable_basis = tuple(variable_basis)
    
    kpmodel_tex = []
    
    for i in range(len(kpmodel)):
        #tmp = kpmodel[i].copy()
        kpmodel[i]=sp.simplify(kpmodel[i])
        #print(sp.simplify(kpmodel[i]-tmp))
        
        shape = kpmodel[i].shape
        
        tmp_kp = sp.zeros(*shape)
        tmp_kp_latex = sp.zeros(*shape)
        
        for ii in range(shape[0]):
            
            for jj in range(shape[1]):
                
                #print(kpmodel[i][ii,jj])
                tmp_ = sp.expand(kpmodel[i][ii,jj])
                tmp_ = sp.collect(tmp_, variable_basis)
                tmp_kp[ii,jj] = tmp_
                tmp_kp_latex_ele = 0
                
                for l in variable_basis:
                    coeff = tmp_.coeff(l)
                    tmp_kp_latex_ele += coeff*(l.subs(Bz,Bz_).subs(By,By_).subs(Bx,Bx_))
                    
                tmp_kp_latex[ii,jj] = tmp_kp_latex_ele
                #print(tmp_kp_latex_ele)
                
        
        #print(sp.simplify(tmp-tmp_kp))
        #print(sp.simplify(tmp-tmp_kp_latex.subs(kx_,kx).subs(ky_,ky).subs(kz_,kz)))
        
        kpmodel[i] = tmp_kp
        kpmodel_tex.append(tmp_kp_latex)
        
    for i in range(len(kpmodel)):
        sum_kpmodel_g = sum_kpmodel_g + coe_g_sym[i]*kpmodel[i]
        sum_kpmodel_g_tex += coe_g_sym[i]*kpmodel_tex[i]
    
    #print(sp.simplify(sum_kpmodel_g_tex.subs(Bx_,Bx).subs(By_,By).subs(Bz_,Bz)-sum_kpmodel_g))
    
    wrong_flag = 0
        
    ###########################################################################
    # construct the linear equation set
    eql=[]
    all_dif = 0
    
    for ii in range(3):
        
        tmp_sum_kpmodel_g = sum_kpmodel_g.copy()
        tmp_sum_kpmodel_g = tmp_sum_kpmodel_g.subs(id_B[ii], 1)
        wrong_flag_per = 0
        
        for l in range(3):
            tmp_sum_kpmodel_g = tmp_sum_kpmodel_g.subs(id_B[l], 0)
            
        G_4c4_part = G_4c4[:,:,ii]
        
        for i in range(dimension):
            
            for j in range(dimension):
                
                ele = sp.Eq(tmp_sum_kpmodel_g[i,j]-G_4c4_part[i,j],0)
                
                ###############################################################
                # calculate the errors of the matrix elements whose theoretical value is zero
                if (ele is sp.false) or (ele is sp.true):
                    # calculate the error
                    dif = abs(G_4c4_part[i,j])
                    all_dif += dif
                    
                    if dif > tol["Zeeman"]:
                        
                        if error_flag == 0:
                            pass
                        
                        elif error_flag == 1:
                            sys.stdout = save_stdout
                            
                            if log == 1:
                                logging.close()
                            
                            if print_flag == 2:
                                outfile.close()
                            
                            raise NumericalStandardKpMatrixError("The numerical Zeeman's coupling and the analytical Zeeman's coupling do not match!")
                            
                            sys.exit()
                            
                        elif error_flag == 2:
                            
                            if log == 1:
                                
                                if wrong_flag == 0:
                                    error_file = open("Matrix_error.log","a")
                                    error_file.write('\n\n==========  Error of Zeeman\'s coupling matrix  ==========\n')
                                    error_file.write('========== {} ==========\n'.format(char_B[ii]))
                                    error_file.write("VASP_element \t\t Difference\n")
                                    error_file.write(str(G_4c4_part[i,j])+'\t'+str(dif)+'\n')
                                    wrong_flag = 1
                                    wrong_flag_per = 1
                                    
                                elif wrong_flag_per == 0:
                                    error_file.write('\n========== {} ==========\n'.format(char_B[ii]))
                                    error_file.write("VASP_element \t\t Difference\n")
                                    error_file.write(str(G_4c4_part[i,j])+'\t'+str(dif)+'\n')
                                    wrong_flag_per = 1
                                    
                                else:
                                    error_file.write(str(G_4c4_part[i,j])+'\t'+str(dif)+'\n')
                                    
                            else:
                                tmp_stdout = sys.stdout
                                sys.stdout.flush()
                                sys.stdout = save_stdout
                                
                                if wrong_flag == 0:
                                    print('\n\n==========  Error of Zeeman\'s coupling matrix  ==========')
                                    print('========== {} =========='.format(char_B[ii]))
                                    print("VASP_element \t\t Difference")
                                    print(str(G_4c4_part[i,j])+'\t'+str(dif))
                                    wrong_flag = 1
                                    wrong_flag_per = 1
                                
                                elif wrong_flag_per == 0:
                                    print('\n========== {} =========='.format(char_B[ii]))
                                    print("VASP_element \t\t Difference")
                                    print(str(G_4c4_part[i,j])+'\t'+str(dif))
                                    wrong_flag_per = 1
                                
                                else:
                                    print(str(G_4c4_part[i,j])+'\t'+str(dif))
                                
                                sys.stdout.flush()
                                sys.stdout = tmp_stdout
                                    
                                    
                    continue
                
                eql.append(ele)
                
        if wrong_flag_per == 1:
            
            if log == 1:
                error_file.write('---------- VASP matrix for {} ----------\n'.format(char_B[ii]))
                error_file.write(str(sp.Matrix(G_4c4_part))+'\n')
                error_file.write('---------- Zeeman\'s coupling matrix for {} ----------\n'.format(char_B[ii]))
                error_file.write(str(tmp_sum_kpmodel_g)+'\n')
                
            else:
                sys.stdout.flush()
                tmp_stdout = sys.stdout
                sys.stdout = sys.__stdout__
                
                print('---------- VASP matrix for {}----------'.format(char_B[ii]))
                print(sp.Matrix(G_4c4_part))
                print('---------- Zeeman\'s coupling matrix for {} ----------'.format(char_B[ii]))
                print(tmp_sum_kpmodel_g)
                
                sys.stdout.flush()
                sys.stdout = tmp_stdout
    
    
    if len(kpmodel) == 0:  # reduce calculation time
    
        print("No symmetry-allowed Zeeman's coupling.")
        
        #sys.stdout = save_stdout
        
        if print_flag == 2:
            
            if order == 0:
                texfile_name = "Zeeman's_coupling"
                    
                texfile = open(texfile_name+".tex","w")
                
                # begin lines in .tex
                texfile.write('\\documentclass[aps,amssymb,onecolumn]{revtex4}\n')
                
                texfile.write('\\usepackage{breqn}\n')
                
                texfile.write('\\usepackage{hyperref}\n\n')
                
                texfile.write('\n\\begin{document}\n\n')
                
            texfile.write("\\section{Zeeman's coupling}\n\n")
            
            texfile.write("No symmetry-allowed Zeeman's coupling.")
            
            texfile.write(r'\end{document}')
            
            texfile.close()
            
            # if the system has the command 'pdflatex': compile the .tex
            try:
                import platform
                
                system = platform.system()
                
                if system == 'Windows':
                    if order !=0:
                        pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian-Zeeman's_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                    else:
                        pdf_flag = subprocess.run(r"pdflatex Zeeman's_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                else:
                    if order !=0:
                        pdf_flag = subprocess.run("pdflatex kp_Hamiltonian-Zeeman\'s_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    else:
                        pdf_flag = subprocess.run("pdflatex Zeeman\'s_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                if pdf_flag.returncode == 0:
                    
                    try:
                        os.remove(texfile_name+".log")
                        os.remove(texfile_name+".aux")
                        os.remove(texfile_name+".out")
                        
                    except:
                        pass
                    
                else:
                    pass
                
            except:
                pass
            
            all_dif_print = "{:.2e}".format(all_dif)
            print("Sum of absolute values of numerical zero elements:",all_dif_print)
            
            outfile.close()
            
        if log == 0:
            logging.close()
        
        if log == 1:
            logging.close()
            
            if wrong_flag == 1:
                error_file.close()
        
        sys.stdout = save_stdout
        
        return 0,dict()
    
    
    
    # calculate equations by the linear least squares method
    coe_g, residuals=get_coeff(eql,coe_g_sym)
    error = residuals[0]
    
    ###########################################################################
    # output the result
     
        
    if print_flag != 0:
        print("Zeeman\'s coupling")
        print('==========  Result of Zeeman\'s coupling  ==========')
        print(sum_kpmodel_g)
        print("Parameters:")
        
        # write to a .out file or print to screen
        for i in range(coe_g_num):
            coe_g[i][0]=round(coe_g[i][0].real,4)
            print(coe_g_sym[i],"=",coe_g[i][0].real,";")
            
        error_print = "{:.2e}".format(error)
        all_dif_print = "{:.2e}".format(all_dif)
        
        print("Error of the linear least square method:",error_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
        
        if print_flag == 2:
            
            # write to a .tex source file
            if order == 0:
                texfile_name = "Zeeman's_coupling"
                    
                texfile = open(texfile_name+".tex","w")
                
                # begin lines in .tex
                texfile.write('\\documentclass[aps,amssymb,onecolumn]{revtex4}\n')
                
                texfile.write('\\usepackage{breqn}\n')
                
                texfile.write('\\usepackage{hyperref}\n\n')
                
                texfile.write('\n\\begin{document}\n\n')
                
            texfile.write("\\section{Zeeman's coupling}\n\n")
            
            texfile.write("\\subsection{Hamiltonian}\n\n")
            
            texfile.write("\\begin{dgroup*}\n")
            
            coe_g_sym_ = sp.symbols('g_1:{}'.format(coe_g_num+1))
            
            for i in range(sum_kpmodel_g_tex.shape[0]):
                
                for j in range(i,sum_kpmodel_g_tex.shape[0]):
                    
                    # set subscript
                    # ele = sum_kpmodel_g_tex[i,j].subs(Bx,Bx_).subs(By,By_).subs(Bz,Bz_)
                    ele = sum_kpmodel_g_tex[i,j]
                    
                    for k in range(coe_g_num):
                        ele = ele.subs(coe_g_sym[k],coe_g_sym_[k])
                        
                    ele_string = sp.latex(ele)
                    ele_string = ele_string.replace("\\cdot","")
                    
                    texfile.write("\\begin{dmath}\n")
                    #texfile.write("\\noindent ")
                    #texfile.write("$ H^{Z}_{"+str(i+1)+str(j+1)+"}/\\mu_B = "+ele_string+" $;\n\n~\\\\\n\n")
                    
                    texfile.write("\tH^{Z}_{"+str(i+1)+str(j+1)+"}/\\mu_B = "+ele_string+"\n")
                    texfile.write("\\end{dmath}\n\n")
            
            texfile.write("\\end{dgroup*}\n")
            texfile.write("\n")
            
            texfile.write("\\subsection{Parameters}\n\n")
            
                
            for i in range(coe_g_num):
                texfile.write("\\noindent ")
                texfile.write("$ "+sp.latex(coe_g_sym[i])+" = "+str(coe_g[i][0].real)+' $;\n\n')
            
            texfile.write(r'\end{document}')
            
            texfile.close()
            
            # if the system has the command 'pdflatex': compile the .tex
            try:
                
                import platform
                
                system = platform.system()
                
                if system == 'Windows':
                    if order !=0:
                        pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian-Zeeman's_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    else:
                        pdf_flag = subprocess.run(r"pdflatex Zeeman's_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                else:
                    if order!=0:
                        pdf_flag = subprocess.run("pdflatex kp_Hamiltonian-Zeeman\\'s_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    else:
                        pdf_flag = subprocess.run(r"pdflatex Zeeman\'s_coupling.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                #delete the compile log
                if pdf_flag.returncode == 0:
                    
                    try:
                        os.remove(texfile_name+".log")
                        os.remove(texfile_name+".aux")
                        os.remove(texfile_name+".out")
                        
                    except:
                        pass
                    
                else:
                    pass
                
            except:
                pass
        
    sum_result_kpmodel_g = sp.zeros(dimension)
    
    
        
    for i in range(len(kpmodel)):
        Zeeman_coe[coe_g_sym[i]]=[coe_g[i,0],kpmodel[i],irrep_expr_basis[i],irrep_repr_basis[i]]
        sum_result_kpmodel_g = sum_result_kpmodel_g + coe_g[i,0]*kpmodel[i]  
 
    if log == 0:
        logging.close()
    
    if log == 1:
        logging.close()
        
        if wrong_flag == 1:
            error_file.close()
        
    if print_flag == 2:
        outfile.close()
        
    sys.stdout = save_stdout
    
    return sum_result_kpmodel_g,Zeeman_coe
###############################################################################



###############################################################################
def get_std_kp(Symmetry,order,gfactor,msg_num,kvec,numeric_kp,print_flag=2,log = 0):
    '''
    construct the standard kp hamiltonian of the certain order 

    Parameters
    ----------
    Symmetry : dictionary
        a dictionary according to the user's input, the rotation matrices, the standard corepresentation matrices of all generators
    order : integer
        the order of kp Hamiltonian to be constructed.
    gfactor : integer, optional
        Whether to solve the Zeeman‘s coupling. 
    msg_num : decimal or integer
        user's input, magnetic space group number. You can set it to None directly.
    kvec : list
        user's input, the high symmetry point coordinate. You can set it to None directly.
    numeric_kp : dictionary
        the numerical standard kp Hamiltonian as well as the numerical standard Zeeman's coupling.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the result of finding the similarity transformation matrices,
                                                        the independent of each order as well as some warnings.
        The default is 0.

    Raises
    ------
    Error_FlagChooseError
        If error_flag is not in [0,1,2]: raise this error.
    NumericalStandardKpMatrixError
        If the error is larger than tol and error_flag is set to be 1, raise this error(error too large)

    Returns
    -------
    result_kp_array : sympy.matrices.dense.MutableDenseMatrix
        the effective kp model(the sum of the effective kp models of each order as well as their parameters).
    kpmodel_coe : a dictionary
        keys: parameters' name, TYPE: sympy.core.symbol.Symbol
            You can use list(kpmodel_coe.keys()) to get the list of keys and then get the variable name.
        values: a list, 
            The zeroth element is the value of the parameters.
            The first element is the matrix of the independent kp hamiltonian. 

    '''
    
    # simple judgement
    if error_flag not in [0,1,2]:
        raise Error_FlagChooseError("The parameter error_flag in _standard_kp must be in [0,1,2]!")
        
        sys.exit(0)
    
    # update log file
    if error_flag == 2:
        file_list = os.listdir()
        
        if "Matrix_error.log" in file_list:
            os.remove("Matrix_error.log")
    
    save_stdout = sys.stdout
    
    order_list = [ [i] for i in range(order+1) ]
    sym_ops = list(Symmetry.values()) # kdotp-generator input parameter
    dimension=sym_ops[0]['repr_matrix'].shape[0]
    result_kp_array = sp.zeros(dimension)
    result_res = 0
    all_dif = 0
    #result_kp = []
    result_kp_no_coe = sp.zeros(dimension)
    
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result_kp_no_coe_tex = sp.matrices.immutable.ImmutableDenseMatrix(sp.zeros(dimension))
        warnings.resetwarnings()

    except:
        warnings.resetwarnings()
        result_kp_no_coe_tex = sp.zeros(dimension)
        
    kpmodel_coe = dict()
    kx,ky,kz = sp.symbols("kx ky kz")
    kx_,ky_,kz_= sp.symbols("k_x k_y k_z",commutative = False)
    variable_list = [kx,ky,kz]
    
    # redirect the output stream
    if log != 0:
        logging = open("independent_kp_models(without numerical parameters).log","w")
        logging.write('\nPrint kp results:')
        
    else:
        # set the output stream to empty
        
        # judge what the system is
        import platform
        system = platform.system()
        
        if system == "Windows":
            logging = open('nul', 'w')
            
        elif system == "Linux":
            logging = open('/dev/null', 'w')
            
        elif system == "Darwin":
            logging = open('/dev/null', 'w')
            
        else:
            logging = open('/dev/null', 'w')
        
    if print_flag == 2:
        if print_each_order_flag:
            outfile = open("kp-parameters_each_order.out","w")
            
        else:
            outfile = open("kp-parameters.out","w")
        
        if print_each_order_flag:
            outfile.write("kp Hamiltonian of each order")
    
    ###########################################################################
    # generate the analytical kp hamiltonian based on invariant theory and 
    #   calculate the parameters to get the standard kp hamiltonian
    
    coe_sym = []
    coe_sym_ = []
    
    # fixing the bug in kdotp-generator
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        np.complex
        warnings.resetwarnings()
        
    except:
        warnings.resetwarnings()
        np.complex = np.complex128
    
    for order in order_list:
        
        # Redirect the output stream to the log file 
        # the function kp.symmetric_hamiltonian will output some information
        sys.stdout = logging
        
        print('\n==========  Result of order = %s  =========='%(str(order)))
        
        # use kdotp-generator to generate the analytical kp hamiltonian based on invariant theory
        kpmodel, irrep_expr_basis, irrep_repr_basis = kp.symmetric_hamiltonian(
                sym_ops,
                kp_variable = 'k',
                order = order,
                repr_basis = 'pauli',
                msg_num = msg_num,
                kvec = kvec
            )
        
            
        
        if print_flag == 2:
            sys.stdout = outfile
            
        else:
            sys.stdout = save_stdout
        
        
        
        if print_each_order_flag and print_flag != 0:
            print('\n==========  Result of order = %s  =========='%(str(order)))
        
        
        # removal of duplicates
        if False:
            kpmodel_delete = []
            k = sp.symbols('k')
            
            for i in range(len(kpmodel)):
                
                if len(kpmodel_delete) == 0:
                    kpmodel_delete.append(kpmodel[i])
                    
                else:
                    add_flag = 1
                    
                    for j in kpmodel_delete:
                        
                        
                        equation = sp.Eq(kpmodel[i], k * j)
                        flag = sp.solve(equation, k)
                        
                        if flag:
                            
                            if flag[k]!=0:
                                
                                try:
                                    float(flag[k])
                                    add_flag = 0
                                    #print(i,j)
                                    break
                                    
                                except:
                                    pass
                                
                    if add_flag == 1:
                        kpmodel_delete.append(kpmodel[i])
                        
        #######################################################################
        # 0-order
        if order[0]==0:
            eigen_trans = numeric_kp["1"]
            
            try:
                sys.stdout.flush()
                coe_0,coe_0_sym,sum_result_kpmodel_0,res0,all_dif0 = get_std_kp_0_order(kpmodel,eigen_trans,print_flag,log)
                all_dif += all_dif0
                sys.stdout.flush()
                
                # cannot find any analytical kp
                if len(coe_0) == 0:
                    
                    if print_each_order_flag:
                        
                        all_dif0_print = "{:.2e}".format(all_dif0)
                        print("No symmetry-allowed kp models.")
                        print("Sum of absolute values of numerical zero elements:",all_dif0_print)
                    
                    continue
                
            except:
                sys.stdout = save_stdout
                
                if print_flag == 2:
                    outfile.close()
                    
                logging.close()
                
                raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                sys.exit()
                
            #result_kp.append(sum_result_kpmodel_0)
            result_kp_array = result_kp_array + sum_result_kpmodel_0
            result_res += res0
            all_dif += all_dif0
            
            for i in range(len(kpmodel)):
                #tmp = kpmodel[i].copy()
                kpmodel[i]=sp.simplify(kpmodel[i])
                #print(sp.simplify(kpmodel[i]-tmp))
                
            # set subscript
            coe_0_sym_ = sp.symbols('a_1:{}'.format(len(coe_0)+1))
            coe_sym_ += coe_0_sym_
            
            coe_sym += coe_0_sym
            
            for i in range(len(coe_0)):
                kpmodel_coe[coe_0_sym[i]]=[coe_0[i,0],kpmodel[i],irrep_expr_basis[i],irrep_repr_basis[i]]
                
                if print_flag != 0:
                    result_kp_no_coe += coe_0_sym[i]*kpmodel[i]
                    result_kp_no_coe_tex += coe_0_sym[i]*kpmodel[i]
            
        #######################################################################
        # 1-order
        if order[0]==1:
            linear_V2 = numeric_kp["k"]
            
            try:
                sys.stdout.flush()
                coe_1,coe_1_sym,sum_result_kpmodel_1,res1,all_dif1 = get_std_kp_1_order(kpmodel,linear_V2,print_flag,log)
                all_dif += all_dif1
                sys.stdout.flush()
                
                # cannot find any analytical kp
                if len(coe_1) == 0:
                    
                    if print_each_order_flag:
                        print("No symmetry-allowed kp models.")
                        all_dif1_print = "{:.2e}".format(all_dif1)
                        print("Sum of absolute values of numerical zero elements:",all_dif1_print)
                    
                    continue
                
            except:
                sys.stdout = save_stdout
                
                if print_flag == 2:
                    outfile.close()
                    
                logging.close()
                
                raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                sys.exit()
                
            #result_kp.append(sum_result_kpmodel_1)
            result_kp_array = result_kp_array + sum_result_kpmodel_1
            result_res += res1
            
            variable_basis = []
            
            for i in range(3):
                    
                variable_basis.append(variable_list[i])
                    
            variable_basis = tuple(variable_basis)

            kpmodel_tex = []
            
            for i in range(len(kpmodel)):
                #tmp = kpmodel[i].copy()
                kpmodel[i]=sp.simplify(kpmodel[i])
                #print(sp.simplify(kpmodel[i]-tmp))
                
                shape = kpmodel[i].shape
                
                tmp_kp = sp.zeros(*shape)
                tmp_kp_latex = sp.zeros(*shape)
                
                for ii in range(shape[0]):
                    
                    for jj in range(shape[1]):
                        
                        #print(kpmodel[i][ii,jj])
                        tmp_ = sp.expand(kpmodel[i][ii,jj])
                        tmp_ = sp.collect(tmp_, variable_basis)
                        tmp_kp[ii,jj] = tmp_
                        tmp_kp_latex_ele = 0
                        
                        for l in variable_basis:
                            coeff = tmp_.coeff(l)
                            tmp_kp_latex_ele += coeff*(l.subs(kz,kz_).subs(ky,ky_).subs(kx,kx_))
                            
                        tmp_kp_latex[ii,jj] = tmp_kp_latex_ele
                        
                
                #print(sp.simplify(tmp-tmp_kp))
                #print(sp.simplify(tmp-tmp_kp_latex.subs(kx_,kx).subs(ky_,ky).subs(kz_,kz)))
                
                kpmodel[i] = tmp_kp
                kpmodel_tex.append(tmp_kp_latex)
                
            # set subscript
            coe_1_sym_ = sp.symbols('b_1:{}'.format(len(coe_1)+1))
            coe_sym_ += coe_1_sym_
            
            coe_sym += coe_1_sym
            
            for i in range(len(coe_1)):
                kpmodel_coe[coe_1_sym[i]]=[coe_1[i,0],kpmodel[i],irrep_expr_basis[i],irrep_repr_basis[i]]
                
                if print_flag != 0:
                    result_kp_no_coe += coe_1_sym[i]*kpmodel[i]
                    
                    result_kp_no_coe_tex += coe_1_sym[i]*kpmodel_tex[i]
        
        #######################################################################
        # 2-order
        if order[0]==2:
            quadratic_V2_symm = numeric_kp["k^2"]
            
            try:
                sys.stdout.flush()
                coe_2,coe_2_sym,sum_result_kpmodel_2,res2,all_dif2 = get_std_kp_2_order(kpmodel,quadratic_V2_symm,print_flag,log)
                all_dif += all_dif2
                sys.stdout.flush()
                
                # cannot find any analytical kp
                if len(coe_2) == 0:
                    
                    all_dif2_print = "{:.2e}".format(all_dif2)
                    print("No symmetry-allowed kp models.")
                    print("Sum of absolute values of numerical zero elements:",all_dif2_print)
                    
                    continue
                
            except:
                
                sys.stdout = save_stdout
                
                if print_flag == 2:
                    outfile.close()
                    
                logging.close()
                
                raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                sys.exit()
                
            #result_kp.append(sum_result_kpmodel_2)
            result_kp_array = result_kp_array + sum_result_kpmodel_2
            result_res += res2
            
            variable_basis = []
            
            for i in range(3):
                
                for j in range(i,3):
                    
                    variable_basis.append(variable_list[i]*variable_list[j])
                    
            variable_basis = tuple(variable_basis)

            kpmodel_tex = []
            
            for i in range(len(kpmodel)):
                #tmp = kpmodel[i].copy()
                kpmodel[i]=sp.simplify(kpmodel[i])
                #print(sp.simplify(kpmodel[i]-tmp))
                
                shape = kpmodel[i].shape
                
                tmp_kp = sp.zeros(*shape)
                tmp_kp_latex = sp.zeros(*shape)
                
                for ii in range(shape[0]):
                    
                    for jj in range(shape[1]):
                        
                        #print(kpmodel[i][ii,jj])
                        tmp_ = sp.expand(kpmodel[i][ii,jj])
                        tmp_ = sp.collect(tmp_, variable_basis)
                        tmp_kp[ii,jj] = tmp_
                        tmp_kp_latex_ele = 0
                        
                        for l in variable_basis:
                            coeff = tmp_.coeff(l)
                            tmp_kp_latex_ele += coeff*(l.subs(kz,kz_).subs(ky,ky_).subs(kx,kx_))
                            
                        tmp_kp_latex[ii,jj] = tmp_kp_latex_ele
                        
                
                #print(sp.simplify(tmp-tmp_kp))
                #print(sp.simplify(tmp-tmp_kp_latex.subs(kx_,kx).subs(ky_,ky).subs(kz_,kz)))
                
                kpmodel[i] = tmp_kp
                kpmodel_tex.append(tmp_kp_latex)

                
                
            # set subscript
            coe_2_sym_ = sp.symbols('c_1:{}'.format(len(coe_2)+1))
            coe_sym_ += coe_2_sym_
            
            coe_sym += coe_2_sym
            
            for i in range(len(coe_2)):
                kpmodel_coe[coe_2_sym[i]]=[coe_2[i,0],kpmodel[i],irrep_expr_basis[i],irrep_repr_basis[i]]
                
                if print_flag != 0:
                    result_kp_no_coe += coe_2_sym[i]*kpmodel[i]
                    result_kp_no_coe_tex += coe_2_sym[i]*kpmodel_tex[i]
        
        #######################################################################
        # 3-order
        if order[0] == 3:
            cubic_V2_symm = numeric_kp["k^3"]
            
            try:
                sys.stdout.flush()
                coe_3,coe_3_sym,sum_result_kpmodel_3,res3,all_dif3 = get_std_kp_3_order(kpmodel,cubic_V2_symm,print_flag,log)
                all_dif += all_dif3
                sys.stdout.flush()
                
                # cannot find any analytical kp
                if len(coe_3) == 0:
                    
                    if print_each_order_flag:
                        
                        all_dif3_print = "{:.2e}".format(all_dif3)
                        print("No symmetry-allowed kp models.")
                        print("Sum of absolute values of numerical zero elements:",all_dif3_print)
                    
                    continue
            
            except:
                sys.stdout = save_stdout
                
                if print_flag == 2:
                    outfile.close()
                    
                if log != 0:
                    logging.close()
                    
                raise NumericalStandardKpMatrixError("The numerical kp model and the analytical kp model do not match!")
                
                sys.exit()
                
            #result_kp.append(sum_result_kpmodel_3)
            result_kp_array = result_kp_array + sum_result_kpmodel_3
            result_res += res3
            
            variable_basis = []
            
            for i in range(3):
                
                for j in range(i,3):
                    
                    for k in range(j,3):
                        
                        variable_basis.append(variable_list[i]*variable_list[j]*variable_list[k])
                    
            variable_basis = tuple(variable_basis)

            
            kpmodel_tex = []
            
            for i in range(len(kpmodel)):
                #tmp = kpmodel[i].copy()
                kpmodel[i]=sp.simplify(kpmodel[i])
                #print(sp.simplify(kpmodel[i]-tmp))
                
                shape = kpmodel[i].shape
                
                tmp_kp = sp.zeros(*shape)
                tmp_kp_latex = sp.zeros(*shape)
                
                for ii in range(shape[0]):
                    
                    for jj in range(shape[1]):
                        
                        #print(kpmodel[i][ii,jj])
                        tmp_ = sp.expand(kpmodel[i][ii,jj])
                        tmp_ = sp.collect(tmp_, variable_basis)
                        tmp_kp[ii,jj] = tmp_
                        tmp_kp_latex_ele = 0
                        
                        for l in variable_basis:
                            coeff = tmp_.coeff(l)
                            tmp_kp_latex_ele += coeff*(l.subs(kz,kz_).subs(ky,ky_).subs(kx,kx_))
                            
                        tmp_kp_latex[ii,jj] = tmp_kp_latex_ele
                        
                
                #print(sp.simplify(tmp-tmp_kp))
                #print(sp.simplify(tmp-tmp_kp_latex.subs(kx_,kx).subs(ky_,ky).subs(kz_,kz)))
                
                kpmodel[i] = tmp_kp
                kpmodel_tex.append(tmp_kp_latex)
            
            #set subscript
            coe_3_sym_ = sp.symbols('d_1:{}'.format(len(coe_3)+1))
            coe_sym_ += coe_3_sym_
            
            coe_sym += coe_3_sym
            
            for i in range(len(coe_3)):
                kpmodel_coe[coe_3_sym[i]]=[coe_3[i,0],kpmodel[i],irrep_expr_basis[i],irrep_repr_basis[i]]
                
                if print_flag != 0:
                    result_kp_no_coe += coe_3_sym[i]*kpmodel[i]
                    result_kp_no_coe_tex += coe_3_sym[i]*kpmodel_tex[i]
         
    ###########################################################################
    # output the result
    
    #print(sp.simplify(result_kp_no_coe_tex.subs(kx_,kx).subs(ky_,ky).subs(kz_,kz)-result_kp_no_coe))
    if print_each_order_flag:
        print("\n\n\nkp Hamiltonian")
        
    else:
        print("kp Hamiltonian")
        
    print('==========  Result of kp Hamiltonian  ==========')
    
    if len(kpmodel_coe) == 0:
        
        all_dif_print = "{:.2e}".format(all_dif)
        print("No symmetry-allowed kp models.")
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
    
    else:
        outfile_result = sp.simplify(result_kp_no_coe)
        print(outfile_result)
        print("Parameters:")
        
        for i in kpmodel_coe.keys():
            kpmodel_coe[i][0]=round(kpmodel_coe[i][0].real,4)
            print(i,"=",kpmodel_coe[i][0].real,';')
        
        result_res_print = "{:.2e}".format(result_res)
        all_dif_print = "{:.2e}".format(all_dif)
        print("Error of the linear least square method:",result_res_print)
        print("Sum of absolute values of numerical zero elements:",all_dif_print)
        
        if print_flag == 2:
            
            
            if gfactor == 0:
                texfile_name = "kp_Hamiltonian"
                
            else:
                texfile_name = "kp_Hamiltonian-Zeeman's_coupling"
                
            texfile = open(texfile_name+".tex","w")
            
            # begin lines in .tex
            texfile.write('\\documentclass[aps,amssymb,onecolumn]{revtex4}\n')
            
            texfile.write('\\usepackage{breqn}\n')
            
            texfile.write('\\usepackage{hyperref}\n\n')
            
            texfile.write('\n\\begin{document}\n\n')
            
            texfile.write('\\section{\\texorpdfstring{$ k\\cdot p $}{} Hamiltonian}\n\n')
            
            texfile.write("\\subsection{Hamiltonian}\n\n")
            
            texfile.write("\\begin{dgroup*}\n")
            
            for i in range(result_kp_no_coe_tex.shape[0]):
                
                for j in range(i,result_kp_no_coe_tex.shape[0]):
                    
                    # set subscript
                    #ele = result_kp_no_coe[i,j].subs(kz,kz_).subs(ky,ky_).subs(kx,kx_)
                    ele = result_kp_no_coe_tex[i,j]
                    
                    for k in range(len(coe_sym)):
                        ele = ele.subs(coe_sym[k],coe_sym_[k])
                        
                    ele_string = sp.latex(ele)
                    #print(ele_string)
                    
                    # delete \cdot to apply dmath environment
                    ele_string = ele_string.replace("\\cdot","")
                    
                    texfile.write("\\begin{dmath}\n")
                    #texfile.write("\\noindent ")
                    #texfile.write("$ H^{kp}_{"+str(i+1)+str(j+1)+"} = "+ele_string+" $;\n\n~\\\\\n\n")
                    
                    texfile.write("\tH^{kp}_{"+str(i+1)+str(j+1)+"} = "+ele_string+"\n")
                    texfile.write("\\end{dmath}\n\n")
            
            texfile.write("\\end{dgroup*}\n")
            
            texfile.write("\n")
            
            texfile.write("\\subsection{Parameters}\n\n")
                          
            for i in kpmodel_coe.keys():
                #texfile.write("\\noindent ")
                
                
                formula_string = sp.latex(i)
                        
                texfile.write("\\noindent ")
                texfile.write("$ "+formula_string+" = "+str(kpmodel_coe[i][0].real)+' $;\n\n')
            
            # if Zeeman's coupling is not intended to be calculated, end the tex file
            if gfactor == 0:
                texfile.write(r'\end{document}')
            
            texfile.close()
            
            if gfactor == 0:

                # if the system has the command 'pdflatex': compile the .tex
                try:
                    import platform
                
                    system = platform.system()
                    
                    if system == 'Windows':
                        pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian.tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    else:
                        pdf_flag = subprocess.run("pdflatex "+texfile_name+".tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    #print(pdf_flag)
                    
                    #delete the compile log
                    if pdf_flag.returncode == 0:
                        
                        try:
                            os.remove(texfile_name+".log")
                            os.remove(texfile_name+".aux")
                            os.remove(texfile_name+".out")
                            
                        except:
                            pass
                        
                    else:
                        pass
                    
                except:
                    pass
    
            
    if print_flag == 2:
        
        outfile.close()
    
    sys.stdout = save_stdout
    
    if log == 0:
        logging.close()
    
    if log != 0:
        logging.close()
    
    return result_kp_array,kpmodel_coe
###############################################################################



###############################################################################
if __name__ == "__main__":
    pass



