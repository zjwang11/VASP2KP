# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:31:09 2023
Last modified on Wedn Dec 6 14:45:00 2023

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Calculate the transformation matrix between two corepresentations
"""

#import copy
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd
import sympy as sp
#import random
#import numba



###############################################################################
class CoRepresentationInequivalenceError(Exception):
    '''
    the VASP corepresentation are not equivalent to the standard corepresentation
    if traces of the corepresentation are not equal: raise this error
    '''
    pass
###############################################################################



###############################################################################
def null_space(A, rcond=None,tol=1e-9):
    """
    get the null space of matrix by svd
    
    Parameters
    ----------
    A : np.array/np.matrix 
        The matrix to be found the null space
    
    Returns
    —---------
    Q : np.array
        The independent vectors of the null space
    """

    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
        
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    
    return Q
###############################################################################



###############################################################################
def cal_nullspace_complex(matrix_in1,matrix_in2,is_TRS = False, Tmat = None,tol = 1e-9):
    """
    solve the equation X^{-1} A^{(i)} X = D^{(i)} and, X^{dagger}X = I
    
    Parameters
    ----------
    matrix_in1 : np.array((N,d,d))
        A^{(i)}, N is the total number of A^{(i)}.
    matrix_in2 : np.array((N,d,d))
        D^{(i)}, N is the total number of D^{(i)}.
    is_TRS : bool, optional
        if there is an antiunitary element. The default is False.
    Tmat : np.array((2,d,d)), optional
        Anti-unitary element. The default is None.
        Tmat[0,:,:]: A^{(i)}
        Tmat[1,:,:]: D^{(i)}
    tol : float, optional
        Tolerance error. The default is 1e-9.

    Returns
    -------
    ns_mat, np.array
        similarity transformation matrix
    cn, integer
        the number of columns
    row, integer
        the number of rows

    """
    import sys
    
    unitary_len = matrix_in1.shape[0]
    dim = matrix_in1.shape[1]
    
    trace_judge = True
    
    if trace_judge:
        
        for i in range(unitary_len):
            trace_dif = abs(np.trace(matrix_in1[i,:])-np.trace(matrix_in2[i,:]))
            # if the trace are not equal, the corepresentations are not equivelent
            if trace_dif > max(tol,5e-3):
                
                raise CoRepresentationInequivalenceError("The standard representation is not equivalent to VASP representation!(traces error)")
                sys.exit()
    
    if dim == 1:
        
        if is_TRS:
            m1 = Tmat[0,0,0]
            m2 = Tmat[1,0,0]
            ele = sp.sqrt(m1/m2)
            
            return np.array([[ele]],dtype = np.complex128),1,1
        
        else:
            return np.array([[1]],dtype = np.complex128),1,1
    
    
    if is_TRS:
        
        if Tmat is None:
            
            print("Please make sure unitary TRS martix exist!")
            sys.exit(0)
            
        num_of_A,row,col, = matrix_in1.shape
        num_of_A += 1
        A = np.zeros((num_of_A,row,col),dtype=complex)
        B = np.zeros((num_of_A,row,col),dtype=complex)
        A[:num_of_A - 1,:,:] = matrix_in1[:,:,:]
        B[:num_of_A - 1,:,:] = matrix_in2[:,:,:]
        A[num_of_A - 1,:,:] = Tmat[0,:,:]
        B[num_of_A - 1,:,:] = Tmat[1,:,:]
        
    else:
        #A = copy.deepcopy(matrix_in1)
        #B = copy.deepcopy(matrix_in2)
        
        A = matrix_in1.copy()
        B = matrix_in2.copy()
        
        num_of_A,row,col, = A.shape

    Ar = np.real(A) ; Ai = np.imag(A)
    Br = np.real(B) ; Bi = np.imag(B)
    
    ### condition of X^{-1} A^{(i)} X = D^{(i)}
    coff_of_X = np.zeros((2*num_of_A*row*row,2*row*row),dtype=complex)
    index1 = 0
    
    
    for num in range(0,num_of_A):
        #antiunitary
        if num == num_of_A - 1:
            
            if is_TRS:
                
                for i in range(0,row):
                    
                    for j in range(0, row):
                        
                        for k in range(0,row):
                            ### real part
                            coff_of_X[index1,k*row + j] +=  Ar[num,i,k]
                            coff_of_X[index1,i*row + k] += -Br[num,k,j]
                            coff_of_X[index1,k*row + j + row*row] +=  Ai[num,i,k]
                            coff_of_X[index1,i*row + k + row*row] +=  Bi[num,k,j]

                            ### imag part
                            coff_of_X[index1 + num_of_A*row*row,k*row + j] +=  Ai[num,i,k]
                            coff_of_X[index1 + num_of_A*row*row,i*row + k] += -Bi[num,k,j]
                            coff_of_X[index1 + num_of_A*row*row,k*row + j + row*row] += -Ar[num,i,k]
                            coff_of_X[index1 + num_of_A*row*row,i*row + k + row*row] += -Br[num,k,j]
                        
                        index1 += 1
                        
                continue
        
        #unitary
        for i in range(0,row):
            
            for j in range(0, row):
                
                for k in range(0,row):
                    ### real part
                    coff_of_X[index1,k*row + j] +=  Ar[num,i,k]
                    coff_of_X[index1,i*row + k] += -Br[num,k,j]
                    coff_of_X[index1,k*row + j + row*row] += -Ai[num,i,k]
                    coff_of_X[index1,i*row + k + row*row] +=  Bi[num,k,j]
                    
                    ### imag part
                    coff_of_X[index1 + num_of_A*row*row,k*row + j] +=  Ai[num,i,k]
                    coff_of_X[index1 + num_of_A*row*row,i*row + k] += -Bi[num,k,j]
                    coff_of_X[index1 + num_of_A*row*row,k*row + j + row*row] +=  Ar[num,i,k]
                    coff_of_X[index1 + num_of_A*row*row,i*row + k + row*row] += -Br[num,k,j]
                # print("ith = " + str(index1) + " , X = " + str(coff_of_X[index1,:]))
                
                index1 += 1
                # print(" ")

    #A = copy.deepcopy(coff_of_X)
    A = coff_of_X.copy()

    if False: # debug
        
        from scipy.io import savemat
        matlab_dict = dict()
        
        for i in range(unitary_len):
            matlab_dict['a'+str(i)] = matrix_in1[i,:,:]
            matlab_dict['b'+str(i)] = matrix_in2[i,:,:]
            
        if is_TRS is True:
            matlab_dict['c'] = Tmat[0,:,:]
            matlab_dict['d'] = Tmat[1,:,:]
        
        matlab_dict['A'] = A
        savemat('data.mat', matlab_dict)
        
        #print(matlab_dict)
        sys.exit()
    


    # from scipy.linalg import null_space
    ns = null_space(A,tol=tol)
    
    print("==================================================")
    print("The tolerence is set to be",tol)
    
    def transform_colC_to_matC(colC, dim ,tol = 1e-9):
        
        from functools import reduce
        # transfrom N^2 dim column vec C to N*N unitary matrix C
        # shape(C) =(dim^2,m), where m is the number of indepent null vec
        #assert np.shape(colC)[0] == dim**2 and np.shape(colC)[1] > 0, ('Wrong colC!',colC)
        assert np.shape(colC)[0] == dim**2 and np.shape(colC)[1] > 0, ('Fail to find unitary U!')
        
        trial_C_list = [ colC[:,ii].reshape((dim,dim)) for ii in range(np.shape(colC)[1])]
        trial_C_list.append(reduce(lambda x, y : x + y, trial_C_list))
        trial_C_list.append((colC[:,0] + colC[:,-1]).reshape((dim,dim)))
        
        for tmp in trial_C_list:
            CCdagger = np.dot(tmp, np.conjugate(tmp).T)
            
            if CCdagger[0,0].real != 0 and np.linalg.norm(CCdagger/CCdagger[0,0].real - np.eye(dim)) < tol:
                print("No need to start unitaryizing procedure!")
                print("End")
                print("==================================================")
                
                return tmp / np.sqrt(CCdagger[0,0].real)
            
        trial_C_list.clear()
        
        for ii in range(np.shape(colC)[1]):
            for jj in range(ii,np.shape(colC)[1]):
                trial_C_list.append((colC[:,ii] + colC[:,jj]).reshape((dim,dim)))
        
        for tmp in trial_C_list:
            CCdagger = np.dot(tmp, np.conjugate(tmp).T)
            
            if CCdagger[0,0].real != 0 and np.linalg.norm(CCdagger/CCdagger[0,0].real - np.eye(dim)) < tol:
                print("No need to start unitaryizing procedure!")
                print("End")
                print("==================================================")
                
                return tmp / np.sqrt(CCdagger[0,0].real)
        
        trial_C_list.clear()
        
        for ii in range(np.shape(colC)[1]):
            for jj in range(ii,np.shape(colC)[1]):
                for kk in range(jj,np.shape(colC)[1]):
                    trial_C_list.append((colC[:,ii] + colC[:,jj] + colC[:,kk]).reshape((dim,dim)))
                    
        for tmp in trial_C_list:
            CCdagger = np.dot(tmp, np.conjugate(tmp).T)
            
            if CCdagger[0,0].real != 0 and np.linalg.norm(CCdagger/CCdagger[0,0].real - np.eye(dim)) < tol:
                print("No need to start unitaryizing procedure!")
                print("End")
                print("==================================================")
                
                return tmp / np.sqrt(CCdagger[0,0].real)
        
        trial_C_list.clear()
        
        for ii in range(np.shape(colC)[1]):
            for jj in range(ii,np.shape(colC)[1]):
                for kk in range(jj,np.shape(colC)[1]):
                    for ll in range(kk,np.shape(colC)[1]):
                        trial_C_list.append((colC[:,ii] + colC[:,jj] + colC[:,kk] + colC[:,ll]).reshape((dim,dim)))
        
        for tmp in trial_C_list:
            CCdagger = np.dot(tmp, np.conjugate(tmp).T)
            
            if CCdagger[0,0].real != 0 and np.linalg.norm(CCdagger/CCdagger[0,0].real - np.eye(dim)) < tol:
                print("No need to start unitaryizing procedure!")
                print("End")
                print("==================================================")
                
                return tmp / np.sqrt(CCdagger[0,0].real)
            
        
        trial_C_list.clear()
        
        for ii in range(np.shape(colC)[1]):
            for jj in range(ii,np.shape(colC)[1]):
                for kk in range(jj,np.shape(colC)[1]):
                    for ll in range(kk,np.shape(colC)[1]):
                        for mm in range(ll,np.shape(colC)[1]):
                            trial_C_list.append((colC[:,ii] + colC[:,jj] + colC[:,kk] + colC[:,ll] + colC[:,mm]).reshape((dim,dim)))
        
                        
        for tmp in trial_C_list:
            CCdagger = np.dot(tmp, np.conjugate(tmp).T)
            
            if CCdagger[0,0].real != 0 and np.linalg.norm(CCdagger/CCdagger[0,0].real - np.eye(dim)) < tol:
                print("No need to start unitaryizing procedure!")
                print("End")
                print("==================================================")
                
                return tmp / np.sqrt(CCdagger[0,0].real)
        
        # unitaryizing by optimal procedure
        num = colC.shape[-1]
        #coe = sp.symbols('b0:{}'.format(num))
        #total_mat = 0
        
        
        import random
        print("Start unitaryzing procedure")
        print("==================================================")
        print("Errors in the optimization process (unitaryizing the similarity transformation matrix)")
        
        basis = []
        
        for i in range(num):
            basis.append(colC[:,i].reshape((dim,dim)))
        
        for tmp in range(30):

            coe_num = []
            random.seed(tmp)
            
            # initial value
            for i in range(num):
                #total_mat += coe[i]*colC[:,i].reshape((dim,dim))
                if tmp == 0:
                    coe_num.append(i/10) #initial optimal coefficients
                    
                elif tmp == 1:
                    coe_num.append(i)
                    
                else: #random value
                    coe_num.append((-1)**random.randint(0,1)*random.random()*(i+2))
                    
            '''
            import random
            random.shuffle(coe_num)
            '''
            
            '''
            def loss(coe_num):
                tmp_matrix = sp.Matrix(total_mat)
                for i in range(len(coe)):
                    tmp_matrix = tmp_matrix.subs(coe[i],coe_num[i])
                #print(abs((tmp_matrix.H@tmp_matrix-sp.eye(tmp_matrix.shape[0]).evalf())))
                min_val = max(abs((tmp_matrix.H@tmp_matrix-sp.eye(tmp_matrix.shape[0])).evalf()))
                print(min_val)
                return min_val
            '''
            
            
            
            opt_num = 1 # Number of optimization steps, nonlocal
            
            #@numba.njit
            def loss(coe_num):
                # define the loss function (error)
                tmp_matrix = 0
                
                for i in range(len(coe_num)):
                    tmp_matrix += coe_num[i]*basis[i]
                    
                #print(abs((tmp_matrix.H@tmp_matrix-sp.eye(tmp_matrix.shape[0]).evalf())))
                min_mat = tmp_matrix.T.conj()@tmp_matrix
                min_val = sum(abs((min_mat-np.eye(tmp_matrix.shape[0]))))
                #min_val = sum(abs((min_mat-np.eye(tmp_matrix.shape[0])))**2)
                
                nonlocal opt_num
                
                print("steps_num:",opt_num,';',"error:",sum(min_val))
                
                opt_num += 1
                
                return sum(min_val)
            
            
            #error = []
            #tol = 1e-5
            #construct the tolerence limit
            constraints=({'type':'ineq','fun':lambda x:tol - loss(x)})
            
            #find the unitary
            result = minimize(loss, coe_num, method='SLSQP',constraints=constraints)
            #result = minimize(loss, coe_num, method='TNC',constraints=constraints)
            #result = minimize(loss, coe_num, method='L-BFGS-B',constraints=constraints)
            
            #print(result.x)
            #from scipy.optimize import fmin_cg
            #print(fmin_cg(loss,coe_num))
            
            
            U = 0
            
            if result.success:
                
                for i in range(num):
                    U+=result.x[i]*basis[i]
                    
                #print(result.x)
                print("End")
                print("==================================================")
                
                return U
            
        else:
            #print(result.x)
            raise ValueError('Fail to find unitary U!')
        

    
    #print(ns.shape)
    _,cn = ns.shape
    
    ### vec in ns: {real,imag} ,dim = 2*row
    ns_new = np.zeros((row*row,cn),dtype=complex)
    
    for index in range(cn):
        ns_new[:,index] = ns[:row*row,index] + 1j*ns[row*row:,index]
    
    
    ns_mat = transform_colC_to_matC(ns_new, row,tol = tol)
    
    
    
    return ns_mat,cn,row
###############################################################################



###############################################################################
def get_transform_matrix(operator,Symmetry,band_num,tol = 1e-9):
    """
    Get the similarity transformation matrix between DFT representation and the standard representation

    Parameters
    ----------
    operator : dictionary
        VASP corepresentation
    Symmetry : dictionary
        Users' input. The standard corepresentation and rotation matrix.
    band_num : integer
        The total number of the bands which we are interested in.
    tol : float, optional
        Tolerance error. The default is 1e-9.

    Returns
    -------
    U : np.matrix
        the similarity transformation matrix between DFT representation and the standard representation

    """
    
    DFT_rep_list = []
    std_rep_list = []
    
    
    # construct the input matrix of finding the unitary similarity transformation matrix
    flag = 0
    
    for i in operator.keys(): 
        # if there are at least one anti-unitary symmetry element
        if Symmetry[i]['repr_has_cc'] is True:
            
            if flag == 0:
                flag = 1
                Tmat = np.zeros((2,band_num,band_num),dtype=complex)
                Tmat[0,:,:]=np.array(operator[i],dtype=complex)
                Tmat[1,:,:]=np.array(Symmetry[i]['repr_matrix'],dtype=complex)
            
            # if the number of anti-unitary symmetry elements is larger than 1
            else:
                tmp = Tmat[0,:,:]@np.array(operator[i],dtype=complex).conj()
                tmp_ = Tmat[1,:,:]@np.array(Symmetry[i]['repr_matrix'],dtype=complex).conj()
                
                DFT_rep_list.append(tmp)
                std_rep_list.append(tmp_)
        
        else:
            DFT_rep_list.append(np.array(operator[i],dtype=complex))
            std_rep_list.append(np.array(Symmetry[i]['repr_matrix'].evalf(),dtype=complex))
    
    
    #list to array
    DFT_rep = np.array(DFT_rep_list)
    std_rep = np.array(std_rep_list)
    #print(Symmetry)
    #print(operator)
    #print(str(DFT_rep))
    #print(str(std_rep))
    
    #np.set_printoptions(precision=5)
    
    if flag == 1:
        ns_mat,cn_tot,row = cal_nullspace_complex(DFT_rep,std_rep,is_TRS = True, Tmat = Tmat,tol=tol)
    
    else:
        ns_mat,cn_tot,row = cal_nullspace_complex(DFT_rep,std_rep,is_TRS = False,tol=tol)
    
    # transform to np.matrix
    U = np.matrix(ns_mat)
    
    #print(U)
    return U
###############################################################################



###############################################################################
if __name__ == "__main__":
#     from sympy import Matrix,sqrt,I
#     from numpy import matrix
    
#     a=np.array([[[ 1.00000e+00-9.71533e-13j]],[[-5.02249e-14+1.00000e+00j]]])
#     b=np.array([[[1.+0.j]],[[0.+1.j]]])
#     cal_nullspace_complex(a,b,is_TRS = False,tol=1e-4)
#     a_={'C3z': {'rotation_matrix': Matrix([[     -1/2, -sqrt(3)/2, 0],
# [sqrt(3)/2,       -1/2, 0],
# [        0,          0, 1]]), 'repr_has_cc': False, 'band_repr_matrix_list': [1, 1, Matrix([
# [-1/2 - sqrt(3)*I/2,                  0],
# [                 0, -1/2 + sqrt(3)*I/2]])], 'repr_matrix': Matrix([[1]])}, 'C2x': {'rotation_matrix': Matrix([
# [1,  0,  0],
# [0, -1,  0],
# [0,  0, -1]]), 'repr_has_cc': False, 'band_repr_matrix_list': [I, -I, Matrix([
# [                0, -1/2 + sqrt(3)*I/2],
# [1/2 + sqrt(3)*I/2,                  0]])], 'repr_matrix': Matrix([[I]])}}
#     b_={'C3z': matrix([[1.-9.71533e-13j]]), 'C2x': matrix([[-5.02249e-14+1.j]])}
#     #get_transform_matrix(b_,a_,1,tol = 1e-9)

#     Symmetry = a_
#     operator = b_
    
#     DFT_rep_list = []
#     std_rep_list = []
#     band_num = 1
    
#     # construct the input matrix of finding the unitary similarity transformation matrix
#     flag = 0
    
#     for i in operator.keys(): 
#         # if there are at least one anti-unitary symmetry element
#         if Symmetry[i]['repr_has_cc'] is True:
            
#             if flag == 0:
#                 flag = 1
#                 Tmat = np.zeros((2,band_num,band_num),dtype=complex)
#                 Tmat[0,:,:]=np.array(operator[i],dtype=complex)
#                 Tmat[1,:,:]=np.array(Symmetry[i]['repr_matrix'],dtype=complex)
            
#             # if the number of anti-unitary symmetry elements is larger than 1
#             else:
#                 tmp = Tmat[0,:,:]@np.array(operator[i],dtype=complex).conj()
#                 tmp_ = Tmat[1,:,:]@np.array(Symmetry[i]['repr_matrix'],dtype=complex).conj()
                
#                 DFT_rep_list.append(tmp)
#                 std_rep_list.append(tmp_)
        
#         else:
#             DFT_rep_list.append(np.array(operator[i],dtype=complex))
#             std_rep_list.append(np.array(Symmetry[i]['repr_matrix'].evalf(),dtype=complex))
    
    
#     #list to array
#     DFT_rep = np.array(DFT_rep_list)
#     std_rep = np.array(std_rep_list)
#     cal_nullspace_complex(DFT_rep,std_rep,is_TRS = False,tol=1e-4)
    from sympy import I
    import numpy as np
    A_ = sp.Matrix([[0.134622219682496 + 0.462662316717115*I, 0.804228737406376 - 0.189053337262626*I, 0.0411058807414449 + 0.141270316371809*I, 0.245564998788803 - 0.0577259100831024*I, -0.000109756611384457 - 0.000377136116812905*I, -0.000655574082269871 + 0.000154077882395687*I], [-0.234694372943257 - 0.792113363773208*I, 0.468518954929098 - 0.112559089422752*I, -0.0716621532177885 - 0.241865627497448*I, 0.143058619805976 - 0.0343690067626257*I, 0.000191340414721596 + 0.000645679993961474*I, -0.000381914611975163 + 9.17338647550633e-5*I], [0.0697156098467269 + 0.0856720140720665*I, 0.20812472176877 - 0.17252417488479*I, -0.228319846562219 - 0.280577208762036*I, -0.681612046354284 + 0.565019387218109*I, 0.00115283147614435 + 0.00141667307735361*I, 0.00344157739494483 - 0.00285285149449254*I], [0.270272867882218 - 0.00575010160406182*I, 0.00335134757311627 + 0.110402580278665*I, -0.885148153530608 + 0.0188318855435968*I, -0.0109756247128709 - 0.36157026574541*I, 0.00446924841579719 - 9.51077999976953e-5*I, 5.54077450468019e-5 + 0.00182562378697647*I], [0.000431257279698483 - 0.00044380004442247*I, 2.4184099657192e-5 - 0.000241175742568914*I, -0.00314452170338582 + 0.00323621342987301*I, -0.000176408185942718 + 0.00175861591194703*I, -0.648872862978298 + 0.667784199493418*I, -0.0363993727868154 + 0.362888164801286*I], [-0.000240950795713958 + 2.63641436286559e-5*I, 0.000439877546673518 - 0.000435260802378173*I, 0.00175694507079155 - 0.000192323794175527*I, -0.00320761238963811 + 0.0031736877109519*I, 0.362543769999472 - 0.0396831114080094*I, -0.66188258150685 + 0.654891729774227*I]])
    B_ = sp.Matrix([[0.0184799094895779 + 0.0635108716996821*I, 0.11039854623802 - 0.0259518905019884*I, -0.139540630741213 - 0.479565587237062*I, -0.833611086033044 + 0.195960341949238*I, 0.000109756188447037 + 0.000377134663552132*I, 0.000655571556072671 - 0.000154077288670796*I], [-0.0322170551178404 - 0.108735470550691*I, 0.0643148052451271 - 0.0154513092111807*I, 0.243268909017657 + 0.82105306449561*I, -0.48563620679156 + 0.116671399649046*I, -0.000191339677407855 - 0.000645677505890504*I, 0.000381913140300675 - 9.17335112666598e-5*I], [0.236660104694516 + 0.290826385464176*I, 0.706510549897474 - 0.585658927470618*I, 0.0313409976074402 + 0.0385142192556592*I, 0.0935634691488433 - 0.0775589662031926*I, -0.00115282703380946 - 0.00141666761832822*I, -0.00344156413312825 + 0.00285284050127963*I], [0.917481614693575 - 0.0195197668170816*I, 0.0113765638531551 + 0.374778022182751*I, 0.121502408406676 - 0.00258507041259968*I, 0.00150657445010324 + 0.0496319753650043*I, -0.0044692311939412 + 9.51074335081892e-5*I, -5.54075315377762e-5 - 0.00182561675209553*I], [0.00303985816686007 - 0.00312848456075141*I, 0.000170532599120045 - 0.00170007699672804*I, 0.000912830987796197 - 0.000939477838925687*I, 5.12186163646281e-5 - 0.000510522674638669*I, 0.648870362603697 - 0.66778162624564*I, 0.0363992325250286 - 0.362886766443607*I], [-0.00169846351250171 + 0.000185917340332416*I, 0.00310083549277768 - 0.00306805486780481*I, -0.000510033819384811 + 5.58412645719413e-5*I, 0.000931175164211447 - 0.00092129444563366*I, -0.362542372968889 + 0.0396829584926238*I, 0.661880031000439 - 0.65488920620645*I]])
    A = np.zeros((1,6,6),dtype=np.complex128)
    B = np.zeros((1,6,6),dtype=np.complex128)
    A[0,:,:] = A_
    B[0,:,:] = B_
    print(str(cal_nullspace_complex(A,B,is_TRS = False, Tmat = None,tol = 1e-4)))
