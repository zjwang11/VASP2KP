# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:26:43 2023
Last modified on Wedn Dec 6 14:45:00 2023

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Main function of VASP2KP
"""


from ._numeric_kp import get_numeric_kp,get_U_pi_sigma,SymmetryERROR,Symmetry_conjugate_flag
from ._standard_kp import get_std_kp,get_Zeeman
from ._read_data import load_data


import sys
import os
import subprocess
import kdotp_generator as kp
import sympy as sp
from sympy.core.numbers import I
import numpy as np
import warnings

###############################################################################
# define ERROR class(type)
class ModelTooHighOrderERROR(Exception):
    '''
    If order >=4: raise this ERROR
    '''
    pass

class PrintFlagERROR(Exception):
    '''
    If print_flag not in [0,1,2]: raise this ERROR
    '''
    pass

class GFactorERROR(Exception):
    '''
    If gfactor not in [0,1]: raise this ERROR
    '''
    pass

class BandChooseERROR(Exception):
    pass

class OrderChooseERROR(Exception):
    '''
    If order not in [1,2,3]: raise this ERROR
    '''
    pass

class LogChooseERROR(Exception):
    '''
    If log not in [0,1]: raise this ERROR
    '''
    pass

class AccSetERROR(Exception):
    '''
    If acc not in [0,1]: raise this ERROR
    '''
    pass

class numbaNotInstallERROR(Exception):
    '''
    If 'import numba' fails: raise this ERROR
    '''
    pass
###############################################################################



###############################################################################
def get_std_kp_Zeeman_given_data(data,operator,eigen_energy,Symmetry,band_interest_set,repr_split = True,dim_list=[],order = 2,gfactor = 1,print_flag = 2,log = 0, acc = 0, msg_num = None,kvec = None):
    '''
    After reading the VASP calculation data, get the kp Hamiltonian matrix as well as the Zeeman's coupling as well as the numerical parameters

    Parameters
    ----------
    data : dictionary
        the input DFT data, generalized momentum matrices as well as spin matrices
        if gfactor=1,keys are Pix,Piy,Piz,sigx,sigy,sigz;
        if gfactor=0,keys are Pix,Piy,Piz
    operator : dictionary 
        generators' corepresentation matrices obtained by DFT(vasp_song).
    eigen_energy : list/numpy.array
        the eigenvalues of each band got by DFT(VASP)
    Symmetry : dictionary
        a dictionary according to the user's input, the rotation matrices, the standard corepresentation matrices of all generators
    band_interest_set : list
        id list of the bands of interest
    order : integer, optional
        the oreder of kp Hamiltonian to be constructed. The default is 2.
    gfactor : integer, optional
        Whether to solve the Zeeman‘s coupling. The default is 0.
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

    Returns
    -------
    result_kp_array : sympy.matrices.dense.MutableDenseMatrix
        The sum of all independent kp models(the coefficients are also calculated).
    kpmodel_coe : dictionary
        the keys are coefficients' names, the values are lists
                    the zeroth element of each list is the value of the coefficient
                    the first element of each list is a matrix of each independent kp model
                    result_kp_array is the sum of the zeroth element* the first element of each list
    
    if gfactor is set to be 1, there are two more return values:
    result_Zeeman_array : sympy.matrices.dense.MutableDenseMatrix
        The sum of all independent Zeeman's coupling(the coefficients are also calculated and substituted in).
    Zeeman_coe : dictionary
        the keys are coefficients' names, the values are lists
                    the zeroth element of each list is the value of the coefficient
                    the first element of each list is a matrix of each independent Zeeman's coupling
                    result_Zeeman_array is the sum of the zeroth element* the first element of each list
        
    '''
    
    if dim_list == []:
        dim_list = list(operator.values())[0].shape[0]

    # get the list of pi (3 components), the list of sigma (3 components), similarity transformation matrix between two corepresentation
    pi_list,sigma_list,U,Symmetry = get_U_pi_sigma(data,operator,Symmetry,repr_split ,dim_list,gfactor,log)
    print("Find the unitary transformation U successfully (U^-1 D^num U= D^std) !")
    
    # calculate the numerical kp hamiltonian and Zeeman's coupling
    numeric_kp = get_numeric_kp(pi_list,sigma_list,U,eigen_energy,band_interest_set,order,gfactor,acc)
    print("Finish downfolding processes!")
    
    # calculate the standard kp Hamiltonian
    if order>=1:
        result_kp_array,kpmodel_coe = get_std_kp(Symmetry,order,gfactor,msg_num,kvec,numeric_kp,print_flag,log)
        if print_flag == 2:
            print('''Finish constructing kp invariant Hamiltonian in "kp-parameters.out"!''')
        else:
            print('''Finish constructing kp invariant Hamiltonian!''')
    else:
        result_kp_array = 0
        kpmodel_coe={}
    
    if gfactor == 1: # calculate the Zeeman's coupling
    
        G_4c4 = numeric_kp["B"]
        result_Zeeman_array,Zeeman_coe = get_Zeeman(Symmetry,msg_num,kvec,G_4c4,order,print_flag,log)
        if print_flag == 2:
            print('''Finish constructing Zeeman's coupling in "g-factors.out"!''')
        else:
            print('''Finish constructing Zeeman's coupling!''')
        return result_kp_array,kpmodel_coe,result_Zeeman_array,Zeeman_coe
    
    return result_kp_array,kpmodel_coe
###############################################################################



###############################################################################
def get_std_kp_Zeeman(Symmetry,vaspMAT='mat',kpmodel = 1, repr_split = True,order = 2,gfactor = 1,print_flag = 2,log = 0,acc = 0,msg_num = None,kvec = None):
    '''
    The main function 
    Obtain the kp Hamiltonian matrix as well as the Zeeman's coupling as well as the numerical parameters
    
    Notice: you can treat the parameters msg_num and kvec as not existing
    
    Parameters
    ----------
    Symmetry : Dictonary
        the user's input, a dictionary comprised of the optimal/standard representations matrices of each generator
        keys: generators' names, it must be consistent with the X in the .m file name "MAT_X.m" generated by vasp_song.
        values: also a dictionary. 
        The dictionary of values:
            key-value: 
                'rotation_matrix': 3d rotation matrix
                'repr_matrix': optimal/standard representation matrix
                'repr_has_cc': if the element is an anti-unitary element, True/False
    vaspMAT : string, optional
        user's input, the path of the file folder which contains the input files
        The default is 'mat'.
    kpmodel : integer, optional(one of [0,1])
        user's input. The default is 1.
        If kpmodel = 1, the kp Hamiltonian will be generated and calculated.
        If kpmodel = 0, the kp Hamiltonian will not be generated.
    order : integer, optional(one of [2,3])
        user's input. The default is 2. The order of kp Hamiltonian to be calculated.
        If order=0, the kp Hamiltonian will not be generated.
    gfactor : integer, optional(one of [1,2])
        user's input. The default is 1. 
        If gfactor=1, the Zeeman coupling will be calculated and output.
        If gfactor=0, the Zeeman coupling won't be calculated.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional(one of [0,1])
        user's input. The default is 0. 
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the independent of each order as well as some warning.
    acc : integer, optional(one of [0,1])
        user's input. The default is 0.
        If acc == 0, the code will not be accalerated.
        If acc == 1, the code will be accelerated by numba.njit.
    msg_num : decimal or integer, optional
        user's input, magnetic space group number. The default is None. 
    kvec : list, optional
        user's input, the high symmetry point coordinate. The default is None.

    Raises
    ------
    ModelTooHighOrderERROR
        if order>4, raise this ERROR: the package can just deal with at most third-order Hamiltonians.
    OrderChooseERROR
        if order<0 or order is not a integer, raise this ERROR.
    PrintFlagERROR
        if print_flag not in [0,1,2], raise this ERROR: the wrong parameter print_flag.
    GFactorERROR
        if gfactor not in [0,1], raise this ERROR: the wrong parameter gfactor.
    LogChooseERROR
        if log not in [0,1], raise this ERROR: the wrong parameter log
    AccSetERROR
        if acc not in [0,1], raise this ERROR: the wrong parameter acc
    numbaNotInstallERROR
        if acc==1 and numba has not been installed, raise this ERROR

    Returns
    -------
    result_kp_array : sympy.matrices.dense.MutableDenseMatrix
        the effective kp model(the sum of the effective kp models of each order as well as their parameters)
    result_kpmodel_g : sympy.matrices.dense.MutableDenseMatrix
        the effective Zeeman's coupling(the parameters have been found and substituted in.)
        
    Other significants variables
    -------
    kpmodel_coe : a dictionary
        keys: coefficient name, TYPE: sympy.core.symbol.Symbol
            You can use list(kpmodel_coe.keys()) to get the list of keys and then get the variable name.
        values: a list, 
            The zeroth element is the value of the parameter.
            The first element is the matrix of the independent kp hamiltonian. 
    Zeeman_coe : a dictionary
        keys: coefficient name, TYPE: sympy.core.symbol.Symbol
        values: a list,
            The zeroth element is the value of the parameter.
            The first element is the matrix of the independent Zeeman coupling. 
            
    '''

    # simple input ERROR detection
    if order>=4:
        #raise ModelTooHighOrderERROR("The order of the kp model is too high"+"("+str(order)+")"+", please adjust the parameter order to <=3!")
        print("ERROR: The order of the kp model is too high"+"("+str(order)+")"+", please adjust the parameter order to <=3!")
        sys.exit()

    if order not in [1,2,3]:
        #raise OrderChooseERROR("The parameter order cannot be negative or decimal!")
        print("ERROR: The parameter order cannot be negative or decimal!")
        sys.exit()
        
    if print_flag not in [0,1,2]:
        #raise PrintFlagERROR("The parameter print_flag should be 0,1 or 2!")
        print("TERROR: The parameter print_flag should be 0,1 or 2!")
        sys.exit()
    
    if gfactor not in [0,1]:
        #raise GFactorERROR("The parameter gfactor should be 0,1!")
        print("ERROR: The parameter gfactor should be 0,1!")
        sys.exit()
    
    if log not in [0,1]:
        #raise LogChooseERROR("The parameter log should be 0,1!")
        print("ERROR: The parameter log should be 0,1!")
        sys.exit()
        
    if acc not in [0,1]:
        #raise AccSetERROR("The parameter acc should be 0,1!")
        print("ERROR: The parameter acc should be 0,1!")
        sys.exit()
        
    if acc == 1:
        
        # test
        try:
            import numba
            
        except:
            #raise numbaNotInstallERROR("The package numba is not installed! If you do not want to accelerate the code, set parameter acc to 0.")
            print("ERROR: The package numba is not installed! If you do not want to accelerate the code, set parameter acc to 0.")
            sys.exit()
    
    for i in Symmetry.keys():
        i_val = Symmetry[i]
        
        if 'rotation_matrix' not in i_val.keys():
            wrong_str = "The key 'rotation_matrix' of '"+i+"' not exist!"
            
            #raise SymmetryERROR(wrong_str)
            print("ERROR: "+wrong_str)
            sys.exit()
            
        if 'repr_has_cc' not in i_val.keys():
            wrong_str = "The key 'repr_has_cc' of '"+i+"' not exist!"
            
            #raise SymmetryERROR(wrong_str)
            print('ERROR: '+wrong_str)
            sys.exit()
    
    if kpmodel == 0:
        order = 0        
    
    # load data from .m file
    operator,data,eigen_energy,band_interest_set,dim_list = load_data(Symmetry,vaspMAT,gfactor,repr_split)
    
    # unified dimensions
    size = 0
    
    for i in data.keys():
        size = max(size,data[i].shape[0])
        
    eigen_energy = eigen_energy[:size]
    
    #print(Symmetry)
    
    if Symmetry_conjugate_flag:
        
        Symmetry_copy = Symmetry.copy()
        
        for i in Symmetry.keys():
            #print(Symmetry[i].keys())
            if 'repr_matrix' in Symmetry[i].keys():

                Symmetry_copy[i]['repr_matrix'] = sp.conjugate(Symmetry[i]['repr_matrix'])
                
            if 'band_repr_matrix_list' in Symmetry[i].keys():
                Symmetry_copy[i]['band_repr_matrix_list'] = [sp.conjugate(j) for j in Symmetry[i]['band_repr_matrix_list']]
        
        Symmetry = Symmetry_copy
        
    #print(Symmetry_conjugate_flag)
    
    #print(Symmetry)
    
    if gfactor == 1: # calculate the Zeeman coupling
        result_kp_array,kpmodel_coe,result_kpmodel_g,Zeeman_coe=get_std_kp_Zeeman_given_data(data,operator,eigen_energy,Symmetry,band_interest_set,repr_split ,dim_list,order,gfactor,print_flag,log,acc,msg_num,kvec)
        #return result_kp_array,kpmodel_coe,result_kpmodel_g,Zeeman_coe
        warnings.filterwarnings("ignore")
        return result_kp_array,result_kpmodel_g
    
    else:
        result_kp_array,kpmodel_coe = get_std_kp_Zeeman_given_data(data,operator,eigen_energy,Symmetry,band_interest_set,repr_split ,dim_list,order,gfactor,print_flag,log,acc,msg_num,kvec)
        #return result_kp_array,kpmodel_coe,None,None
        warnings.filterwarnings("ignore")
        return result_kp_array,None
###############################################################################



###############################################################################
def get_std_kp_Zeeman_no_coe(Symmetry,order=2,gfactor=1,print_flag=2,log=0,msg_num=None,kvec=None):
    '''
    The auxiliary function 
    Obtain the kp Hamiltonian matrix as well as the Zeeman's coupling without numerical parameters
    
    Notice: you can treat the parameters msg_num and kvec as not existing
    
    Parameters
    ----------
    Symmetry : Dictonary
        the user's input, a dictionary comprised of the optimal/standard representations matrices of each generator
        keys: generators' names, it must be consistent with the X in the .m file name "MAT_X.m" generated by vasp_song.
        values: also a dictionary. 
        The dictionary of values:
            key-value: 
                'rotation_matrix': 3d rotation matrix
                'repr_matrix': optimal/standard representation matrix
                'repr_has_cc': if the element is an anti-unitary element, True/False
    order : integer, optional
        user's input. The default is 2. The order of kp Hamiltonian to be calculated.
    gfactor : integer, optional(one of [1,2])
        user's input. The default is 1. 
        If gfactor=1, the Zeeman coupling will be calculated and output.
        If gfactor=0, the Zeeman coupling won't be calculated.
    print_flag : integer, optional(one of [0,1,2])
        user's input. The default is 2. 
        If print_flag = 2, the result will be output in the file 'XXX.out'.
        If print_flag = 1, the result will be output on the screen(or command line).
        If print_flag = 0, the result won't be output.
    log : integer, optional(one of [0,1])
        user's input. The default is 0. 
        If log == 0, there will be no .log file.
        If log == 1, there will be log files containing the independent of each order as well as some warning.
    msg_num : decimal or integer, optional
        user's input, magnetic space group number. The default is None. 
    kvec : list, optional
        user's input, the high symmetry point coordinate. The default is None.

    Raises
    ------
    OrderChooseERROR
        if order<0 or order is not a integer, raise this ERROR.
    PrintFlagERROR
        if print_flag not in [0,1,2], raise this ERROR: the wrong parameter print_flag.
    GchooseERROR
        if gfactor not in [0,1], raise this ERROR: the wrong parameter gchoose.
    LogChooseERROR
        if log not in [0,1], raise this ERROR: the wrong parameter log

    Returns
    -------
    sum_kp : sympy.matrices.dense.MutableDenseMatrix
        the effective kp model(without getting the values of the parameters.)
    sum_Zeeman : sympy.matrices.dense.MutableDenseMatrix
        the effective Zeeman coupling(without getting the values of the parameters.)

    '''
    
    if int(order) != order and order > 0:
        raise OrderChooseERROR("The parameter order cannot be negative or decimal!")
        sys.exit()
        
    if print_flag not in [0,1,2]:
        raise PrintFlagERROR("The parameter print_flag should be 0,1 or 2!")
        sys.exit()
    
    if gfactor not in [0,1]:
        raise GFactorERROR("The parameter gchoose should be 0,1!")
        sys.exit()
    
    if log not in [0,1]:
        raise LogChooseERROR("The parameter log should be 0,1!")
        sys.exit()
    
    if log != 0:
        logging = open("independent_kp_models(without numerical parameters).log","w")
        logging.write('\nPrint kp results:\n')
        
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
            
    
        
    save_stdout = sys.stdout
    sys.stdout = logging
        
    sym_ops = list(Symmetry.values())
    
    sum_kp = sp.zeros(sym_ops[0]['repr_matrix'].shape[0])

    order_list = [ [i] for i in range(order+1) ]
    
    sym_list_0 = []
    sym_list_all = []
    sym_list_all_ = []
    
    kx_,ky_,kz_= sp.symbols("k_x k_y k_z",commutative = False)
    kx,ky,kz = sp.symbols("kx ky kz")
    #variable_list = [kx,ky,kz]
    #variable_list_ = [kx_,ky_,kz_]
    
    for i in range(order+1):
        sym_list_0.append(chr(i+97))
        
    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        np.complex
        warnings.resetwarnings()
        
    except:
        warnings.resetwarnings()
        np.complex = np.complex128
        
    for order in order_list:
        
        kpmodel, irrep_expr_basis, irrep_repr_basis = kp.symmetric_hamiltonian(
            sym_ops,
            kp_variable = 'k',
            order = order,
            repr_basis = 'pauli',
            msg_num = msg_num,
            kvec = kvec
        )
        
        sym_list = sp.symbols((sym_list_0[order[0]]+'1:{}').format(len(kpmodel)+1))
        
        for i in range(len(kpmodel)):
            sum_kp += kpmodel[i]*sym_list[i]
            
        sym_list_all += sym_list
        
        sym_list_ = sp.symbols((sym_list_0[order[0]]+'_1:{}').format(len(kpmodel)+1))
        
        sym_list_all_ += sym_list_
        
    logging.close()
    
    sys.stdout = save_stdout
    
    if print_flag == 0:
        pass
    
    elif print_flag == 1:
        print('\n==========  Result of kp model  ==========')
        print(sum_kp)
        
    elif print_flag == 2:
        outfile = open("kp_model_without_parameters.out","w")
        outfile.write('==========  Result of kp model  ==========\n')
        outfile.write(str(sum_kp))
        outfile.close()
        

        if gfactor == 0:
            texfile_name = "kp_Hamiltonian(without_numerical_parameters)"
            
        else:
            texfile_name = "kp_Hamiltonian-Zeeman's_coupling(without_numerical_parameters)"
            
        texfile = open(texfile_name+".tex","w")
        
        texfile.write('\\documentclass[aps,amssymb,onecolumn]{revtex4}\n')
        
        texfile.write('\\usepackage{breqn}\n')
        
        texfile.write('\\usepackage{hyperref}\n\n')
        
        texfile.write('\n\\begin{document}\n\n')
        
        texfile.write('\\section{\\texorpdfstring{$ k\\cdot p $}{} Hamiltonian}\n\n')
        
        texfile.write("\\begin{dgroup*}\n")
        
        for i in range(sum_kp.shape[0]):
            
            for j in range(i,sum_kp.shape[0]):
                
                # add subscript
                ele = sum_kp[i,j].subs(kx,kx_).subs(ky,ky_).subs(kz,kz_)
                
                for k in range(len(sym_list_all)):
                    ele = ele.subs(sym_list_all[k],sym_list_all_[k])
                    
                ele_string = sp.latex(ele)
                
                ele_string = ele_string.replace("\\cdot","")
                
                texfile.write("\\begin{dmath}\n")
                #texfile.write("\\noindent ")
                #texfile.write("$ H^{kp}_{"+str(i+1)+str(j+1)+"} = "+ele_string+" $;\n\n~\\\\\n\n")
                
                texfile.write("\tH^{kp}_{"+str(i+1)+str(j+1)+"} = "+ele_string+"\n")
                texfile.write("\\end{dmath}\n\n")
        
        texfile.write("\\end{dgroup*}\n")
        
        texfile.write("\n")
        
        
        # if Zeeman's coupling is not intended to be calculated, end the tex file
        if gfactor == 0:
            
            # end the .tex file
            texfile.write(r'\end{document}')
            
            texfile.close()
            
            # if the system has the command 'pdflatex': compile the .tex
            try:
                import platform
                
                System = platform.system()
                
                if System == 'Windows':
                    pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian(without_numerical_parameters).tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                else:
                    pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian\(without_numerical_parameters\).tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
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
        
    #calculate Zeeman's coupling
    if gfactor == 1:
        
        if log != 0:
            logging = open("independent_Zeeman's_coupling(without numerical parameters).log","w")
            logging.write('\nPrint Zeeman\'s coupling results:\n')
            
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
                
        save_stdout = sys.stdout
        sys.stdout = logging
        sum_Zeeman = sp.zeros(sym_ops[0]['repr_matrix'].shape[0])
        
        # calculate kp hamiltonian
        kpmodel, irrep_expr_basis, irrep_repr_basis = kp.symmetric_hamiltonian(
            sym_ops,
            kp_variable = 'B',
            order = [1],
            repr_basis = 'pauli',
            msg_num = msg_num,
            kvec = kvec
        )
        
        sym_list = sp.symbols(('g1:{}').format(len(kpmodel)+1))
        sym_list_ = sp.symbols(('g_1:{}').format(len(kpmodel)+1))
        
        for i in range(len(kpmodel)):
            sum_Zeeman += kpmodel[i]*sym_list[i]
            
        logging.close()
        sys.stdout = save_stdout
        
        if print_flag == 0:
            pass
        
        elif print_flag == 1:
            print('\n==========  Result of Zeeman\'s coupling  ==========')
            print(sum_Zeeman)
            
        elif print_flag == 2:
            outfile = open("Zeeman's_coupling_without_parameters.out","w")
            outfile.write('==========  Result of Zeeman\'s coupling  ==========\n')
            outfile.write(str(sum_Zeeman))
            outfile.close()
            
            Bx_,By_,Bz_= sp.symbols("B_x B_y B_z")
            Bx,By,Bz = sp.symbols("Bx By Bz")
            
            # output the result as the latex source file
            texfile.write("\\section{Zeeman's coupling}\n\n")
            
            texfile.write("\\begin{dgroup*}\n")
            
            for i in range(sum_Zeeman.shape[0]):
                
                for j in range(i,sum_Zeeman.shape[0]):
                    
                    # add subscript
                    ele = sum_Zeeman[i,j].subs(Bx,Bx_).subs(By,By_).subs(Bz,Bz_)
                    
                    for k in range(len(sym_list)):
                        ele = ele.subs(sym_list[k],sym_list_[k])
                        
                    ele_string = sp.latex(ele)
                    ele_string = ele_string.replace("\\cdot","")
                    
                    texfile.write("\\begin{dmath}\n")
                    #texfile.write("\\noindent ")
                    #texfile.write("$ H^{Z}_{"+str(i+1)+str(j+1)+"} = "+ele_string+" $;\n\n~\\\\\n\n")
                    
                    texfile.write("\tH^{Z}_{"+str(i+1)+str(j+1)+"} = "+ele_string+"\n")
                    texfile.write("\\end{dmath}\n\n")
            
            texfile.write("\\end{dgroup*}\n")
            
            texfile.write("\n")
            
            texfile.write(r'\end{document}')
            
            texfile.close()
            
            # if the system has the command 'pdflatex': compile the .tex
            try:
                import platform
                
                System = platform.system()
                
                if System == 'Windows':
                    pdf_flag = subprocess.run(r"pdflatex kp_Hamiltonian-Zeeman's_coupling(without_numerical_parameters).tex")
                
                else:
                    pdf_flag = subprocess.run("pdflatex kp_Hamiltonian-Zeeman\\'s_coupling\(without_numerical_parameters\).tex", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                #delete the compile log
                if pdf_flag.returncode == 0:
                    
                    try:
                        os.remove(texfile_name+".log")
                        os.remove(texfile_name+".aux")
                        os.remove(texfile_name+".out") # may not exist
                        
                    except:
                        pass
                    
                else:
                    pass
                
            except:
                pass
            
    else:
        sum_Zeeman = None
    
    warnings.filterwarnings("ignore")
    
    return sum_kp,sum_Zeeman
###############################################################################



###############################################################################
if __name__ == "__main__":
    
    Symmetry = {
    'C2z' : {
        'rotation_matrix': sp.Matrix([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]]),
        'repr_matrix': sp.Matrix([[-I,0, 0,0],
                                  [0,I,0,0],
                                  [0,0,I,0],
                                  [0,0,0,-I]]),     #GM8
        'repr_has_cc': False  
    },
    
    
    
    'C2y' : {
        'rotation_matrix': sp.Matrix([[-1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, -1]]),
        'repr_matrix': sp.Matrix([[0,-1, 0,0],
                                  [1,0,0,0],
                                  [0,0,0,I],
                                  [0,0,I,0]]),     #GM8
        'repr_has_cc': False  
    },
    
    
    
    'C3111' : {
        'rotation_matrix': sp.Matrix([[0, 0, 1],
                                      [1, 0, 0],
                                      [0, 1, 0]]),
        'repr_matrix': sp.Matrix([[sp.exp(5*I*sp.pi/12)*sp.sqrt(2)/2,sp.exp(-I*sp.pi/12)*sp.sqrt(2)/2,0,0],
                                    [sp.exp(5*I*sp.pi/12)*sp.sqrt(2)/2,sp.exp(11*I*sp.pi/12)*sp.sqrt(2)/2,0,0],
                                    [0,0,sp.exp(-5*I*sp.pi/12)*sp.sqrt(2)/2,sp.exp(-5*I*sp.pi/12)*sp.sqrt(2)/2],
                                    [0,0,sp.exp(I*sp.pi/12)*sp.sqrt(2)/2,sp.exp(-11*I*sp.pi/12)*sp.sqrt(2)/2]]),
        'repr_has_cc': False  
    },
    
    
    
    'M1-10' : {
        'rotation_matrix': sp.Matrix([[0, 1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]]),
        'repr_matrix': sp.Matrix([[0,0,-1,0],
                                  [0,0,0,-1],
                                  [1,0,0,0],
                                  [0,1,0,0]]),
        'repr_has_cc': False  
    }
    ,
    
    
    
    
    'T' : {
        'rotation_matrix': sp.eye(3),
        'repr_matrix': sp.Matrix([[0, 0, -sp.sqrt(2)/2-I*sp.sqrt(2)/2,0],
                                  [0, 0, 0, sp.sqrt(2)/2-I*sp.sqrt(2)/2],
                                  [sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0],
                                  [0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0]]),
        'repr_has_cc': True  
    }
    }

    
    log = 0
    print_flag = 2
    gfactor = 1
    order = 2         # Generate all kp models with order <= 5
    
    get_std_kp_Zeeman_no_coe(Symmetry=Symmetry,order=order,gfactor=gfactor,print_flag=print_flag,log=log)
 
    
    
    
    