# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 00:31:56 2023
Last modified on Wedn Dec 6 14:45:00 2023

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Load the data (generators' DFT corepresentation and momentum/spin elements) in the input folder
"""

import numpy as np
import os
import sys
import scipy.io as scio
import re
from sympy import pi,sqrt,sin,cos
import sympy as sp
from ._numeric_kp import flag_reduce_Symmetry, get_corep_dim_reduce,bandreplist2rep,get_block_dim,get_max_dim
from ._transform_matrix import CoRepresentationInequivalenceERROR

#import numba

###############################################################################

set_vasp_zero = True
eig_cut = 0.001

###############################################################################



###############################################################################
class EIGENVALNotFindERROR(Exception):
    '''
    if the file 'EIGENVAL' not exists: raise this ERROR
    '''
    pass

class EIGENVALERROR(Exception):
    '''
    if the file 'EIGENVAL' is empty: raise this ERROR
    '''
    pass
###############################################################################



###############################################################################
def load_data_matlab(operator_path = os.path.join(os.getcwd(),'operator.mat'),data_path = os.path.join(os.getcwd(),'data.mat'),eigen_energy_path = os.path.join(os.getcwd(),"data\\EIGENVAL.Pi")):
    '''
    if you have matlab, you can directly use it to read the data transformed by matlab
    '''
    data = scio.loadmat(data_path)
    del data['__header__']
    del data['__version__']
    del data['__globals__']
    
    operator = scio.loadmat(operator_path)
    del operator['__header__']
    del operator['__version__']
    del operator['__globals__']
    
    file=open(eigen_energy_path,"r")
    
    for i in range(8):
        file.readline()

    eigen_energy=[]
    
    for i in file.readlines():
        i=i.strip().split(' ')
        eigen_energy.append(eval(i[-1]))
    
    
    file.close()
    
    
    for i in operator.keys():
        operator[i]=np.matrix(operator[i])
        
    return operator,data,eigen_energy
###############################################################################



###############################################################################
def has_digit(input_string):
    '''
    judge whether there is at least one numeric characters in a string 
    '''
    
    judge_s = any(char.isdigit() for char in input_string)
    
    return judge_s
###############################################################################



###############################################################################
def get_operator_data_one(path):
    '''
    read one input data(XXX.m file) got by vasp_song and arrange it to a matrix

    Parameters
    ----------
    path : string
        the path of the file which is to be dealed with.

    Returns
    -------
    operator : dictionary
        key->value : name -> matrix

    '''
    
    file = open(path,'r')
    operator = dict()
    
    part = 0
    col = 0
    row = 0
    tmp = 0
    
    for i in file.readlines():
        i=i.strip()
       # print(i)
       
        # Skip comment lines
        if i=='' or i[0]=='%':
            continue
        
        i=i.split(' ')
        # remove all '' in i
        while True:
            
            try:
                i.remove('')
                
            except:
                break
        
        
        if i[0][0].isalpha(): # judge if the first charactor is a letter
            
            name = i[0][:-1] # the generator/matrix element name
            
            if name in operator.keys():
                
                col = col + part
                row = 0
                position = 0
                tmp += part
                
                continue
            
            dimension1 = eval(i[2][:-1])
            dimension2 = eval(i[4][:-3])
            dimension = max(dimension1,dimension2)
            part = min(dimension1,dimension2)
            
            
            operator[name] = np.zeros((dimension,dimension),dtype = np.complex128)
            operator[name] = np.matrix(operator[name],dtype=np.complex128)
            
            col = 0
            row = 0
            tmp = 0
            position = 0
            
        else:
            
            for j in i:
                
                if has_digit(j):
                    j=j.strip().replace('i','j') #in python, 'j' is an imaginary unit
                    
                else:
                    continue
                
                if j[-1] == ")":
                    j = j[:-1]
                    
                if j[0] == "(":
                    j = j[1:]
                    
                operator[name][row,col] += eval(j)
                
                if position == 0:
                    position += 1
                    
                else:  
                    col += 1
                    position = 0
                
                if (col-tmp) / part >=1 :
                    row += 1
                    col -= part
                
                if col>=dimension:
                    row += 1
                    col-=dimension-part
    
    file.close()
    
    return operator
###############################################################################



###############################################################################
def get_opertors_or_data(vaspMAT,namelist):    
    '''
    get all DFT generators' representations and momentum elements and spin elements
    
    Parameters
    ----------
    vaspMAT : string
        the path of the folder of the .m files to import.
    namelist : list
        the list of generators' names or spin/generalized momentum matrices' names

    Returns
    -------
    operators : dictionary
        key->values : name->matrix

    '''
    
    folder_path = vaspMAT
    operators=dict()
    
    for i in namelist:
        i = "MAT_"+i+".m" # file name of i
        
        operator_i = get_operator_data_one(os.path.join(folder_path,i))
        
        # record
        operators.update(operator_i)
        
    return operators
###############################################################################



###############################################################################
def get_interest_band_range(file_path):
    '''
    get the range of the band which we are interested in

    Parameters
    ----------
    file_path : string
        the .m file to read the band range of interest.

    Returns
    -------
    band_start : integer
        the id of start band we are interested in.
    band_end : integer
        the id of end band we are interested in

    '''
    
    file = open(file_path,"r")
    
    for i in file.readlines():
        
        i=i.strip().split(' ')
        
        while True:
            # remove all '' in i
            try:
                i.remove('')
                
            except:
                break

        # find the line like % On bands :   33   34   35   36   37   38
        if 'On' in i and 'bands' in i:
             band_list = []
             
             for j in i:
                 if j.isdigit():
                     band_list.append(eval(j))
                     
             #band_start = min(band_list)
             #band_end = max(band_list)
             band_interest_set = np.array(band_list, dtype=np.int64)
             
             break
         
    file.close()
    
    return band_interest_set
###############################################################################



###############################################################################
def get_bstart_bend(vaspMAT):
    '''
    obtain the bstart and bend in INCAR.song

    Parameters
    ----------
    vaspMAT : str
        the path of the file folder of the input files.

    Returns
    -------
    band_start_all : integer
        bstart in vasp_kp.
    band_end_all : integer
        bend in vasp_kp.

    '''
    
    folder_path = vaspMAT
    
    file = open(os.path.join(folder_path,"MAT_Pi.m"),'r')
    
    band_flag_start = True
    band_last_record = None
    band_start_all = 0
    band_end_all = 0
    
    while True:
        i = file.readline()
        
        if i == '':
            break
        
        if not band_flag_start:
            
            if '% On bands :' not in i:
                break
        
        if '% On bands :' in i:
            
            if band_flag_start:
                band_flag_start = False
                band_last_record = i
                i = i.strip().split(' ')
                
                while True:
                    
                    try:
                        i.remove('')
                        
                    except:
                        break
                    
                
                band_start_all = eval(i[4])
                
            else:
                band_last_record = i
    
    
    band_last_record = band_last_record.strip().split(' ')
    
    while True:
        
        try:
            band_last_record.remove('')
            
        except:
            break
        
    band_end_all=eval(band_last_record[-1])
    
    file.close()
    
    return band_start_all,band_end_all
###############################################################################



###############################################################################
def load_data(Symmetry,vaspMAT,gfactor=1,repr_split = True, data_name_list=["Pi","sig"]):
    '''
    load all the input files

    Parameters
    ----------
    Symmetry : dictionary
        a dictionary according to the user's input, the rotation matrices, the standard corepresentation matrices of all generators.
    vaspMAT : string
        the path of the file folder of the input files.
    gfactor : integer, optional
        Whether to solve the Zeeman‘s coupling. The default is 1.
    data_name_list : list, optional
        The name of the generalized momentum matrices and the spin matrices. The default is ["Pi","sig"].

    Returns
    -------
    operator : dictionary
        VASP corepresentation.
    data : dictionary
        the generalized momentum matrices and the spin matrices.
    eigen_energy : list
        the list of eigen energies obtained by VASP.
    band_interest_set : list
        id list of the bands of interest.
    '''
    
    folder_path = vaspMAT
    
    try:
        file = open(os.path.join(folder_path,"MAT_Pi.m"),'r')
    except:
        try:
            file = open(os.path.join(folder_path,"MAT_pi.m",'r'))
        except:
            print("ERROR: Cannot fild the file 'MAT_Pi.m'!")
            sys.exit()
    file.readline()
    file.readline()
    kpoints = file.readline().strip().split(' ')
    file.close()
    
    while '' in kpoints:
        kpoints.remove('')
        
    kpoints = np.array([eval(i) for i in kpoints[3:6]])
    
    # read the EIGENVAL file
    try:
        file=open(os.path.join(folder_path,"EIGENVAL"),"r")
        
    except:
        
        try:
            file=open(os.path.join(folder_path,"EIGENVAL.dat"),"r")
            
        except:
            
            try:
                file=open(os.path.join(folder_path,"EIGENVAL.Pi"),"r")
                
            except:
                
                #raise EIGENVALNotFindERROR("Cannot find the file 'EIGENVAL'!")
                print("ERROR: Cannot find the file 'EIGENVAL'!")
                sys.exit()
    
    for i in range(5):
        file.readline()
    
    band_inf = file.readline()
    band_inf = band_inf.strip().split(' ')
    
    while True:
        try:
            band_inf.remove('')
        except:
            break
    
    num_k = eval(band_inf[1])
    num_band = eval(band_inf[2])
    # if vmat_k > num_k:
    #     print("ERROR: vmat_k is larger than the number of total kpoints!")
    #     sys.exit()
    
    # skipline = (vmat_k - 1)*(num_band + 2) + 2
    
    # for i in range(skipline):
    #     file.readline()

    find_flag = False
    for i in range(num_k):
        file.readline()
        kpoints_eig = file.readline().strip().split(' ')
        while '' in kpoints_eig:
            kpoints_eig.remove('')
        kpoints_eig = np.array([eval(l) for l in kpoints_eig[0:3]])
        try:
            dif = np.linalg.norm(kpoints_eig-kpoints)
        except:
            if i==0: 
                print("ERROR: Empty file 'EIGENVAL'!!! You should use the 'EIGENVAL' generated by VASP when computing wavefunction 'WAVECAR' but not the 'EIGENVAL' generated by vasp2mat!")
            else:
                print("ERROR: Please check the file 'EIGENVAL'!!!")
            sys.exit()
        
        if dif < 1e-6:
            find_flag = True
            break
        else:
            for j in range(num_band):
                file.readline()
    
    if not find_flag:
        print("ERROR: Cannot find KPOINT "+str(kpoints[0])+' '+str(kpoints[1])+'  '+str(kpoints[2])+' in EIGENVAL!!!')
    
    eigen_energy = []
    eigen_start = 0
    
    for i in file.readlines():
        i=i.strip().split(' ')
        
        # remove all '' in i
        while True:
            
            try:
                i.remove("") #remove all empty elements
                
            except:
                
                if i==[]:
                    break
                
                eigen_energy.append(eval(i[1])) #zeroth element: band id; first element: eigen energy
                
                if eigen_start == 0:
                    eigen_start = eval(i[0])
                
                break
    
    file.close()
    
    if len(eigen_energy) == 0:
        #raise EIGENVALERROR("ERROR: Empty EIGENVAL file! You should not use EIGENVAL file generated by vasp2mat!!!")
        print("ERROR: Empty EIGENVAL file! You should not use EIGENVAL file generated by vasp2mat!!!")
        sys.exit()
        
    
    
    if 'repr_matrix' not in list(Symmetry.values())[0].keys():
        for i in Symmetry.keys():
            Symmetry[i]['repr_matrix'] = bandreplist2rep(Symmetry[i]['band_repr_matrix_list'])
            del Symmetry[i]['band_repr_matrix_list']
            
    operator_name_list = Symmetry.keys()
    
    band_interest_set = get_interest_band_range(os.path.join(folder_path,"MAT_"+list(operator_name_list)[0]+'.m'))
    
    operator = get_opertors_or_data(folder_path,operator_name_list)
                
    
    # unify dimension, scale parameters
    band_start_all,band_end_all = get_bstart_bend(folder_path)
    
    #band_start -= band_start_all - 1
    #band_end -= band_start_all - 1
    band_interest_set -= band_start_all - 1
    #print(band_start)
    #print(band_end)
    #print(band_start_all - eigen_start)
    eigen_energy = eigen_energy[band_start_all - eigen_start:]
    eigen_energy_array = np.array(eigen_energy,dtype=float)
    #print(eigen_energy[0])
    
    if set_vasp_zero or flag_reduce_Symmetry:
        
        eigen_energy_interest = eigen_energy_array[band_interest_set-1]
        
        id_list = list(range(len(eigen_energy_interest)))
        id_list_ = id_list.copy()
        
        while len(id_list)!= 0:
            i = id_list[0]
            del id_list[0]
            energy = eigen_energy_interest[i]
            for j in id_list:
                if abs(energy - eigen_energy_interest[j])>eig_cut:
                    for k in operator.keys():

                        operator[k][i,j] = 0
                        operator[k][j,i] = 0
        
        dim = list(operator.values())[0].shape[0]
        
        for k in operator.keys():
            for i in range(dim):
                for j in range(dim):
                    if abs(operator[k][i,j])<1e-3:
                        operator[k][i,j] = 0
        
        sum_operator = np.zeros((dim,dim),dtype = np.float128)
        for i in operator.values():
            sum_operator += np.abs(i)
        
        operator_irrep_dim_list = get_block_dim(sp.Matrix(sum_operator))

        energy_irrep_dim_list = [1]
        
        energy = eigen_energy_interest[0]
        for i in id_list_[1:]:
            if abs(energy - eigen_energy_interest[i])<eig_cut:
                energy_irrep_dim_list[-1] += 1
            else:
                energy_irrep_dim_list.append(1)
                energy = eigen_energy_interest[i]
    
    
    
    if flag_reduce_Symmetry:
        try:
            Symmetry_copy = Symmetry.copy()
            #print(operator_irrep_dim_list,energy_irrep_dim_list)
            if repr_split:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[operator_irrep_dim_list])
            else:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[[list(operator.values())[0].shape[0]]])

            Symmetry_copy = Symmetry_tmp.copy()
            for i in Symmetry_tmp.keys():
                Symmetry_copy[i] = Symmetry_tmp[i].copy()
                Symmetry_copy[i]['band_repr_matrix_list'] = Symmetry_tmp[i]['band_repr_matrix_list'].copy()
            
            dimension = 0
            
            for i in range(len(dim_list)):
                conjugate_flag = False
                sym_chr_list = []
                opr_chr_list = []
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:

                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(np.conjugate(sym_chr_list[-1])-opr_chr_list[-1]) < 1e-2:
                                conjugate_flag = True
                            
                if conjugate_flag:
                    
                    for j in operator_name_list:
                        Symmetry_tmp[j]['band_repr_matrix_list'][i] = sp.conjugate(Symmetry_tmp[j]['band_repr_matrix_list'][i])
                    
                    
                    
                dimension += dim_list[i]
            
            dimension = 0

            
            for i in range(len(dim_list)):
                
                sym_chr_list = []
                opr_chr_list = []
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:
                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(sym_chr_list[-1]+opr_chr_list[-1]) > 1e-2:
                            #raise CoRepresentationInequivalenceERROR("ERROR for representation " + str(i+1) + " of operator " + j)
                               
                                break



        except:
            Symmetry_copy = Symmetry.copy()
            #print(operator_irrep_dim_list,energy_irrep_dim_list)
            if repr_split:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[operator_irrep_dim_list,energy_irrep_dim_list])
            else:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[[list(operator.values())[0].shape[0]]])
            Symmetry_copy = Symmetry_tmp.copy()
            for i in Symmetry_tmp.keys():
                Symmetry_copy[i] = Symmetry_tmp[i].copy()
                Symmetry_copy[i]['band_repr_matrix_list'] = Symmetry_tmp[i]['band_repr_matrix_list'].copy()
            
            dimension = 0
            conjugate_record = []
            
            for i in range(len(dim_list)):
                conjugate_flag = False
                sym_chr_list = []
                opr_chr_list = []
                
                
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:
                        
                        #print(str(sym_chr_list[-1])+'\\'+str(opr_chr_list[-1]),end='   ')
                        #print(f'{sym_chr_list[-1]:>11.2f}',end='\\')
                       # print(f'{sym_chr_list[-1]:>11.2f}',end='   ')
                        #print(f'{opr_chr_list[-1]:>11.2f}',end='   ')
                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(np.conjugate(sym_chr_list[-1])-opr_chr_list[-1]) < 1e-2:
                                conjugate_flag = True
                            
                    #if Symmetry_tmp[j]['repr_has_cc']:
                     #   print(f'{sym_chr_list[-1]:>11.2f}',end='K   ')
                        #print(f'{opr_chr_list[-1]:>11.2f}',end='K ')
                        
                        
                    
                if conjugate_flag:
                    
                    for j in operator_name_list:
                        Symmetry_tmp[j]['band_repr_matrix_list'][i] = sp.conjugate(Symmetry_tmp[j]['band_repr_matrix_list'][i])
                    
                    
                    
                dimension += dim_list[i]
                conjugate_record.append(conjugate_flag)
            
            
            
            
            dimension = 0
            print("Low-energy bands:" ,band_interest_set)
            print("Energies of these bands:" ,eigen_energy_array[band_interest_set-1])
            print("Number of low-energy bands:",sum(dim_list))
            print("Number of IrReps:",len(dim_list))
            print("IrReps' dimensions:",dim_list)
            print('''Loading D^std from "mat2kp.in" successfully! ''')
            print("Traces of the matrix representations (D^num) obtained by vasp2mat are given below:")
            
            print("IrRep ",end=' ')
            print("Dim " ,end=' ')
            for i in operator_name_list:
                print(f"{i:>11s}",end='   ')
                
            print()
            ERROR_string = ''
            warning_string_list = []
            
            for i in range(len(dim_list)):
                
                sym_chr_list = []
                opr_chr_list = []
                warning_i = []
                
                pr_str = "{:>5d}".format(i+1)
                #print(f"{i+1:>5d}",end='')
                print(pr_str.replace("j","i"),end='')
                
                pr_str = "{:>5d}".format(dim_list[i])
                #print(f"{dim_list[i]:>5d}",end='  ')
                print(pr_str.replace("j","i"),end='  ')
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:
                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(sym_chr_list[-1]+opr_chr_list[-1]) > 1e-2:
                            #raise CoRepresentationInequivalenceERROR("ERROR for representation " + str(i+1) + " of operator " + j)
                                ERROR_string = "ERROR: The condition trace(D^std(" + j+"))=trace(D^num("+j+")) is not satisfied in IrRep "+str(i+1)+" !!!"
                                print("      ERROR",end = '')    
                                ERROR_string2 = "trace(D^std(" + j+"))=" + "{:>11.2f}".format(np.trace(np.array(Symmetry_copy[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                                ERROR_string2 += ",    trace(D^num(" + j+"))=" + "{:>11.2f}".format(opr_chr_list[-1])
                                break
                            else:
                                Symmetry_tmp[j]['band_repr_matrix_list'][i] = - Symmetry_tmp[j]['band_repr_matrix_list'][i]
                                
                                warning_string = "Warning: -D^std("+j+") is used in IrRep "+str(i+1)+"!!!"
                                warning_i.append(warning_string)
                                
                        pr_str = "{:>11.2f}".format(opr_chr_list[-1])
                        print(pr_str.replace('j','i'),end='   ',flush=True)
                    else:
                        pr_str = f"{opr_chr_list[-1]:>11.2f}"
                        print(pr_str.replace('j','i'),end=' K  ')
                
                warning_string_list.append(warning_i)
                dimension += dim_list[i]
                
                if len(ERROR_string)!=0:
                    print('''\nNOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                    
                    print(ERROR_string+"    "+ERROR_string2.replace('j', 'i')+'.')
                    
                    
                    #print("Please rewrite the standard representation with correct traces! This could be caused by wrong std matrice in mat2kp.in or wrong matrices obtained by vasp2mat! Please check if the WAVECAR are not empty when run vasp2mat!")
                    print('''Please check: 1. Whether the standard representation matrices in "mat2kp.in" are correct.''')
                    print('''              2. Whether WAVECAR is not empty and LWAVE=.TRUE. is set in INCAR when running vasp2mat.''')
                    print('''              3. Whether rot_n and rot_tau in "INCAR.mat" in INCAR.mat are correct when running vasp2mat.''')
                    if repr_split:
                        print('''              4. If the above points are all checked but mat2kp still fails, or you do not use representation matrices from BCS Server, you can set "repr_split = False" in "mat2kp.in" and run mat2kp again.''')
                    sys.exit()
                    
                print()
                
            for i in operator_name_list:
                if Symmetry[i]['repr_has_cc']:
                    print("K is the complex conjugation for anti-unitary operations.")
                    break
            for j in operator_name_list:
                Symmetry_tmp[j]['repr_matrix'] = bandreplist2rep(Symmetry_tmp[j]['band_repr_matrix_list'])
            
            exe = 0
            for i in range(len(dim_list)):
                if conjugate_record[i]:
                    if exe == 0:
                        print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                        exe += 1
                    #print("Warning: To satisfy the character condition, Irrep "+str(i+1)+" is conjugated!!!")
                    print("Warning: standard IrRep "+str(i+1)+" is conjugated!!!")
                    
                for j in warning_string_list[i]:
                    if exe == 0:
                        print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                        exe += 1
                    print(j)
            
            for i in operator_name_list:
                Symmetry[i] = Symmetry_tmp[i]
            #Symmetry = Symmetry_tmp.copy()
            #print(Symmetry)
    
    
        else:
            
            Symmetry_copy = Symmetry.copy()
            #print(operator_irrep_dim_list,energy_irrep_dim_list)
            if repr_split:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[operator_irrep_dim_list])
            else:
                Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[[list(operator.values())[0].shape[0]]])
            
            Symmetry_copy = Symmetry_tmp.copy()
            for i in Symmetry_tmp.keys():
                Symmetry_copy[i] = Symmetry_tmp[i].copy()
                Symmetry_copy[i]['band_repr_matrix_list'] = Symmetry_tmp[i]['band_repr_matrix_list'].copy()
            
            dimension = 0
            conjugate_record = []
            
            for i in range(len(dim_list)):
                conjugate_flag = False
                sym_chr_list = []
                opr_chr_list = []
                
                
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:
                        
                        #print(str(sym_chr_list[-1])+'\\'+str(opr_chr_list[-1]),end='   ')
                        #print(f'{sym_chr_list[-1]:>11.2f}',end='\\')
                       # print(f'{sym_chr_list[-1]:>11.2f}',end='   ')
                        #print(f'{opr_chr_list[-1]:>11.2f}',end='   ')
                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(np.conjugate(sym_chr_list[-1])-opr_chr_list[-1]) < 1e-2:
                                conjugate_flag = True
                            
                    #if Symmetry_tmp[j]['repr_has_cc']:
                     #   print(f'{sym_chr_list[-1]:>11.2f}',end='K   ')
                        #print(f'{opr_chr_list[-1]:>11.2f}',end='K ')
                        
                        
                    
                if conjugate_flag:
                    
                    for j in operator_name_list:
                        Symmetry_tmp[j]['band_repr_matrix_list'][i] = sp.conjugate(Symmetry_tmp[j]['band_repr_matrix_list'][i])
                    
                    
                    
                dimension += dim_list[i]
                conjugate_record.append(conjugate_flag)
            
            
            
            
            dimension = 0
            print("Low-energy bands:" ,band_interest_set)
            print("Energies of these bands:" ,eigen_energy_array[band_interest_set-1])
            print("Number of low-energy bands:",sum(dim_list))
            print("Number of IrReps:",len(dim_list))
            print("IrReps' dimensions:",dim_list)
            print('''Loading D^std from "mat2kp.in" successfully! ''')
            print("Traces of the matrix representations (D^num) obtained by vasp2mat are given below:")
            
            print("IrRep ",end=' ')
            print("Dim " ,end=' ')
            for i in operator_name_list:
                print(f"{i:>11s}",end='   ')
                
            print()
            ERROR_string = ''
            warning_string_list = []
            
            for i in range(len(dim_list)):
                
                sym_chr_list = []
                opr_chr_list = []
                warning_i = []
                
                pr_str = "{:>5d}".format(i+1)
                #print(f"{i+1:>5d}",end='')
                print(pr_str.replace("j","i"),end='')
                
                pr_str = "{:>5d}".format(dim_list[i])
                #print(f"{dim_list[i]:>5d}",end='  ')
                print(pr_str.replace("j","i"),end='  ')
                
                for j in operator_name_list:
                    sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                    opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                    
                    if not Symmetry_tmp[j]['repr_has_cc']:
                        
                        if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                            if abs(sym_chr_list[-1]+opr_chr_list[-1]) > 1e-2:
                            #raise CoRepresentationInequivalenceERROR("ERROR for representation " + str(i+1) + " of operator " + j)
                                ERROR_string = "ERROR: The condition trace(D^std(" + j+"))=trace(D^num("+j+")) is not satisfied in IrRep "+str(i+1)+" !!!"
                                print("      ERROR",end = '')    
                                ERROR_string2 = "trace(D^std(" + j+"))=" + "{:>11.2f}".format(np.trace(np.array(Symmetry_copy[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                                ERROR_string2 += ",    trace(D^num(" + j+"))=" + "{:>11.2f}".format(opr_chr_list[-1])
                                break
                            else:
                                Symmetry_tmp[j]['band_repr_matrix_list'][i] = - Symmetry_tmp[j]['band_repr_matrix_list'][i]
                                
                                warning_string = "Warning: -D^std("+j+") is used in IrRep "+str(i+1)+"!!!"
                                warning_i.append(warning_string)
                                
                        pr_str = "{:>11.2f}".format(opr_chr_list[-1])
                        print(pr_str.replace('j','i'),end='   ',flush=True)
                    else:
                        pr_str = f"{opr_chr_list[-1]:>11.2f}"
                        print(pr_str.replace('j','i'),end=' K  ')
                
                warning_string_list.append(warning_i)
                dimension += dim_list[i]
                
                if len(ERROR_string)!=0:
                    print('''\nNOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                    
                    print(ERROR_string+"    "+ERROR_string2.replace('j', 'i')+'.')
                    
                    
                    #print("Please rewrite the standard representation with correct traces! This could be caused by wrong std matrice in mat2kp.in or wrong matrices obtained by vasp2mat! Please check if the WAVECAR are not empty when run vasp2mat!")
                    print('''Please check: 1. Whether the standard representation matrices in "mat2kp.in" are correct.''')
                    print('''              2. Whether WAVECAR is not empty and LWAVE=.TRUE. is set in INCAR when running vasp2mat.''')
                    print('''              3. Whether rot_n and rot_tau in "INCAR.mat" in INCAR.mat are correct when running vasp2mat.''')
                    if repr_split:
                        print('''              4. If the above points are all checked but mat2kp still fails, or you do not use representation matrices from BCS Server, you can set "repr_split = False" in "mat2kp.in" and run mat2kp again.''')
                    sys.exit()
                    
                print()
                
            for i in operator_name_list:
                if Symmetry[i]['repr_has_cc']:
                    print("K is the complex conjugation for anti-unitary operations.")
                    break
            for j in operator_name_list:
                Symmetry_tmp[j]['repr_matrix'] = bandreplist2rep(Symmetry_tmp[j]['band_repr_matrix_list'])
            
            exe = 0
            for i in range(len(dim_list)):
                if conjugate_record[i]:
                    if exe == 0:
                        print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                        exe += 1
                    #print("Warning: To satisfy the character condition, Irrep "+str(i+1)+" is conjugated!!!")
                    print("Warning: standard IrRep "+str(i+1)+" is conjugated!!!")
                    
                for j in warning_string_list[i]:
                    if exe == 0:
                        print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                        exe += 1
                    print(j)
            
            for i in operator_name_list:
                Symmetry[i] = Symmetry_tmp[i]
            #Symmetry = Symmetry_tmp.copy()
            #print(Symmetry)
    print('''Loading D^num from "MAT_*.m" successfully! ''')
            

    
    
    
    
    
    
    
    
    
    
    
    
    if gfactor == 0:
        data_name_list=["Pi"]
        data = get_opertors_or_data(folder_path,data_name_list)
        
    elif os.path.exists(os.path.join(folder_path,"MAT_sig.m")):
        data = get_opertors_or_data(folder_path,data_name_list)
        
    else:
        print("Warning: No MAT_sig.m file! All spin matrices are set to zero!")
        data_name_list=["Pi"]
        operator_name_list = list(operator_name_list)
        data = get_opertors_or_data(folder_path,data_name_list)
        data['sigx']=np.zeros_like(operator[operator_name_list[0]],dtype=np.complex128)
        data['sigy']=np.zeros_like(operator[operator_name_list[0]],dtype=np.complex128)
        data['sigz']=np.zeros_like(operator[operator_name_list[0]],dtype=np.complex128)
    
    

    #print("Finish loading data!")
    
    return operator,data,eigen_energy,band_interest_set,dim_list
###############################################################################



###############################################################################
def get_rot_information(file_name):
    '''
    Read rotation information from XXX.m file
    Notice: this function are not used in VASP2KP

    Parameters
    ----------
    file_name : string
        The path of the file.

    Returns
    -------
    axis : list
        DESCRIPTION.
    angle : float
        Rotation angle.
    det : float
        Determinant of the rotation matrix.
    tau : float
        The spatial part of the symmetry operator.
    time_rev : boolean
        If the symmetry operator has time reversal.

    '''
    file = open(file_name,"r")
    find_flag = 0
    time_rev = False
    
    for i0 in file.readlines():
        i = i0.strip().split(' ')
        
        if len(i) == 0:
            continue
        
        if i[0] != "%":
            continue
        
        while True:
            
            try:
                i.remove("")
                
            except:
                break
            
        if len(i) <= 1:
            continue
        
        if i[1] == 'det':
            det = eval(i[3])
            find_flag += 1
            
        elif i[1] == 'rot_angle':
            angle = round(eval(i[3]),6)*pi/180
            find_flag += 1
            
        elif i[1] == 'with':
            
            if i[3] == 'reversal':
                time_rev = True
                find_flag += 1
    
        elif i[1] == 'rot_tau':
            listf = re.split(r'[\[\ \]]',i0.strip())
            
            while True:
                
                try:
                    listf.remove("")
                    
                except:
                    break
                
            tau = [round(eval(listf[k]),6) for k in range(3,6)]
            find_flag += 1
        
        elif i[1] == 'n':
            listf = re.split(r'[\[\ \]]',i0.strip())
            
            while True:
                
                try:
                    listf.remove("")
                    
                except:
                    break
                
            axis = [round(eval(listf[k]),6) for k in range(3,6)]
            find_flag += 1
        
        if find_flag == 5:
            break
        
    file.close()
    
    return axis,angle,det,tau,time_rev
###############################################################################     



###############################################################################
def get_O3_matrix(angle, axis, det):
    '''
    By Yi Jiang in kdotp-generator and revised by Sheng Zhang
    This function is used to generate O(3) matrix in Cartesian coordinates.
    Inputs: rotation angle (-360,360), rotation axis (length-3 array), and determinate (+1 or -1)
    
    Formula:
    R[i,j] = cos(w) delta_ij + (1 - cos(w)) * ni * nj - sin(w) sum(epsilon_ijk nk)
    '''
    #assert -2 * pi <= angle <= 2 * pi, ('angle should be in radian', angle)
    
    assert angle in [-180,-120,-90,-60,0,60,90,120,180], ('angle should be in degree', angle)
    
    assert det == 1 or det == -1, det
    
    angle = angle*pi/180
    
    n = sp.Matrix(axis) / sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)

    R = sp.zeros(3, 3)
    
    for i in range(3):
        
        for j in range(3):
            
            if i == j:
                R[i, j] += cos(angle)
                
            R[i, j] += (1 - cos(angle)) * n[i] * n[j]
            
    R += -sin(angle) * sp.Matrix([[0, n[2], -n[1]], [-n[2], 0, n[0]], [n[1], -n[0], 0]])
    R *= det
    
    return R
###############################################################################
    
    
    
###############################################################################        
if __name__ == "__main__":
    if False:
        a=get_opertors_or_data('InTe',["C3z",'Mz','Mx','T'])
        operator = scio.loadmat('operator.mat')
        
        del operator['__header__']
        del operator['__version__']
        del operator['__globals__']
        
    if True:
        
        from sympy import I
        
        Symmetry = {

        'C2z' : {
            'rotation_matrix': sp.Matrix([[-1,0,0],
                                          [0, -1, 0],
                                          [0, 0, 1]]),
            'repr_matrix': sp.Matrix([[-I,0,0,0,0,0],
                                      [0,I,0,0,0,0],
                                      [0,0,-I,0,0,0],
                                      [0,0,0,I,0,0],
                                      [0,0,0,0,-I,0],
                                      [0,0,0,0,0,I]]),   #GM9  9  8
            'repr_has_cc': False  
        },



        'C2y' : {
            'rotation_matrix': sp.Matrix([[-1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, -1]]),
            'repr_matrix': sp.Matrix([[0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0],
                                      [sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0,0],     
                                      [0,0,0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0],
                                      [0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0],     #GM9  9  8
                                      [0,0,0,0,0,sp.sqrt(2)/2-I*sp.sqrt(2)/2],
                                      [0,0,0,0,-sp.sqrt(2)/2-I*sp.sqrt(2)/2,0]]),
            
            
            'repr_has_cc': False  
        },


        'C4z' : {
            'rotation_matrix': sp.Matrix([[0, -1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]]),
            'repr_matrix': sp.Matrix([[sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0,0,0,0],
                                      [0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0,0,0],
                                      [0,0,sp.sqrt(2)/2-I*sp.sqrt(2)/2,0,0,0],
                                      [0,0,0,sp.sqrt(2)/2+I*sp.sqrt(2)/2,0,0],
                                      [0,0,0,0,-sp.sqrt(2)/2+I*sp.sqrt(2)/2,0],
                                      [0,0,0,0,0,-sp.sqrt(2)/2-I*sp.sqrt(2)/2]]),    #GM9  9  8

            'repr_has_cc': False  
        },




        'P' : {
            'rotation_matrix': sp.Matrix([[-1,0,0], 
                                          [0, -1, 0], 
                                          [0, 0, -1]]),
            'repr_matrix': sp.Matrix([[-1,0,0,0,0,0],
                                      [0,-1,0,0,0,0],
                                      [0,0,-1,0,0,0],
                                      [0,0,0,-1,0,0],
                                      [0,0,0,0,-1,0],
                                      [0,0,0,0,0,-1]]),
            'repr_has_cc': False  
        },

        'T' : {
            'rotation_matrix': sp.eye(3),
            'repr_matrix': sp.Matrix([[0,-1,0,0,0,0],
                                      [1,0,0,0,0,0],
                                      [0,0,0,-1,0,0],
                                      [0,0,1,0,0,0],
                                      [0,0,0,0,0,-1],
                                      [0,0,0,0,1,0]]),
            'repr_has_cc': True  
        }
        }

        vaspMAT = 'Cd3As2'
        operator,data,eigen_energy,band_start,band_end = load_data(Symmetry,vaspMAT,gfactor=1,data_name_list=["Pi","sig"])
    
    
    
 
    
            
            
"""
    if flag_reduce_Symmetry:
        
        Symmetry_copy = Symmetry.copy()
        #print(operator_irrep_dim_list,energy_irrep_dim_list)
        Symmetry_tmp, dim_list = get_corep_dim_reduce(Symmetry_copy,[operator_irrep_dim_list])
        Symmetry_copy = Symmetry_tmp.copy()
        for i in Symmetry_tmp.keys():
            Symmetry_copy[i] = Symmetry_tmp[i].copy()
            Symmetry_copy[i]['band_repr_matrix_list'] = Symmetry_tmp[i]['band_repr_matrix_list'].copy()
        
        dimension = 0
        conjugate_record = []
        
        for i in range(len(dim_list)):
            conjugate_flag = False
            sym_chr_list = []
            opr_chr_list = []
            
            
            
            for j in operator_name_list:
                sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                
                if not Symmetry_tmp[j]['repr_has_cc']:
                    
                    #print(str(sym_chr_list[-1])+'\\'+str(opr_chr_list[-1]),end='   ')
                    #print(f'{sym_chr_list[-1]:>11.2f}',end='\\')
                   # print(f'{sym_chr_list[-1]:>11.2f}',end='   ')
                    #print(f'{opr_chr_list[-1]:>11.2f}',end='   ')
                    
                    if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                        if abs(np.conjugate(sym_chr_list[-1])-opr_chr_list[-1]) < 1e-2:
                            conjugate_flag = True
                        
                #if Symmetry_tmp[j]['repr_has_cc']:
                 #   print(f'{sym_chr_list[-1]:>11.2f}',end='K   ')
                    #print(f'{opr_chr_list[-1]:>11.2f}',end='K ')
                    
                    
                
            if conjugate_flag:
                
                for j in operator_name_list:
                    Symmetry_tmp[j]['band_repr_matrix_list'][i] = sp.conjugate(Symmetry_tmp[j]['band_repr_matrix_list'][i])
                
                
                
            dimension += dim_list[i]
            conjugate_record.append(conjugate_flag)
        
        
        
        
        dimension = 0
        print("Low-energy bands:" ,band_interest_set)
        print("Energies of these bands:" ,eigen_energy_array[band_interest_set-1])
        print("Number of low-energy bands:",sum(dim_list))
        print("Number of IrReps:",len(dim_list))
        print("IrReps' dimensions:",dim_list)
        print('''Loading D^std from "mat2kp.in" successfully! ''')
        print("Traces of the matrix representations (D^num) obtained by vasp2mat are given below:")
        
        print("IrRep ",end=' ')
        print("Dim " ,end=' ')
        for i in operator_name_list:
            print(f"{i:>11s}",end='   ')
            
        print()
        ERROR_string = ''
        warning_string_list = []
        
        for i in range(len(dim_list)):
            
            sym_chr_list = []
            opr_chr_list = []
            warning_i = []
            
            pr_str = "{:>5d}".format(i+1)
            #print(f"{i+1:>5d}",end='')
            print(pr_str.replace("j","i"),end='')
            
            pr_str = "{:>5d}".format(dim_list[i])
            #print(f"{dim_list[i]:>5d}",end='  ')
            print(pr_str.replace("j","i"),end='  ')
            
            for j in operator_name_list:
                sym_chr_list.append(np.trace(np.array(Symmetry_tmp[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                opr_chr_list.append(np.trace(operator[j][dimension:dimension+dim_list[i],dimension:dimension+dim_list[i]]))
                
                if not Symmetry_tmp[j]['repr_has_cc']:
                    
                    if abs(sym_chr_list[-1]-opr_chr_list[-1]) > 1e-2:
                        if abs(sym_chr_list[-1]+opr_chr_list[-1]) > 1e-2:
                        #raise CoRepresentationInequivalenceERROR("ERROR for representation " + str(i+1) + " of operator " + j)
                            ERROR_string = "ERROR: The condition trace(D^std(" + j+"))=trace(D^num("+j+")) is not satisfied in IrRep "+str(i+1)+" !!!"
                            print("      ERROR",end = '')    
                            ERROR_string2 = "trace(D^std(" + j+"))=" + "{:>11.2f}".format(np.trace(np.array(Symmetry_copy[j]['band_repr_matrix_list'][i],dtype=np.complex128)))
                            ERROR_string2 += ",    trace(D^num(" + j+"))=" + "{:>11.2f}".format(opr_chr_list[-1])
                            break
                        else:
                            Symmetry_tmp[j]['band_repr_matrix_list'][i] = - Symmetry_tmp[j]['band_repr_matrix_list'][i]
                            
                            warning_string = "Warning: -D^std("+j+") is used in IrRep "+str(i+1)+"!!!"
                            warning_i.append(warning_string)
                            
                    pr_str = "{:>11.2f}".format(opr_chr_list[-1])
                    print(pr_str.replace('j','i'),end='   ',flush=True)
                else:
                    pr_str = f"{opr_chr_list[-1]:>11.2f}"
                    print(pr_str.replace('j','i'),end=' K  ')
            
            warning_string_list.append(warning_i)
            dimension += dim_list[i]
            
            if len(ERROR_string)!=0:
                print('''\nNOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                
                print(ERROR_string+"    "+ERROR_string2.replace('j', 'i')+'.')
                
                
                #print("Please rewrite the standard representation with correct traces! This could be caused by wrong std matrice in mat2kp.in or wrong matrices obtained by vasp2mat! Please check if the WAVECAR are not empty when run vasp2mat!")
                print('''Please check: 1. Whether the standard representation matrices in "mat2kp.in" are correct.''')
                print('''              2. Whether WAVECAR is not empty and LWAVE=.TRUE. is set in INCAR when running vasp2mat.''')
                print('''              3. Whether rot_n and rot_tau in "INCAR.mat" in INCAR.mat are correct when running vasp2mat.''')
                sys.exit()
                
            print()
            
        for i in operator_name_list:
            if Symmetry[i]['repr_has_cc']:
                print("K is the complex conjugation for anti-unitary operations.")
                break
        for j in operator_name_list:
            Symmetry_tmp[j]['repr_matrix'] = bandreplist2rep(Symmetry_tmp[j]['band_repr_matrix_list'])
        
        exe = 0
        for i in range(len(dim_list)):
            if conjugate_record[i]:
                if exe == 0:
                    print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                    exe += 1
                #print("Warning: To satisfy the character condition, Irrep "+str(i+1)+" is conjugated!!!")
                print("Warning: standard IrRep "+str(i+1)+" is conjugated!!!")
                
            for j in warning_string_list[i]:
                if exe == 0:
                    print('''NOTE: Please make sure that traces of the given D^std in "mat2kp.in" are consistent with the above ones!!!''')
                    exe += 1
                print(j)
        
        for i in operator_name_list:
            Symmetry[i] = Symmetry_tmp[i]
        #Symmetry = Symmetry_tmp.copy()
        #print(Symmetry)
        print('''Loading D^num from "MAT_*.m" successfully! ''')
    
    
    
    
"""   