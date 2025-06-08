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
#import numba

###############################################################################
class EIGENVALNotFindError(Exception):
    '''
    if the file 'EIGENVAL' not exists: raise this error
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
def load_data(Symmetry,vaspMAT,gfactor=1,data_name_list=["Pi","sig"]):
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
    
    operator_name_list = Symmetry.keys()
    operator = get_opertors_or_data(folder_path,operator_name_list)
    
    if gfactor == 0:
        data_name_list=["Pi"]
        
    data = get_opertors_or_data(folder_path,data_name_list)
    
    band_interest_set = get_interest_band_range(os.path.join(folder_path,"MAT_"+list(operator_name_list)[0]+'.m'))
    #print(band_start,band_end)
    
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
                
                raise EIGENVALNotFindError("Cannot find the file 'EIGENVAL'!")
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
    
    skipline = (num_k - 1)*(num_band + 2) + 2
    
    for i in range(skipline):
        file.readline()

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
    
    # unify dimension, scale parameters
    band_start_all,band_end_all = get_bstart_bend(folder_path)
    
    #band_start -= band_start_all - 1
    #band_end -= band_start_all - 1
    band_interest_set -= band_start_all - 1
    #print(band_start)
    #print(band_end)
    #print(band_start_all - eigen_start)
    eigen_energy = eigen_energy[band_start_all - eigen_start:]
    #print(eigen_energy[0])
    
    return operator,data,eigen_energy,band_interest_set
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
    
    
    
