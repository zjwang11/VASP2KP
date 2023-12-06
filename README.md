# *VASP2KP* 
This package is used to automatically generate the $k \cdot p$ Hamiltonians as well as the Zeeman's coupling and calculate the values of all the parameters based on the *VASP* calculations. 

Here, we develop an open-source package VASP2KP (including two parts: "vasp2mat" and "mat2kp") to calculate these parameters and Land ́e g-factors directly from the wavefunctions provided by the density functional theory (DFT) as implemented in Vienna ab initio Simulation Package (VASP). 

First, we develop a patch "vasp2mat"
to compute matrix presentations of the generalized momentum operator πˆ, spin operator sˆ, time reversal operator Tˆ and crytalline symmetry operators Rˆ on the DFT wavefunctions. The matrix elements of the operators are derived comprehensively and computed correctly within the Projector Augmented Wave method.

Second, we develop a python code "mat2kp" to obtain the unitary transformation U that rotates the DFT basis towards the standard basis, and then automatically compute the parameters and Land ́e g-factors. 

To get the detailed information about this package, please visit the website www.vasp2kp.com.