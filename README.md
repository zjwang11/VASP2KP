# VASP2KP
compute kp parameters and Land ́e g-factors directly from the VASP wavefunctions. 

Here, we develop an open-source package VASP2KP
(including two parts: "vasp2mat" and "mat2kp") to calculate these parameters and Land ́e g-factors
directly from the wavefunctions provided by the density functional theory (DFT) as implemented in
Vienna ab initio Simulation Package (VASP). 

First, we develop a patch "vasp2mat"
to compute matrix presentations of the generalized momentum operator πˆ, spin operator sˆ, time reversal operator Tˆ and crytalline symmetry operators Rˆ on the DFT wavefunctions. The matrix elements of the operators are derived comprehensively and computed correctly within the Projector Augmented Wave method.

Second, we develop a python code "mat2kp" to obtain the unitary transformation U that rotates the DFT basis towards the standard basis, and then automatically compute the parameters and g-factors. 
