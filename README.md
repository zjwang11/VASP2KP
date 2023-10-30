# VASP2KP
compute kp parameters and Land ́e g-factors directly from the VASP wavefunctions. 

Here, we develop an open-source package VASP2KP
(including two parts: vasp-kp and vasp2kp.py) to calculate these parameters and Land ́e g-factors
directly from the wavefunctions provided by the density functional theory (DFT) as implemented in
Vienna ab initio Simulation Package (VASP). 

First, we develop a patch vasp-kp (for VASP.5.3.3)
to compute matrix presentations of the generalized momentum operator πˆ, spin operator sˆ, time reversal operator Tˆ and crytalline symmetry operators Rˆ on the DFT wavefunctions. 

Second, we develop a python code vasp2kp.py to obtain the unitary transformation U that rotates the DFT basis towards the standard basis, and then automatically compute the param- eters and g-factors. 
