&vmat_para
    !   
    ! operator-------------------------------------
    !
    vmat = 10
    vmat_name = 'sig'
    vmat_k = 1
    bstart=25,  bend=26
    ! vmat_bands(:) = 11, 12, 17, 18  ! s_up s_down p_up p_down
    print_only_diagnal = .false.
    !
    ! soc------------------------------------------
    !
    cfactor=1.0
    socfactor=1.0
    nosoc_inH = .false.
    !
    ! rotation-------------------------------------
    !
    rot_n(:) = 0  1  0
    rot_alpha = 0
    rot_det = 1
    rot_tau(:) = 0.000000     0.000000    0.000000
    rot_spin2pi = .false.
    time_rev = .false.
/
