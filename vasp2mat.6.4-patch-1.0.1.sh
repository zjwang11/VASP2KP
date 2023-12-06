#!/bin/bash

############################################################################################################
# Patch Instructions
# vasp2mat.6.4-patch-1.0.1.sh
# created by Sheng Zhang, Institute of Physics, Chinese Academy of Sciences
# compile the script vasp2mat.6.4
# step 1: copy the folder vasp.6.4 in the same folder as this script
# step 2: create and revise makefile in vasp.6.4 so that vasp_ncl can be compiled successfully (no need to compile)
# step 3: run this patch: bash vasp2mat.6.4-patch-1.0.1.sh
# After this steps, wait a few miniutes and then if "Finishing installing vasp2mat!" is shown on the shell
# then the compile of vasp2mat is compiled successfully.
############################################################################################################

# begin
set -e

# content in song_data.F
song_data_F=$(cat <<EOF
module song_data
    !
    use prec
    use base
    !
    implicit none
    !
    ! control variables =========================================
    !
    integer :: vmat
    character(len=10) :: vmat_name
    integer :: vmat_k, vmat_nbands
    integer :: vmat_bands(1000), bstart, bend
    !
    ! soc 
    real(q) cfactor, socfactor
    logical :: nosoc_inH
    !
    ! rotation
    real(q) :: rot_n(3), rot_alpha, rot_det, rot_tau(3)
    logical :: rot_spin2pi, time_rev
    !
    ! print
    !
    logical :: print_only_diagnal
    !
    ! input car
    namelist /vmat_para/ cfactor, vmat, vmat_k, vmat_bands, socfactor, nosoc_inH, &
                    vmat_name, bstart, bend, &
                    rot_n, rot_alpha, rot_det, rot_tau, rot_spin2pi, time_rev, print_only_diagnal
    !
    ! DFT data =================================================
    !
    real(q), allocatable :: POTAE_all(:,:,:,:)
    !
 contains
    !
    subroutine song_getinp(IO)
        !
        type(in_struct) IO
        integer stat, ii
        !
        ! initialize---------------------------
        !
        cfactor = 1.0_q
        socfactor = 1.0_q
        vmat = 0                ! calculate nothing
        vmat_name = ''
        vmat_k = 1
        vmat_bands = 0
        bstart=0; bend=0
        nosoc_inH = .false.
        !
        rot_n = (/ 1.0_q,0.0_q,0.0_q /)
        rot_alpha = 0.0_q
        rot_det = 1.0_q
        rot_tau = (/ 0.0_q, 0.0_q, 0.0_q /)
        time_rev = .false.
        !
        print_only_diagnal = .false.
        !
        ! read in------------------------------
        !
        open(unit=201, file='INCAR.mat', status='old', iostat=stat)
        if (stat/=0 .and. IO%IU0>=0) then
            write(IO%IU0,*) 'Can not open file INCAR.mat !!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            stop
        endif
        !
        read(201, vmat_para ,iostat=stat)
        if (stat/=0 .and. IO%IU0>=0) then
            write(IO%IU0,*) 'Can not read "vmat_para" list in file INCAR.mat !!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            stop
        endif
        !
        close(201)
        !
        if(bend>=bstart .and. bstart>0) then
            do ii= bstart, bend
                vmat_bands(ii-bstart+1) = ii
            enddo
            vmat_nbands = bend-bstart+1
        else
            vmat_nbands=0
            do ii=1,100
                if (vmat_bands(ii)>0) vmat_nbands=vmat_nbands+1
            enddo
        endif
        !
    endsubroutine song_getinp

endmodule song_data

EOF
)

# content in song_vmat.F
song_vmat_F=$(cat <<EOF
module song_vmat
    !
    use prec
    use base
    use mkpoints
    use wave
    use mgrid
    use lattice,            only : latt
    use nonlr_struct_def
    use nonl_struct_def
    use nonlr
    use nonl
    use nonl_high,          only : W1_PROJ
    use asa,                only : SETYLM_NABLA_YLM, SETYLM_XX_YLM
    use radial,             only : GRAD
    use pseudo,             only : potcar, PP_POINTER
    use poscar,             only : type_info
    use song_data,          only : vmat, vmat_k, vmat_bands, vmat_nbands, &
                                   POTAE_all, nosoc_inH, vmat_name, &
                                   rot_n, &  ! for rotation
                                   rot_alpha, &
                                   rot_det, rot_tau, rot_spin2pi, time_rev, &
                                   print_only_diagnal
    !
    !
    implicit none
    !
    public :: vmat_top
    private
    !
    !type (wavefun1) :: W1, W2
    !type (wavedes1) :: WDES1
    !complex(q), allocatable, target :: cwork1(:), cwork2(:)
    !
    REAL(q),PARAMETER  :: PI =3.141592653589793238_q, TPI=2*PI
    REAL(q), PARAMETER :: AUTOA=0.529177249_q, RYTOEV=13.605826_q, HATOEV = 2* RYTOEV
    complex(q), parameter :: imag=cmplx(0.0_q, 1.0_q, kind=q)
    complex(q), dimension(0:1, 0:1) :: sig0, sig1, sig2, sig3
    integer :: maxng
    !
contains

subroutine vmat_top(IO, T_INFO, LATT_CUR, KPOINTS, WDES, GRID, W, LMDIM, &
                    CDIJ, CQIJ, SV, P, NONLR_S, NONL_S)
    !
    use main_mpi, only : comm_world
    !
    type(in_struct) IO
    TYPE (type_info) T_INFO
    type(latt) :: LATT_CUR
    type(kpoints_struct) KPOINTS
    type(wavedes) :: WDES
    type(grid_3d) :: GRID
    type(wavespin) W
    integer LMDIM
    complex(q) CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    complex(q) CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    complex(q) :: SV(GRID%MPLWV,WDES%NCDIJ)
    TYPE(potcar), TARGET :: P(T_INFO%NTYP)
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    !
    complex(q), allocatable :: matrix(:,:,:), matrix_(:,:,:)
    !
    real(q) :: kp(3)
    !
    integer :: ii,wzjk
    character(len=40) :: str
    !
    ! initialize pauli matrix==============================================
    ! 
    call initial_pauli(WDES%SAXIS)
    !
    ! check ===============================================================
    !
     maxng= maxval(WDES%NRSPINORS*WDES%NGVECTOR(:))        ! in format of k1 wavefunction

    if ( IO%IU0>=0 ) then
        if( comm_world%ncpu /=1 )    then
            write(IO%IU0, *) 'Error: vmat can not run parallelly for now !!!!!!'
            stop
        endif
        if(vmat_k > KPOINTS%NKPTS)   then
            write(IO%IU0, *) "Error: vmat_k > NKPTS !!!"
            stop
        endif
        if ( maxval(vmat_bands) > WDES%NB_TOT) then
            write(IO%IU0, *) "Error: bands > nbands !!!"
            stop
        endif
        if ( minval(vmat_bands(1:vmat_nbands)) <=0 ) then
            write(IO%IU0, *) "Error: bands <= 0 !!!"
            stop
        endif
        if ( vmat==14 .and. print_only_diagnal==.false. ) then
            write(IO%IU0, *) "Error: Please set print_only_diagnal=.true. for wilson loop calculation"
            write(IO%IU0, *) "       Because we only print eigenvalues of W matrix."
        endif
    endif
    !
    ! output file header
    !
    if ( IO%IU6>=0 ) then
        !
        str='MAT_'//trim(adjustl(vmat_name))//'.m'
        OPEN(unit=91,file=str,status='replace')
        !
        write(91, '("%                            vmat                              ")')
        write(91, '("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")')
        !
        do ii=1,3
           kp(ii)= (KPOINTS%VKPT(1,vmat_k)*LATT_CUR%B(ii,1)+ &
                    KPOINTS%VKPT(2,vmat_k)*LATT_CUR%B(ii,2)+ &
                    KPOINTS%VKPT(3,vmat_k)*LATT_CUR%B(ii,3) ) *TPI*AUTOA
        enddo
        !
        write(91, '("%    k     =",3F12.8, "  in b_i")') KPOINTS%VKPT(1:3,vmat_k)
        write(91, '("%    B1    =",3F12.8, "  in bohr^-1")') LATT_CUR%B(1:3,1)* TPI*AUTOA
        write(91, '("%    B2    =",3F12.8, "  in bohr^-1")') LATT_CUR%B(1:3,2)* TPI*AUTOA
        write(91, '("%    B3    =",3F12.8, "  in bohr^-1")') LATT_CUR%B(1:3,3)* TPI*AUTOA
        !
        if(vmat==14) then
            write(91, '("% Number of k-points along the wilson loop: " , I8 )') KPOINTS%NKPTS
        else
            write(91, '("%    k     =",3F12.8, "  in bohr^-1")') kp(1:3)
        endif
        !
        write(91, '("% On bands :",8I5)') ( vmat_bands(1:vmat_nbands))
        !
    endif
    !
    ! calculation ========================================================
    !
    allocate( matrix(vmat_nbands, vmat_nbands, WDES%ISPIN) )

    matrix = cmplx( 0.0_q, 0.0_q, kind=q) 
    !
    select case (vmat)
    case(1)     ! Overlap
        !
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi_i|psi_j> in eV', &
                           trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(2)     ! soft local potential
        !  
        call vmat_vlocal_pseudo_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, GRID, SV)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, &
                            '<\tilde{psi}| \tilde{V}_eff |\tilde{psi}> in eV', &
                            trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(3)     ! kinetic energy of pseudowave
        !
        call vmat_kinetic_pseudo_matrix(matrix, vmat_nbands, vmat_bands, LATT_CUR%B, WDES, W)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, &
                            '<\tilde{psi}| T |\tilde{psi}> in eV', &
                            trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(4)     ! nonlocal potential
        !
        call vmat_nlpot_pseudo_matrix( matrix, vmat_nbands, vmat_bands, WDES, W, CDIJ, LMDIM)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, &
                            '<\tilde{psi}| V_NL |\tilde{psi}> in eV', &
                            trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(5)     ! hamiltonian
        !
        allocate( matrix_(vmat_nbands, vmat_nbands, WDES%ISPIN) )
        matrix_ = 0.0_q
        !
        call vmat_vlocal_pseudo_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, GRID, SV)
        !
        call vmat_kinetic_pseudo_matrix(matrix_, vmat_nbands, vmat_bands, LATT_CUR%B, WDES, W)
        matrix = matrix + matrix_
        !
        call vmat_nlpot_pseudo_matrix( matrix_, vmat_nbands, vmat_bands, WDES, W, CDIJ, LMDIM)
        matrix = matrix + matrix_
        !
        IF(IO%IU6>=0) THEN
            if(nosoc_inH) then
                call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| H-Hso |psi> in eV', &
                    trim(adjustl(vmat_name)), WDES%ISPIN)
            else
                call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| H |psi> in eV', &
                    trim(adjustl(vmat_name)), WDES%ISPIN)
            endif
        ENDIF
        !
        deallocate(matrix_)
        !
    case(7)     ! moment of wavefunction
        !
        call vmat_p_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 1)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| px |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'x', WDES%ISPIN)
        !
        call vmat_p_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 2)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| py |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'y', WDES%ISPIN)
        !
        call vmat_p_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 3)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| pz |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'z', WDES%ISPIN)
        !
    case(8)     ! Hso
        !
        if (IO%IU0>=0 .and. WDES%NCDIJ/=4) then
            write(IO%IU0,*) 'Error: Hso can only be computed in non-collinear case'
            stop
        endif
        !
        call vmat_hso_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LMDIM, T_INFO, P)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| Hso |psi> in eV', &
                        trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(10)    ! \sigma,  i.e. pauli matrix
        !
        IF(WDES%NCDIJ/=4) THEN
            if(IO%IU0>=0) write(IO%IU0, *) 'Error: pauli matrix can only be computed in non-collinear case'
            stop
        ENDIF
        !
        ! x 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig1)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| sigma_x |psi>', &
                        trim(adjustl(vmat_name))//'x', WDES%ISPIN)
        !
        ! y 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig2)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| sigma_y |psi>', &
                        trim(adjustl(vmat_name))//'y', WDES%ISPIN)
        !
        ! z 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig3)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| sigma_z |psi>', &
                        trim(adjustl(vmat_name))//'z', WDES%ISPIN)
        !
    case(11)    ! pi = p + \alpha^2/4 * sigma x \nabla V
        !
        IF(WDES%NCDIJ/=4) THEN
            if(IO%IU0>=0) write(IO%IU0, *) 'Error: pi can only be computed in non-collinear case'
            stop
        ENDIF
        !
        ! x
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 1)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| pix |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'x', WDES%ISPIN)
        !
        ! y 
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 2)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| piy |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'y', WDES%ISPIN)
        !
        ! z
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 3)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| piy |psi> in hbar/bohr', &
                        trim(adjustl(vmat_name))//'z', WDES%ISPIN)
        !
    case(12)    ! rotate wavefunctions
        !
        call vmat_rot_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, NONLR_S, NONL_S, GRID, LMDIM, CQIJ, LATT_CUR, KPOINTS, &
                             IO, rot_n, rot_alpha, rot_det, rot_tau, rot_spin2pi, time_rev)
        !
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, '<psi| Rot |psi> ', &
                        trim(adjustl(vmat_name)), WDES%ISPIN)
    !case(13)    ! L = r \times p,  in \hbar
    case(14)    ! wilson loop
        !
        call vmat_wilson(matrix, KPOINTS%NKPTS, vmat_nbands, vmat_bands, WDES, W, NONLR_S, NONL_S, GRID, CQIJ, LATT_CUR, KPOINTS)
        IF(IO%IU6>=0) call vmat_print_maxtrix(matrix, vmat_nbands, " Berry's phase in 2pi", &
                           trim(adjustl(vmat_name)), WDES%ISPIN)
        !
    case(13)    ! \sigma,  i.e. pauli matrix and ! pi = p + \alpha^2/4 * sigma x \nabla V
        !write(*,"(2I5)") q,vmat_nbands;stop !wzj
        !
        IF(WDES%NCDIJ/=4) THEN
            if(IO%IU0>=0) write(IO%IU0, *) 'Error: pauli matrix can only be computed in non-collinear case'
            stop
        ENDIF
        !
        if( vmat_k.gt. KPOINTS%NKPTS ) STOP "Wrong: vmat_k"
        write(91, '("% Total k-number:",8I5)')  vmat_k
        write(91, '("% Eigenvalues         are output in fort.1215 in units of eV")')
        write(91, '("% Sigma Matrices(MAT) are output in fort.1216 in units of hbar/bohr")')
        write(91, '("% Velocity (Pi) MAT   are output in fort.1217 in units of hbar/bohr")')
        !
        write(1215,'(2I10)') vmat_nbands,vmat_k!KPOINTS%NKPTS-1
        write(1216,'(2I10)') vmat_nbands,vmat_k!KPOINTS%NKPTS-1
        write(1217,'(2I10)') vmat_nbands,vmat_k!KPOINTS%NKPTS-1
        write(   *,'(2I10)') vmat_nbands,vmat_k!KPOINTS%NKPTS-1
                                  wzjk = vmat_k
        do vmat_k=1,wzjk
        write(   *,'(A,I5)') 'k=',vmat_k
       !write(1215,"(I5,80000E14.6)") vmat_k,(REAL( W%CELTOT(ii,vmat_k,1), KIND=q), ii=1,vmat_nbands)
        write(1215,"(I5,80000E14.6)") vmat_k,(REAL( W%CELTOT(vmat_bands(ii),vmat_k,1), KIND=q),ii=1,vmat_nbands)
        write(1216,'(A,I5)') 'k=',vmat_k
        ! x 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig1)
        write(1216,"(8E14.6)") matrix(:,:,1) !wzj
        !
        ! y 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig2)
        write(1216,"(8E14.6)") matrix(:,:,1) !wzj
        !
        ! z 
        call vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, sig3)
        write(1216,"(8E14.6)") matrix(:,:,1) !wzj
        !
        write(1217,'(A,I5)') 'k=',vmat_k
        ! x
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 1)
        write(1217,"(8E14.6)") matrix(:,:,1) !wzj
        !
        ! y 
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 2)
        write(1217,"(8E14.6)") matrix(:,:,1) !wzj
        !
        ! z
        call vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LATT_CUR%B, T_INFO, P, LMDIM, 3)
        write(1217,"(8E14.6)") matrix(:,:,1) !wzj
        enddo
        !
    endselect
    !
    deallocate( matrix )
    !
endsubroutine vmat_top

subroutine initial_pauli(SAXIS)
    !
    real(q) :: SAXIS(3)
    real(q) :: alpha, beta
    !
    integer :: ii, jj, kk, kkp
    complex(q), dimension(0:1,0:1) :: tmp, rot
    !
    ! initialize===========================================================
    !
    sig0(:,:) = cmplx( 0.0_q, 0.0_q, kind=q)
    sig1(:,:) = cmplx( 0.0_q, 0.0_q, kind=q)
    sig2(:,:) = cmplx( 0.0_q, 0.0_q, kind=q)
    sig3(:,:) = cmplx( 0.0_q, 0.0_q, kind=q)
    !
    sig0(0,0) = 1.0_q
    sig0(1,1) = 1.0_q
    sig1(0,1) = 1.0_q
    sig1(1,0) = 1.0_q
    sig2(0,1) = -imag
    sig2(1,0) = imag
    sig3(0,0) = 1.0_q
    sig3(1,1) = -1.0_q
    !
    ! rotating=============================================================
    !
    CALL EULER( SAXIS, alpha, beta )
    !
    rot(0,0)= COS(beta/2)*EXP(-(0._q,1._q)*alpha/2)
    rot(0,1)=-SIN(beta/2)*EXP(-(0._q,1._q)*alpha/2)
    rot(1,0)= SIN(beta/2)*EXP( (0._q,1._q)*alpha/2)
    rot(1,1)= COS(beta/2)*EXP( (0._q,1._q)*alpha/2)
    !
    ! sig1------------------------------------
    !
    tmp = 0.0_q
    do ii=0,1
    do jj=0,1
        do kk=0,1
        do kkp=0,1
            tmp(ii,jj) = tmp(ii,jj) + conjg(rot(kk,ii))*sig1(kk,kkp)*rot(kkp,jj) 
        enddo
        enddo
    enddo
    enddo
    sig1 = tmp 
    !
    ! sig2------------------------------------
    !
    tmp = 0.0_q
    do ii=0,1
    do jj=0,1
        do kk=0,1
        do kkp=0,1
            tmp(ii,jj) = tmp(ii,jj) + conjg(rot(kk,ii))*sig2(kk,kkp)*rot(kkp,jj) 
        enddo
        enddo
    enddo
    enddo
    sig2 = tmp 
    !
    ! sig3------------------------------------
    !
    tmp = 0.0_q
    do ii=0,1
    do jj=0,1
        do kk=0,1
        do kkp=0,1
            tmp(ii,jj) = tmp(ii,jj) + conjg(rot(kk,ii))*sig3(kk,kkp)*rot(kkp,jj) 
        enddo
        enddo
    enddo
    enddo
    sig3 = tmp 
    !
endsubroutine initial_pauli

subroutine vmat_print_maxtrix(matrix, vmat_nbands, vnote, vname, ISPIN)
    !
    integer :: vmat_nbands
    integer :: ISPIN
    complex(q) :: matrix(vmat_nbands,vmat_nbands,ISPIN)
    character(len=*) :: vnote, vname
    !
    integer :: ii, jj, sp
    character(len=40) :: str
    real(q) :: tmp
    !
    !
    str = ''
    write(str,*) len(trim(adjustl(vnote)))
    str = '("% ", A'//trim(adjustl(str))//')'
    write(91,str)  trim(adjustl(vnote))
    !
    do sp=1,ISPIN
        !
        ! make '+' symbol before zero, this makes convenience for matlab post-processing 
        !
        do ii=1,vmat_nbands
        do jj=1,vmat_nbands
            matrix(ii,jj,sp) = matrix(ii,jj,sp) + cmplx(1.0d-20,1.0d-20,kind=q)
        enddo
        enddo
        !
        ! print
        !
        if ( .not. print_only_diagnal ) then
        !
            DO jj=1, vmat_nbands, 4
                !
                if( ISPIN==1) then
                    write(91, '(A10, "( 1:", I5,  ",", I5, ":", I5, ")=[")') &
                        adjustl(vname), vmat_nbands, jj, min(jj+3,vmat_nbands)
                elseif( ISPIN==2 ) then
                    !
                    if(sp==1) write(91, '(A12, "( 1:", I5,  ",", I5, ":", I5, ")=[")') &
                        trim(adjustl(vname))//"_up", vmat_nbands, jj, min(jj+3,vmat_nbands)
                    !
                    if(sp==2) write(91, '(A12, "( 1:", I5,  ",", I5, ":", I5, ")=[")') &
                        trim(adjustl(vname))//"_dw", vmat_nbands, jj, min(jj+3,vmat_nbands)
                    !
                endif
                !
                !
                str = ''
                write(str,'(A4,I2,A23)') '(SP,', min(jj+3,vmat_nbands)-jj+1, '( "(",2E14.6,"i) " ) )'
                DO ii=1, vmat_nbands
                    write(91, str) matrix(ii,jj: min(jj+3,vmat_nbands), sp)
                ENDDO
                write(91,'("];")')
                !
            ENDDO
        else
            if( ISPIN==1) then
                write(91, '(A10, "( 1:", I5, ", 1:2)=[")') &
                    adjustl(vname), vmat_nbands
            elseif( ISPIN==2 ) then
                !
                if(sp==1)  write(91, '(A10, "( 1:", I5, ", 1:2)=[")') &
                    adjustl(vname)//"_up", vmat_nbands
                !
                if(sp==1)  write(91, '(A10, "( 1:", I5, ", 1:2)=[")') &
                    adjustl(vname)//"_dw", vmat_nbands
                !
            endif
            !
            DO ii=1, vmat_nbands
                write(91, '(T4, I5, SP,"   (", 2F10.5, "i)"  )') vmat_bands(ii), matrix(ii,ii, sp)
            ENDDO
            !
            write(91,'("];")')
            !
        endif
        !
        ! trace
        tmp=0.0_q
        do jj=1,vmat_nbands
            tmp = tmp + real(matrix(jj,jj, sp),q)
        enddo
        tmp=tmp-floor(tmp) !zjwang
        !
        write(91,'(SP  ,  "% trace = ("  ,  2F12.5  ,  "i)" )') tmp,0._q
        !
    enddo
    !
endsubroutine vmat_print_maxtrix

!==========================================================
! this subroutine calculates the wilson loop
!==========================================================
subroutine vmat_wilson(matrix, NKPTS, vmat_nbands, vmat_bands, WDES, W, NONLR_S, NONL_S, GRID, CQIJ, LATT_CUR, KPOINTS)
    !
    ! input variables
    !
    type(wavedes) :: WDES
    integer :: NKPTS, vmat_nbands
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    type(grid_3d) :: GRID
    complex(q) :: CQIJ(:,:,:,:)
    type(latt) :: LATT_CUR
    type(kpoints_struct) KPOINTS
    !
    ! data dictionary
    !
    integer :: ii, jj, sp, nk, nkn,idk(3)
    complex(q) :: Wmat(vmat_nbands,vmat_nbands), Mmat(vmat_nbands,vmat_nbands), &
                  Mmat_(vmat_nbands,vmat_nbands), vs(vmat_nbands,vmat_nbands)
    complex(q) :: eig(vmat_nbands), phase(vmat_nbands)
    !
    ! used for lapack routines
    !
    integer :: lwork, sdim, info
    complex(q), allocatable,  dimension(:) :: work
    real(q), allocatable, dimension(:) :: rwork
    logical, allocatable, dimension(:) :: bwork
    !
    !wzj
    integer :: ng2(maxng)
    logical :: tf1(maxng)
    complex(q), target :: cpt(WDES%NRSPINORS*maxng)
    !
    ng2=0;tf1=.false.
    spin: do sp=1, WDES%ISPIN
        !    
        ! generate W matrix
        !
        do nk=1, NKPTS
            !
            nkn=nk+1
            if (nkn>NKPTS) nkn=1
            !
            idk(:) = nint(KPOINTS%VKPT(1:3,nkn) - KPOINTS%VKPT(1:3,nk))
            !

            do ii=1,vmat_nbands
            do jj=1,vmat_nbands
          CALL vmat_woverlap( cc=Mmat(ii,jj), b1=vmat_bands(ii), b2=vmat_bands(jj), &
               k1=nk, k2=nkn, dk_in=idk, ng2=ng2, tf1=tf1, cpt=cpt, WDES=WDES, W=W, sp=sp, NONLR_S=NONLR_S, &
               NONL_S=NONL_S, GRID=GRID, CQIJ=CQIJ, LATT_CUR=LATT_CUR, KPOINTS=KPOINTS )
            enddo
            enddo
            ! debug zjwang
            write(547,"(5I5)") nk,nkn,idk
            do ii=1,vmat_nbands
            do jj=1,vmat_nbands
            write(547,"(3F18.9)") Mmat(ii,jj),abs(Mmat(ii,jj))
            enddo
            enddo
            !
            if (nk==1) then
                Wmat = Mmat
            else
                if( vmat_nbands>1 ) then
                    Mmat_ = Wmat
                    call zgemm( 'N', 'N', vmat_nbands, vmat_nbands, vmat_nbands, cmplx(1.0_q,0.0_q,kind=q), &
                                Mmat_, vmat_nbands, Mmat, vmat_nbands, cmplx(0.0_q,0.0_q,kind=q), Wmat, vmat_nbands)
                   !Wmat=matmul(Wmat,Mmat)
                elseif ( vmat_nbands==1) then
                    Wmat(1,1) = Wmat(1,1)*Mmat(1,1)
                endif
            endif
            !
        enddo
        !write(*,"(8F10.4)") Wmat(:,:)
        !Print*, info;stop
        !
        ! solve eigenvalues of W matrix
        !
        if (vmat_nbands>1) then
            lwork = 3*vmat_nbands
            allocate( work(lwork), rwork(lwork), bwork(lwork) )
            !
            call zgees('n', 'n', 'n', &                                   ! obvs, sort, select,
                        vmat_nbands, Wmat, vmat_nbands, sdim, eig, &      ! n, a, lda, sdim, w,
                        vs, vmat_nbands, work, lwork, rwork, bwork, &     ! vs, ldvs, work, lwork, rwork, bwork
                        info )
            !
            deallocate(work, rwork, bwork )
        elseif (vmat_nbands==1) then
            eig(1) = Wmat(1,1)
        endif
        !
        ! get the phase
        !
        matrix(:,:,sp) = cmplx(0.0_q,0.0_q,kind=q)
        do ii=1, vmat_nbands
           !matrix(ii,ii,sp) = acos( real(eig(ii),kind=q)/abs(eig(ii)) ) / PI
            matrix(ii,ii,sp) = cmplx( acos( real(eig(ii),kind=q)/abs(eig(ii)) )*0.5_q/PI &
                                      ,abs(eig(ii)), kind=q )
            ! now matrix(ii,ii) \in [0,pi]

            if ( aimag(eig(ii)) < 0 ) then ! matrix(ii,ii,sp) = -matrix(ii,ii,sp)
            matrix(ii,ii,sp) = cmplx(-acos( real(eig(ii),kind=q)/abs(eig(ii)) )*0.5_q/PI &
                                      ,abs(eig(ii)), kind=q )
            endif
           ! % added by zjwang to be consistent with Amn= i<Um|dk|Un>
           !matrix(ii,ii,sp) = -matrix(ii,ii,sp)*0.5_q
           ! % added by zjwang 
        enddo
        
        !
    enddo spin
    !
endsubroutine

subroutine vmat_overlap_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, CQIJ,LMDIM, GRID, pauli)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    complex(q) :: CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    integer :: LMDIM
    type(grid_3d) :: GRID
    complex(q),optional :: pauli(0:1,0:1)
    !
    integer :: ii, jj, sp
    !    
    if( present(pauli) .and. WDES%NCDIJ/=4 ) then 
        write(*, *) 'Error: pauli matrix can only be computed in non-collinear case'
        stop 
    endif
    !
    do sp=1,WDES%ISPIN
        DO ii=1,vmat_nbands
        DO jj=1, vmat_nbands
            !
            call vmat_overlap( matrix(ii,jj,sp), vmat_k, vmat_bands(ii), vmat_k, vmat_bands(jj), WDES, W, CQIJ(1:,1:,1:,sp:), LMDIM, GRID, sp, pauli)
            !
        ENDDO
        ENDDO
    enddo
    !
endsubroutine vmat_overlap_matrix
!
subroutine vmat_vlocal_pseudo_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, GRID, SV)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    type(grid_3d) :: GRID
    complex(q) :: SV(GRID%MPLWV,WDES%NCDIJ)
    !
    integer :: sp, ii, jj
    !
    do sp=1,WDES%ISPIN
    do ii=1,vmat_nbands
    do jj=1,vmat_nbands
        call vmat_vlocal_pseudo( matrix(ii,jj,sp), vmat_bands(ii), vmat_bands(jj), WDES, W, GRID, SV(1:,sp:), sp )
    enddo
    enddo
    enddo
endsubroutine vmat_vlocal_pseudo_matrix
!
subroutine vmat_kinetic_pseudo_matrix(matrix, vmat_nbands, vmat_bands, B, WDES, W)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    real(q) :: B(3,3)
    !
    integer :: ii, jj, sp
    !
    do sp=1,WDES%ISPIN
    DO ii=1,vmat_nbands
    DO jj=1, vmat_nbands
        !
        call vmat_kinetic_pseudo(matrix(ii,jj,sp),  vmat_bands(ii), vmat_bands(jj), B, WDES, W, sp)
        !
    ENDDO
    ENDDO
    enddo
    !
endsubroutine vmat_kinetic_pseudo_matrix
!
subroutine vmat_nlpot_pseudo_matrix( matrix, vmat_nbands, vmat_bands, WDES, W, CDIJ, LMDIM)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    integer :: LMDIM
    complex(q) :: CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    !
    integer :: ii, jj, sp
    !
    do sp=1,WDES%ISPIN
    do ii=1,vmat_nbands
    do jj=1,vmat_nbands
        call vmat_nlpot_pseudo( matrix(ii,jj,sp), vmat_bands(ii), vmat_bands(jj), WDES, W, CDIJ(1:,1:,1:,sp:), LMDIM, sp )
    enddo
    enddo
    enddo
    !
endsubroutine vmat_nlpot_pseudo_matrix
!
subroutine vmat_p_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, B, T_INFO, P, LMDIM, ivec)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    real(q) :: B(3,3)
    TYPE (type_info) T_INFO
    type(potcar), target :: P(T_INFO%NTYP)
    integer :: LMDIM, ivec
    !
    ! local variables
    complex(q) :: cc
    complex(q) :: CPIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    integer :: ii, jj, sp
    !
    call SET_CPIJ( CPIJ, LMDIM, P, WDES, T_INFO, ivec)
    !    
    do sp=1,WDES%ISPIN
        DO ii=1,vmat_nbands
        DO jj=1, vmat_nbands
            !    
            call vmat_nlpot_pseudo( matrix(ii,jj,sp), vmat_bands(ii), vmat_bands(jj), WDES, W, CPIJ(1:,1:,1:,sp:),LMDIM, sp)
            call vmat_p_pseudo( cc, vmat_bands(ii), vmat_bands(jj), B, WDES, W, sp, ivec) 
            matrix(ii,jj,sp) = cc + matrix(ii,jj,sp)
            !    
        ENDDO
        ENDDO
    enddo
    !
endsubroutine vmat_p_matrix
!
subroutine vmat_hso_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, LMDIM, T_INFO, P)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    integer :: LMDIM
    TYPE (type_info) T_INFO
    TYPE(potcar), TARGET :: P(T_INFO%NTYP)
    !
    complex(q) :: CSOIJ(LMDIM, LMDIM, WDES%NIONS, 4 )
    !
    CALL SET_CSOIJ( LMDIM, P, CSOIJ, WDES, T_INFO )
    !
    call vmat_nlpot_pseudo_matrix( matrix, vmat_nbands, vmat_bands, WDES, W, CSOIJ, LMDIM)
    !
endsubroutine vmat_hso_matrix
!
subroutine vmat_pi_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, B, T_INFO, P, LMDIM, ivec)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    real(q) :: B(3,3)
    TYPE (type_info) T_INFO
    type(potcar), target :: P(T_INFO%NTYP)
    integer :: LMDIM, ivec
    !
    ! local variables
    !
    complex(q) :: CPIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    integer :: ii, jj, ivec_ 
    complex(q), dimension(0:1, 0:1) :: sig
    complex(q) :: mat_p(vmat_nbands,vmat_nbands, WDES%ISPIN)
    complex(q) :: mat_pp(vmat_nbands,vmat_nbands, WDES%ISPIN)
    !
    ! p -------------------------------------------------------------
    !
    call SET_CPIJ( CPIJ, LMDIM, P, WDES, T_INFO, ivec)
    call vmat_p_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, B, T_INFO, P, LMDIM, ivec)
    !
    ! sig_j * \nabla_k V --------------------------------------------
    !
    if( ivec == 1) then
        sig = sig2
        ivec_ = 3
    elseif ( ivec ==2 ) then
        sig = sig3
        ivec_ = 1
    elseif ( ivec ==3 ) then
        sig = sig1
        ivec_ = 2
    endif
    !
    call SET_CNABLAVIJ( LMDIM, P, CPIJ, WDES, T_INFO, ivec_ , sig)
    !
    call vmat_nlpot_pseudo_matrix( mat_p, vmat_nbands, vmat_bands, WDES, W, CPIJ, LMDIM)
    !
    ! sig_k * \nabla_j V---------------------------------------------
    !
    if( ivec == 1) then
        sig = sig3
        ivec_ = 2
    elseif ( ivec ==2 ) then
        sig = sig1
        ivec_ = 3
    elseif ( ivec ==3 ) then
        sig = sig2
        ivec_ = 1
    endif
    !
    !
    call SET_CNABLAVIJ( LMDIM, P, CPIJ, WDES, T_INFO, ivec_ , sig)
    !
    call vmat_nlpot_pseudo_matrix( mat_pp, vmat_nbands, vmat_bands, WDES, W, CPIJ, LMDIM)
    !
    ! add them ------------------------------------------------------
    !  note that ISPIN=1 in non-collinear case
    !
    DO ii=1,vmat_nbands
    DO jj=1, vmat_nbands
        !
        matrix(ii,jj,1) = matrix(ii,jj,1) + mat_p(ii,jj,1)/4.0_q/137.0_q/137.0_q &
                                      -mat_pp(ii,jj,1)/4.0_q/137.0_q/137.0_q

        !
    ENDDO
    ENDDO
    !
endsubroutine vmat_pi_matrix
!
subroutine vmat_rot_matrix(matrix, vmat_nbands, vmat_bands, WDES, W, NONLR_S, NONL_S, GRID, LMDIM, CQIJ, LATT_CUR, KPOINTS, &
                           IO, rot_n, rot_alpha, rot_det, rot_tau, rot_spin2pi, time_rev)
    !
    integer :: vmat_nbands
    type(wavedes) :: WDES
    complex(q) :: matrix(vmat_nbands,vmat_nbands, WDES%ISPIN)
    integer :: vmat_bands(vmat_nbands)
    type(wavespin) :: W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    type(grid_3d) :: GRID
    integer :: LMDIM
    complex(q) CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    type(latt) :: LATT_CUR
    type(kpoints_struct) KPOINTS
    type(in_struct) :: IO
    real(q) :: rot_n(3), rot_alpha, rot_det, rot_tau(3)
    logical :: rot_spin2pi, time_rev
    !
    ! local variable
    !
    integer :: ii, jj, kk, sp
    real(q) :: phi, kp(3), rot(3,3), rot_direc(3,3), tmpr(3,3)
    real(q) :: lm, mu, nu, lent
    complex(q) :: sig(0:1,0:1), tmpc(0:1,0:1)
    !
    ! real space------------------------------------------------------------------
    !
    lent = sqrt( rot_n(1)**2 + rot_n(2)**2 + rot_n(3)**2 )
    if( lent < 1.0d-4 ) then
        if(IO%IU0>=0) write(IO%IU0, *) 'Error: rot_n(1:3) are too small !!!!'
        stop
    endif
    lm = rot_n(1)/lent
    mu = rot_n(2)/lent
    nu = rot_n(3)/lent
    !
    phi = -rot_alpha/180*PI     ! apply the inversion of given operator to coordinates
    !
    ! R psi(r) = psi(R^-1 r) = psi( rot r )
    rot(1,:) = (/ cos(phi)+lm**2*(1-cos(phi)), lm*mu*(1-cos(phi))-nu*sin(phi), lm*nu*(1-cos(phi))+mu*sin(phi) /)
    rot(2,:) = (/ lm*mu*(1-cos(phi))+nu*sin(phi), cos(phi)+mu**2*(1-cos(phi)), mu*nu*(1-cos(phi))-lm*sin(phi) /)
    rot(3,:) = (/ nu*lm*(1-cos(phi))-mu*sin(phi), mu*nu*(1-cos(phi))+lm*sin(phi), cos(phi)+nu**2*(1-cos(phi)) /)
    !
    rot(:,:) = rot*rot_det
    !
    ! transform to direc coordinates
    ! r_' = rot_dirac r_ 
    tmpr = 0.0_q
    do ii=1,3
    do jj=1,3
        do kk=1,3
            tmpr(ii,jj) = tmpr(ii,jj) + rot(ii,kk)*LATT_CUR%A(kk,jj)
        enddo
    enddo
    enddo
    !
    rot_direc = 0.0_q
    do ii=1,3
    do jj=1,3
        do kk=1,3
            rot_direc(ii,jj) = rot_direc(ii,jj) + LATT_CUR%B(kk,ii)*tmpr(kk,jj)
        enddo
    enddo
    enddo
#if .false.
    write(*,'(3F8.3)') rot(1,:)
    write(*,'(3F8.3)') rot(2,:)
    write(*,'(3F8.3)') rot(3,:)
    write(*,'(3F8.3)') rot_direc(1,:)
    write(*,'(3F8.3)') rot_direc(2,:)
    write(*,'(3F8.3)') rot_direc(3,:)
#endif
    !
    ! apply to k
    kp = 0.0_q
    do ii=1,3
    do jj=1,3
        kp(ii) = kp(ii) + KPOINTS%VKPT(jj,vmat_k)*rot_direc(jj,ii)
    enddo
    enddo
    !
    ! k' - k
    if (time_rev) then
        kp(1:3) = kp(1:3) + KPOINTS%VKPT(1:3,vmat_k)
    else
        kp(1:3) = kp(1:3) - KPOINTS%VKPT(1:3,vmat_k)
    endif
    !
    do ii=1,3
        if ( abs(kp(ii)-real(nint(kp(ii)),kind=q)) .gt. 0.2e-3  ) then
            if(IO%IU0>=0) write(IO%IU0, *) 'Error: k do not have this symmetry !!!!'
            stop
        endif
    enddo
    !
    ! spin space ------------------------------------------------------------------------------------------------
    !
    spinop: IF( WDES%NCDIJ==4 ) THEN
        !
        phi = rot_alpha/180*PI     ! apply the given operator to coordinates
        if( rot_spin2pi ) phi = phi + TPI
        !
        sig(:,:) = cos(phi/2)*sig0(:,:) - imag*sin(phi/2)*( lm*sig1(:,:) + mu*sig2(:,:) + nu*sig3(:,:) )
        !
        ! time reversal 
        !
        if ( time_rev ) then
            !
            tmpc(0:1,0:1) = sig(0:1,0:1)
            sig(0:1,0:1) = 0.0_q
            !
            do ii=0,1
            do jj=0,1
            do kk=0,1
                sig(ii,jj) = sig(ii,jj) - imag * tmpc(ii,kk) * sig2(kk,jj)
            enddo
            enddo
            enddo
        endif
    !
    ! collinear case
    !
    ELSE
        sig(:,:) = 0.0_q
        sig(0,0) = 1.0_q
        sig(1,1) = 1.0_q
    ENDIF spinop
    !
    ! matrix element---------------------------------------------------------------------------------------------
    !
    do sp=1,WDES%ISPIN
    DO ii=1,vmat_nbands
    DO jj=1, vmat_nbands
        !
        call vmat_rotation( cc=matrix(ii,jj,sp), b1=vmat_bands(ii), b2=vmat_bands(jj), k1=vmat_k, k2=vmat_k, dk_in=kp, &
                            rot_in=rot_direc, tau_in=rot_tau, sig_in=sig, rev_in=time_rev, &
                            WDES=WDES, W=W, sp=sp, NONLR_S=NONLR_S, NONL_S=NONL_S, GRID=GRID, CQIJ=CQIJ, &
                            LATT_CUR=LATT_CUR, KPOINTS=KPOINTS )
        !
    ENDDO
    ENDDO
    enddo
    !
    IF(IO%IU6>=0) THEN
           write(91,'("% ")')
           write(91,'("% Rotation :")')
           write(91,'("%            n  = [",  3F10.6,  "] in cart." )') lm, mu, nu
           write(91,'("%    rot_angle  =  ",   F10.6,  " in deg.")') rot_alpha
           write(91,'("%          det  =  ",   F10.6 )') rot_det
           write(91,'("%      rot_tau  = [",  3F10.6,  "] in direc." )') rot_tau(1:3) 
        if( time_rev ) &
           write(91,'("% with time reversal ")')
        if( .not. time_rev ) &
           write(91,'("%    R^-1 k - k = [", 3F10.6, " ]  in b_i")') kp(1:3)
        if( time_rev ) &
           write(91,'("%    R^-1 k + k = [", 3F10.6, " ]  in b_i")') kp(1:3)
           !
        if(WDES%NCDIJ==4) then
           write(91,'("%          sig  = [", 2("(", 2F10.6, "i)  ")  )') sig(0,0:1)
           write(91,'("%                  ", 2("(", 2F10.6, "i)  "), " ];"  )') sig(1,0:1)
        endif
           !
    ENDIF
    !
endsubroutine vmat_rot_matrix 

!=============================================================
! this subroutine calculates Mmnkb < u_1| |u_2 >
! in units of hbar/bohr
!=============================================================
subroutine vmat_woverlap(cc, b1, b2, k1, k2, dk_in,ng2,tf1,cpt,WDES, W, sp, NONLR_S, NONL_S, GRID, CQIJ, LATT_CUR, KPOINTS)
    !
    complex(q) :: cc
    integer :: b1, b2, k1, k2
    integer, optional :: dk_in(3)                                ! dk = R^-1 k2 - k1. dk should be a reciprocal vector
  ! real(q), optional :: rot_in(3,3), tau_in(3)        ! rotation operator is defined by rot, tau, sig and rev
  ! complex(q), optional :: sig_in(0:1, 0:1)
  ! logical, optional :: rev_in
    integer :: ng2(maxng)
    logical :: tf1(maxng)
    type(wavedes) :: WDES
    type(wavespin) :: W
    integer :: sp
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    type(grid_3d) :: GRID
    complex(q) :: CQIJ(:,:,:,:)
    type(latt) :: LATT_CUR
    type(kpoints_struct) KPOINTS
    !
    integer :: dk(3)
    integer,save :: kkk = 0
    real(q) :: rot(3,3), tau(3)
    complex(q) :: sig(0:1, 0:1)
    logical :: rev
    logical :: ldk, lrot, ltau, lsig
    !
    integer :: isp, m, mm, isp_, m_sz, mm_
    integer :: NI, NIS, NT, NPRO, NPRO_, LMMAXC
    integer :: ii, jj
    real(q) :: kp(3), G(3), Gp(3), rtmp
    integer :: Gpt(3)
    complex(q), target :: cpt(WDES%NRSPINORS*maxng)
    complex(q), allocatable, target :: cptr(:)
    complex(q) :: c1, c2, c3
    !
    TYPE (wavefun1) :: W1
    TYPE (wavedes1), target :: WDES1

    !    
    ! rotation operator:
    !
    ldk=.false.; lrot=.false.;  ltau=.false.;  lsig=.false.
    !
    if(present(dk_in)) then
        dk(:)=dk_in(:)
        do ii=1,3
            if ( dk(ii) .ne. 0 ) then
            ldk=.true.
            exit
            endif
        enddo
    else
        dk=0
    endif
    !

        rot(1,:) = (/1.0_q, 0.0_q, 0.0_q/)
        rot(2,:) = (/0.0_q, 1.0_q, 0.0_q/)
        rot(3,:) = (/0.0_q, 0.0_q, 1.0_q/)
    !
        tau = (/0.0_q, 0.0_q, 0.0_q/)
    !
        sig(:,:) = 0.0_q
        sig(0,0) = 1.0_q
        sig(1,1) = 1.0_q
    !
        rev=.false.
    !
    kp = KPOINTS%VKPT(1:3,k1)   ! the coordinates of k1
    !
   !allocate( cpt( WDES%NRSPINORS*WDES%NGVECTOR(k1) ) )         ! in format of k1 wavefunction
   !cpt(:) = 0.0_q
    !
    ! rotate the b2 pseudo-wavefunction======================================================
    !
    IF(kkk/=k1) THEN
    ng2(:)=0;tf1(:)=.false.
    do m=1, WDES%NGVECTOR(k2)
        do m_sz=1, WDES%NGVECTOR(k1)
            !
            if ( WDES%IGX(m,k2)+dk(1)==WDES%IGX(m_sz,k1) .and. &
                 WDES%IGY(m,k2)+dk(2)==WDES%IGY(m_sz,k1) .and. &
                 WDES%IGZ(m,k2)+dk(3)==WDES%IGZ(m_sz,k1) ) then
                !
                ng2(m_sz)=m; tf1(m_sz)=.true.
                !
                exit ! exit the loop
                !
            endif
            !
        enddo
    enddo
    ENDIF

        cpt(:) = cmplx(0._q,0._q,kind=q)
        do m_sz=1, WDES%NGVECTOR(k1)
           IF(tf1(m_sz)) THEN
               do isp_=0, WDES%NRSPINORS-1
               do isp =0, WDES%NRSPINORS-1
                   mm_= m_sz+ isp_*WDES%NGVECTOR(k1)
                   mm = ng2(m_sz) + isp *WDES%NGVECTOR(k2)
                   !
                    cpt(mm_) = cpt(mm_) + sig(isp_,isp) * W%CPTWFP(mm ,b2, k2, sp)!*c1
                   !
               enddo
               enddo
           ENDIF
        enddo
!
! rotate the pseudo wavefunction------------------------------------------
#if .false.
    !
    c1=0.0_q
    c2=0.0_q
    c3=0.0_q
    !
    do m=1, WDES%NGVECTOR(k1)
    do isp =0, WDES%NRSPINORS-1
        mm = m + isp *WDES%NGVECTOR(k1)
        !
        c1 = c1 + conjg(cpt(mm))*cpt(mm)
        c2 = c2 + conjg(W%CPTWFP(mm , b1, k1, sp)) * W%CPTWFP(mm , b1, k1, sp)
        c3 = c3 + conjg(W%CPTWFP(mm , b1, k1, sp))*cpt(mm)
        !
    enddo
    enddo
    !
    cc = c3/sqrt(abs(c1))/sqrt(abs(c2))
    !
!
! rotate the AE wavefunction----------------------------------------------
#else
    !
    ! transform to real space ------------------------------------------
    !
    if( lrot .or. ltau .or. lsig .or. ldk) then
        !
        allocate(cptr(WDES%NRSPINORS * GRID%MPLWV))
        cptr = 0.0_q
        !
        do isp=0,WDES%NRSPINORS-1
                call FFTWAV( WDES%NGVECTOR(k1),WDES%NINDPW(1,k1), &
                             cptr(1+isp*GRID%MPLWV), &
                             cpt(1+isp*WDES%NGVECTOR(k1)), &
                             GRID)
        enddo
        !
        call SETWDES(WDES,WDES1,k1)
        !
        W1%CPTWFP=>cpt
        allocate( W1%CPROJ( size( W%CPROJ, 1) ) )
        W1%CR=>cptr
        W1%FERWE =W%FERWE(b2,k1,sp)
        W1%CELEN =W%CELEN(b2,k1,sp)
        W1%WDES1 => WDES1
        W1%NB =b2
        W1%ISP =sp
        W1%LDO=.TRUE.
        !
        ! get projections
        IF ( NONLR_S%LREAL) CALL PHASER(GRID,LATT_CUR,NONLR_S,k1,W%WDES)
        IF ( .not. NONLR_S%LREAL) CALL PHASE(W%WDES,NONL_S,k1)
        call W1_PROJ(W1, NONLR_S, NONL_S) 
        !
    endif
    !
    ! overlap-----------------------------------------------------------
    !
    c1 = 0.0_q
    do isp =0, WDES%NRSPINORS-1
        do m=1, WDES%NGVECTOR(k1)
            mm = m +  isp*WDES%NGVECTOR(k1)
            c1 = c1 +  conjg( W%CPTWFP(mm ,b1, k1, sp)) &
                       *cpt(mm) 
        enddo
    enddo
    !
    c2 = 0.0_q
    do isp =0, WDES%NRSPINORS-1
    do isp_=0, WDES%NRSPINORS-1
        !
        NPRO = isp * ( WDES%NPRO/2 )
        NPRO_= isp_* ( WDES%NPRO/2 )
        NIS = 1
        !
        atomtype:DO NT=1, WDES%NTYP
            LMMAXC=WDES%LMMAX(NT)
            !
            atom:DO NI = NIS, WDES%NITYP(NT)+ NIS -1
                !
                do mm =1,LMMAXC
                do mm_=1,LMMAXC
                    !
                    ! VASP uses a reverted notation
                    ! D(lm, l'm',alpha+2*alpha') = <alpha'|<l'm'|D|lm>|alpha>
                    ! While pauli matrix's notaion is normal
                    !
                    if( lrot .or. ltau .or. lsig .or. ldk) then
                        c2 = c2 + CQIJ(mm_,mm,NI,sp+isp_+2*isp) &
                                * conjg( W%CPROJ(mm+NPRO ,b1, k1, sp)) &
                                * W1%CPROJ(mm_+NPRO_)
                    else
                        c2 = c2 + CQIJ(mm_,mm,NI,sp+isp_+2*isp) &
                                * conjg( W%CPROJ(mm+NPRO ,b1, k1, sp)) &
                                * W%CPROJ(mm_+NPRO_,b2,k2,sp)
                    endif
                    !
                enddo
                enddo
                !
                NPRO = NPRO + LMMAXC
                NPRO_= NPRO_+ LMMAXC
                !
            ENDDO atom
            NIS = NIS + WDES%NITYP(NT)
            !
        ENDDO atomtype
        !
    enddo
    enddo
    !
    cc = c1 + c2
    !
    if( lrot .or. ltau .or. lsig .or. ldk) then
        deallocate( W1%CPROJ )
        deallocate( cptr )
    endif
#endif
    !
   !deallocate( cpt )
    !
    kkk=k1
endsubroutine

!=============================================================
! this subroutine calculates < psi_1| Rot |psi_2 >
! in units of hbar/bohr
!=============================================================
subroutine vmat_rotation(cc, b1, b2, k1, k2, dk_in, rot_in, tau_in, sig_in, rev_in, WDES, W, sp, NONLR_S, NONL_S, GRID, CQIJ, LATT_CUR, KPOINTS)
    !
    complex(q) :: cc
    integer :: b1, b2, k1, k2
    real(q), optional :: dk_in(3)                                ! dk = R^-1 k2 - k1. dk should be a reciprocal vector
    real(q), optional :: rot_in(3,3), tau_in(3)        ! rotation operator is defined by rot, tau, sig and rev
    complex(q), optional :: sig_in(0:1, 0:1)
    logical, optional :: rev_in
    type(wavedes) :: WDES
    type(wavespin) :: W
    integer :: sp
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    type(grid_3d) :: GRID
    complex(q) :: CQIJ(:,:,:,:)
    type(latt) :: LATT_CUR
    type(kpoints_struct) KPOINTS
    !
    real(q) :: dk(3)
    real(q) :: rot(3,3), tau(3)
    complex(q) :: sig(0:1, 0:1)
    logical :: rev
    logical :: ldk, lrot, ltau, lsig
    !
    integer :: isp, m, mm, isp_, m_sz, mm_
    integer :: NI, NIS, NT, NPRO, NPRO_, LMMAXC
    integer :: ii, jj
    real(q) :: kp(3), G(3), Gp(3), rtmp
    integer :: Gpt(3)
    complex(q), allocatable, target :: cpt(:), cptr(:)
    complex(q) :: c1, c2, c3
    !
    TYPE (wavefun1) :: W1
    TYPE (wavedes1), target :: WDES1
    !    
    ! rotation operator:
    !
    ldk=.false.; lrot=.false.;  ltau=.false.;  lsig=.false.
    !
    if(present(dk_in)) then
        dk=dk_in
        do ii=1,3
            if ( abs(dk(ii)) .ge. 0.1 ) ldk=.true.
        enddo
    else
        dk=(/0.0_q, 0.0_q, 0.0_q/)
    endif
    !
    if (present(rot_in)) then
        rot=rot_in;     lrot=.true. 
    else 
        rot(1,:) = (/1.0_q, 0.0_q, 0.0_q/)
        rot(2,:) = (/0.0_q, 1.0_q, 0.0_q/)
        rot(3,:) = (/0.0_q, 0.0_q, 1.0_q/)
    endif
    !
    if (present(tau_in)) then
        tau=tau_in;     ltau=.true. 
    else
        tau = (/0.0_q, 0.0_q, 0.0_q/)
    endif
    !
    if (present(sig_in)) then
        sig=sig_in;     lsig=.true. 
    else
        sig(:,:) = 0.0_q
        sig(0,0) = 1.0_q
        sig(1,1) = 1.0_q
    endif
    !
    if (present(rev_in)) then
        rev=rev_in
    else
        rev=.false.
    endif
    !
    kp = KPOINTS%VKPT(1:3,k1)   ! the coordinates of k1
    !
    allocate( cpt( WDES%NRSPINORS*WDES%NGVECTOR(k1) ) )         ! in format of k1 wavefunction
    cpt(:) = 0.0_q
    !
    ! rotate the b2 pseudo-wavefunction======================================================
    !
    do m=1, WDES%NGVECTOR(k2)
        G(1)=real( WDES%IGX(m,k2), kind=q )
        G(2)=real( WDES%IGY(m,k2), kind=q )
        G(3)=real( WDES%IGZ(m,k2), kind=q )
        !
        Gp(:) = 0.0_q
        do ii=1,3
        do jj=1,3
            Gp(ii) = Gp(ii) + G(jj)*rot(jj,ii)
        enddo
        enddo
        !
        Gp = Gp + dk
        if(rev) Gp = -Gp
        !
        ! check G grid
        do ii=1,3
            Gpt(ii)=nint( Gp(ii) )
            if( abs(Gp(ii)-real(Gpt(ii),kind=q)) .gt. 0.2e-3 ) then
                write(*, *) 'Error: G grid do not have the symmetry !!!!'
                stop
            endif
        enddo
        !
        ! find the index of Gp
        do m_sz=1, WDES%NGVECTOR(k1)
            !
            if ( Gpt(1)==WDES%IGX(m_sz,k1) .and. &
                 Gpt(2)==WDES%IGY(m_sz,k1) .and. &
                 Gpt(3)==WDES%IGZ(m_sz,k1) ) then
                !
                ! phase factor
                c1 = exp( -imag*TPI*( (kp(1)+Gpt(1))*tau(1)+(kp(2)+Gpt(2))*tau(2)+(kp(3)+Gpt(3))*tau(3) ) )
                !c1 = exp( -imag*TPI*( Gpt(1)*tau(1) + Gpt(2)*tau(2)+ Gpt(3)*tau(3) ) )  ! divide e^(-ik \tau)
                !
                do isp_=0, WDES%NRSPINORS-1
                do isp =0, WDES%NRSPINORS-1
                    mm_= m_sz+ isp_*WDES%NGVECTOR(k1)
                    mm = m + isp *WDES%NGVECTOR(k2)
                    !
                    if(.not. rev) cpt(mm_) = cpt(mm_) + sig(isp_,isp) * W%CPTWFP(mm ,b2, k2, sp)*c1
                    if(      rev) cpt(mm_) = cpt(mm_) + sig(isp_,isp) * conjg(W%CPTWFP(mm , b2, k2, sp))*c1
                    !
                enddo
                enddo
                !
                exit ! exit the loop
                !
            endif
            !
        enddo
        !
    enddo
!
! rotate the pseudo wavefunction------------------------------------------
#if .false.
    !
    c1=0.0_q
    c2=0.0_q
    c3=0.0_q
    !
    do m=1, WDES%NGVECTOR(k1)
    do isp =0, WDES%NRSPINORS-1
        mm = m + isp *WDES%NGVECTOR(k1)
        !
        c1 = c1 + conjg(cpt(mm))*cpt(mm)
        c2 = c2 + conjg(W%CPTWFP(mm , b1, k1, sp)) * W%CPTWFP(mm , b1, k1, sp)
        c3 = c3 + conjg(W%CPTWFP(mm , b1, k1, sp))*cpt(mm)
        !
    enddo
    enddo
    !
    cc = c3/sqrt(abs(c1))/sqrt(abs(c2))
    !
!
! rotate the AE wavefunction----------------------------------------------
#else
    !
    ! transform to real space ------------------------------------------
    !
    if( lrot .or. ltau .or. lsig .or. ldk) then
        !
        allocate(cptr(WDES%NRSPINORS * GRID%MPLWV))
        cptr = 0.0_q
        !
        do isp=0,WDES%NRSPINORS-1
                call FFTWAV( WDES%NGVECTOR(k1),WDES%NINDPW(1,k1), &
                             cptr(1+isp*GRID%MPLWV), &
                             cpt(1+isp*WDES%NGVECTOR(k1)), &
                             GRID)
        enddo
        !
        call SETWDES(WDES,WDES1,k1)
        !
        W1%CPTWFP=>cpt
        allocate( W1%CPROJ( size( W%CPROJ, 1) ) )
        W1%CR=>cptr
        W1%FERWE =W%FERWE(b2,k1,sp)
        W1%CELEN =W%CELEN(b2,k1,sp)
        W1%WDES1 => WDES1
        W1%NB =b2
        W1%ISP =sp
        W1%LDO=.TRUE.
        !
        ! get projections
        IF ( NONLR_S%LREAL) CALL PHASER(GRID,LATT_CUR,NONLR_S,k1,W%WDES)
        IF ( .not. NONLR_S%LREAL) CALL PHASE(W%WDES,NONL_S,k1)
        call W1_PROJ(W1, NONLR_S, NONL_S) 
        !
    endif
    !
    ! overlap-----------------------------------------------------------
    !
    c1 = 0.0_q
    do isp =0, WDES%NRSPINORS-1
        do m=1, WDES%NGVECTOR(k1)
            mm = m +  isp*WDES%NGVECTOR(k1)
            c1 = c1 +  conjg( W%CPTWFP(mm ,b1, k1, sp)) &
                       *cpt(mm) 
        enddo
    enddo
    !
    c2 = 0.0_q
    do isp =0, WDES%NRSPINORS-1
    do isp_=0, WDES%NRSPINORS-1
        !
        NPRO = isp * ( WDES%NPRO/2 )
        NPRO_= isp_* ( WDES%NPRO/2 )
        NIS = 1
        !
        atomtype:DO NT=1, WDES%NTYP
            LMMAXC=WDES%LMMAX(NT)
            !
            atom:DO NI = NIS, WDES%NITYP(NT)+ NIS -1
                !
                do mm =1,LMMAXC
                do mm_=1,LMMAXC
                    !
                    ! VASP uses a reverted notation
                    ! D(lm, l'm',alpha+2*alpha') = <alpha'|<l'm'|D|lm>|alpha>
                    ! While pauli matrix's notaion is normal
                    !
                    if( lrot .or. ltau .or. lsig .or. ldk) then
                        c2 = c2 + CQIJ(mm_,mm,NI,sp+isp_+2*isp) &
                                * conjg( W%CPROJ(mm+NPRO ,b1, k1, sp)) &
                                * W1%CPROJ(mm_+NPRO_)
                    else
                        c2 = c2 + CQIJ(mm_,mm,NI,sp+isp_+2*isp) &
                                * conjg( W%CPROJ(mm+NPRO ,b1, k1, sp)) &
                                * W%CPROJ(mm_+NPRO_,b2,k2,sp)
                    endif
                    !
                enddo
                enddo
                !
                NPRO = NPRO + LMMAXC
                NPRO_= NPRO_+ LMMAXC
                !
            ENDDO atom
            NIS = NIS + WDES%NITYP(NT)
            !
        ENDDO atomtype
        !
    enddo
    enddo
    !
    cc = c1 + c2
    !
    if( lrot .or. ltau .or. lsig .or. ldk) then
        deallocate( W1%CPROJ )
        deallocate( cptr )
    endif
#endif
    !
    deallocate( cpt )
    !
endsubroutine

!=============================================================
! this subroutine calculates < \tildle{psi}_nk| p |\tilde{psi}_mk >
! in units of hbar/bohr
!=============================================================
subroutine vmat_p_pseudo(cc, ii, jj, B, WDES, W, sp, ivec)
    !
    complex(q) :: cc
    integer :: ii, jj
    real(q) :: B(3,3)
    type(wavedes) :: WDES
    type(wavespin) :: W
    integer :: sp
    integer :: ivec
    !
    REAL(q),PARAMETER  :: PI =3.141592653589793238_q,TPI=2*PI
    REAL(q), PARAMETER :: AUTOA=0.529177249_q
    !
    integer :: isp, m, mm
    real(q) :: G1, G2, G3, G
    complex(q), allocatable :: wl_real(:), wr_real(:)
    !
    cc=0
    !
    do isp=0, WDES%NRSPINORS-1
        do m=1, WDES%NGVECTOR(vmat_k)
            !
            G1=(WDES%IGX(m,vmat_k)+WDES%VKPT(1,vmat_k))
            G2=(WDES%IGY(m,vmat_k)+WDES%VKPT(2,vmat_k))
            G3=(WDES%IGZ(m,vmat_k)+WDES%VKPT(3,vmat_k))
            !
            ! do not consider spin spiral, see SET_DATAKE in wave.f90
            !
            if ( WDES%NRSPINORS == 1) then
                G= (G1*B(ivec,1)+G2*B(ivec,2)+G3*B(ivec,3)) *TPI
            else
                if (isp==0) then
                    G= ((G1-WDES%QSPIRAL(1)/2)*B(ivec,1)+(G2-WDES%QSPIRAL(2)/2)*B(ivec,2)+(G3-WDES%QSPIRAL(3)/2)*B(ivec,3)) *TPI
                else
                    G= ((G1+WDES%QSPIRAL(1)/2)*B(ivec,1)+(G2+WDES%QSPIRAL(2)/2)*B(ivec,2)+(G3+WDES%QSPIRAL(3)/2)*B(ivec,3)) *TPI
                endif
            endif
            G = G * AUTOA
            !
            mm = m + isp*WDES%NGVECTOR(vmat_k)
            !
            cc = cc +  conjg( W%CPTWFP(mm ,ii, vmat_k, sp)) &
                       * G * W%CPTWFP(mm ,jj, vmat_k, sp)
        enddo
    enddo
    !
endsubroutine vmat_p_pseudo

!=============================================================
! this subroutine calculates <\tilde{psi}_nk| T |\tilde{psi}_mk>
!=============================================================
subroutine vmat_kinetic_pseudo(cc, ii, jj, B, WDES, W, sp)
    !
    complex(q) :: cc
    integer :: ii, jj
    real(q) :: B(3,3)
    type(wavedes) :: WDES
    type(wavespin) :: W
    integer :: sp
    !
    !
    integer :: isp, m, mm
    real(q) :: G1, G2, G3, GX, GY, GZ
    !
    !
    ! calculate in reciprecal space
    cc=0
    !
#if .true.
    do isp=0, WDES%NRSPINORS-1
        do m=1, WDES%NGVECTOR(vmat_k)
            mm = m + isp*WDES%NGVECTOR(vmat_k)
            cc = cc +  conjg( W%CPTWFP(mm ,ii, vmat_k, sp) ) &
                            * W%CPTWFP(mm ,jj, vmat_k, sp)   &
                            * WDES%DATAKE(m, isp+1, vmat_k)
        enddo
    enddo
    !
#else
    do isp=0, WDES%NRSPINORS-1
        do m=1, WDES%NGVECTOR(vmat_k)
            !
            G1=(WDES%IGX(m,vmat_k)+WDES%VKPT(1,vmat_k))
            G2=(WDES%IGY(m,vmat_k)+WDES%VKPT(2,vmat_k))
            G3=(WDES%IGZ(m,vmat_k)+WDES%VKPT(3,vmat_k))
            !
            ! do not consider spin spiral, see SET_DATAKE in wave.f90
            !
            if ( WDES%NRSPINORS == 1) then
                GX= (G1*B(1,1)+G2*B(1,2)+G3*B(1,3)) *TPI
                GY= (G1*B(2,1)+G2*B(2,2)+G3*B(2,3)) *TPI
                GZ= (G1*B(3,1)+G2*B(3,2)+G3*B(3,3)) *TPI
            else
                if (isp==0) then
                    GX= ((G1-WDES%QSPIRAL(1)/2)*B(1,1)+(G2-WDES%QSPIRAL(2)/2)*B(1,2)+(G3-WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
                    GY= ((G1-WDES%QSPIRAL(1)/2)*B(2,1)+(G2-WDES%QSPIRAL(2)/2)*B(2,2)+(G3-WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
                    GZ= ((G1-WDES%QSPIRAL(1)/2)*B(3,1)+(G2-WDES%QSPIRAL(2)/2)*B(3,2)+(G3-WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
                else
                    GX= ((G1+WDES%QSPIRAL(1)/2)*B(1,1)+(G2+WDES%QSPIRAL(2)/2)*B(1,2)+(G3+WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
                    GY= ((G1+WDES%QSPIRAL(1)/2)*B(2,1)+(G2+WDES%QSPIRAL(2)/2)*B(2,2)+(G3+WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
                    GZ= ((G1+WDES%QSPIRAL(1)/2)*B(3,1)+(G2+WDES%QSPIRAL(2)/2)*B(3,2)+(G3+WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
                endif
            endif
            !
            mm = m + isp*WDES%NGVECTOR(vmat_k)
            !
            cc = cc +  conjg( W%CPTWFP(mm ,ii, vmat_k, sp)) &
                       * W%CPTWFP(mm ,jj, vmat_k, sp) &
                       * (GX*GX+GY*GY+GZ*GZ) * RYTOEV*AUTOA*AUTOA
            !
        enddo
    enddo
#endif
    !
endsubroutine vmat_kinetic_pseudo

!=============================================================
! this subroutine calculates < u_ii(kp1) | u_jj(kp2) >
!=============================================================
subroutine vmat_overlap(cc, kp1, ii, kp2, jj, WDES, W, CQIJ, LMDIM, GRID, sp, pauli)
    !    
    complex(q) :: cc
    integer :: kp1, ii, kp2, jj
    type(wavedes) :: WDES 
    type(wavespin) :: W 
    complex(q) CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    integer :: LMDIM
    type(grid_3d) :: GRID 
    integer :: sp
    complex(q),optional :: pauli(0:1,0:1)
    !    
    complex(q) :: overlap_p, overlap_n
    integer :: isp, isp_, m, m_sz,  mm, mm_
    integer :: NI, NIS, NT, NPRO, NPRO_, LMMAXC
    complex(q), allocatable :: wl_real(:), wr_real(:)
    !    
    if( present(pauli) .and. WDES%NCDIJ/=4 ) then 
        write(*, *) 'Error: pauli matrix can only be computed in non-collinear case'
        stop 
    endif
    !    
    ! pseudowave's contribution-------------------------------------
    !    
    ! the commentted code calculate in real space
    !    
    !allocate( wl_real( WDES%NRSPINORS * GRID%MPLWV), & 
    !          wr_real( WDES%NRSPINORS * GRID%MPLWV) )
    !wl_real=0;  wr_real=0
    !    
    !do isp=0,WDES%NRSPINORS-1
    !    !
    !    call FFTWAV_MPI( WDES%NGVECTOR(vmat_k),WDES%NINDPW(1,vmat_k), &
    !                 wl_real(1+isp*GRID%MPLWV: (isp+1)*GRID%MPLWV ), &
    !                 W%CPTWFP(1+isp*WDES%NGVECTOR(vmat_k) : (isp+1)*WDES%NGVECTOR(vmat_k) ,ii,vmat_k,sp), &
    !                 GRID)
    !    call FFTWAV_MPI( WDES%NGVECTOR(vmat_k),WDES%NINDPW(1,vmat_k), &
    !                 wr_real(1+isp*GRID%MPLWV: (isp+1)*GRID%MPLWV ), &
    !                 W%CPTWFP(1+isp*WDES%NGVECTOR(vmat_k) : (isp+1)*WDES%NGVECTOR(vmat_k) ,jj,vmat_k,sp), &
    !                 GRID)
    !    !
    !enddo
    ! 
    !overlap_p=0
    !do isp=0,wdes%NRSPINORS-1
    !    do m=1,GRID%RL%NP
    !        mm = m+ isp*GRID%MPLWV
    !        overlap_p = overlap_p + conjg(wl_real(mm))*wr_real(mm)
    !    enddo
    !enddo
    !overlap_p = overlap_p/GRID%NPLWV
    !
    !deallocate(wl_real, wr_real)
    !
    ! calculate in reciprecal space
    !
    overlap_p=0
    IF( present(pauli) ) THEN
        if (kp1==kp2) then
            !
            do isp =0, WDES%NRSPINORS-1
            do isp_=0, WDES%NRSPINORS-1
                do m=1, WDES%NGVECTOR(kp1)
                    mm = m +  isp*WDES%NGVECTOR(kp1)
                    mm_= m + isp_*WDES%NGVECTOR(kp2)
                    overlap_p = overlap_p +  conjg( W%CPTWFP(mm ,ii, kp1, sp)) &
                                                  * W%CPTWFP(mm_,jj, kp2, sp)  &
                                                  * pauli(isp,  isp_)
                enddo
            enddo
            enddo
            !
        else
            do m =1, WDES%NGVECTOR(kp1)
            do m_sz=1, WDES%NGVECTOR(kp2)
                !
                if ( WDES%IGX(m ,kp1) == WDES%IGX(m_sz,kp2) .and. &
                     WDES%IGY(m ,kp1) == WDES%IGY(m_sz,kp2) .and. &
                     WDES%IGZ(m ,kp1) == WDES%IGZ(m_sz,kp2) ) then
                    !
                    do isp =0, WDES%NRSPINORS-1
                    do isp_=0, WDES%NRSPINORS-1
                        mm = m +  isp*WDES%NGVECTOR(kp1)
                        mm_= m_sz+ isp_*WDES%NGVECTOR(kp2)
                        overlap_p = overlap_p +  conjg( W%CPTWFP(mm ,ii, kp1, sp)) &
                                                      * W%CPTWFP(mm_,jj, kp2, sp)  &
                                                      * pauli(isp,  isp_)
                    enddo
                    enddo
                    !
                endif
                !
            enddo
            enddo
        endif
    ELSE
        if (kp1==kp2) then
            !
            do isp =0, WDES%NRSPINORS-1
                do m=1, WDES%NGVECTOR(kp1)
                    mm = m +  isp*WDES%NGVECTOR(kp1)
                    overlap_p = overlap_p +  conjg( W%CPTWFP(mm ,ii, kp1, sp)) &
                                                  * W%CPTWFP(mm ,jj, kp2, sp)
                enddo
            enddo
            !
        else
            do m =1, WDES%NGVECTOR(kp1)
            do m_sz=1, WDES%NGVECTOR(kp2)
                !
                if ( WDES%IGX(m ,kp1) == WDES%IGX(m_sz,kp2) .and. &
                     WDES%IGY(m ,kp1) == WDES%IGY(m_sz,kp2) .and. &
                     WDES%IGZ(m ,kp1) == WDES%IGZ(m_sz,kp2) ) then
                    !
                    do isp =0, WDES%NRSPINORS-1
                        mm = m +  isp*WDES%NGVECTOR(kp1)
                        mm_= m_sz+  isp*WDES%NGVECTOR(kp2)
                        overlap_p = overlap_p +  conjg( W%CPTWFP(mm ,ii, kp1, sp)) &
                                                      * W%CPTWFP(mm_,jj, kp2, sp) 
                    enddo
                    !
                endif
                !
            enddo
            enddo
        endif
    ENDIF
    !
    ! non-local contribution-----------------------------------------
    !
    overlap_n =0
    do isp =0, WDES%NRSPINORS-1
    do isp_=0, WDES%NRSPINORS-1
        NPRO = isp * ( WDES%NPRO/2 )
        NPRO_= isp_* ( WDES%NPRO/2 )
        NIS = 1
        !
        atomtype:DO NT=1, WDES%NTYP
            LMMAXC=WDES%LMMAX(NT)
            !
            atom:DO NI = NIS, WDES%NITYP(NT)+ NIS -1
                !
                do mm =1,LMMAXC
                do mm_=1,LMMAXC
                    !
                    ! VASP uses a reverted notation
                    ! D(lm, l'm',alpha+2*alpha') = <alpha'|<l'm'|D|lm>|alpha>
                    ! While pauli matrix's notaion is normal
                    !
                    if(present(pauli)) then
                        overlap_n = overlap_n + CQIJ(mm_,mm,NI,1) &      ! CQIJ is diagnal in spinor space
                                              * pauli(isp, isp_) &
                                              * conjg( W%CPROJ(mm+NPRO ,ii, kp1, sp)) &
                                              * W%CPROJ(mm_+NPRO_, jj, kp2, sp)
                    else
                        overlap_n = overlap_n + CQIJ(mm_,mm,NI,1+isp_+2*isp) &
                                              * conjg( W%CPROJ(mm+NPRO ,ii, kp1, sp)) &
                                              * W%CPROJ(mm_+NPRO_, jj, kp2, sp)
                    endif
                    !
                enddo
                enddo
                !
                NPRO = NPRO + LMMAXC
                NPRO_= NPRO_+ LMMAXC
                !
            ENDDO atom
            NIS = NIS + WDES%NITYP(NT)
            !
        ENDDO atomtype
        !
    enddo
    enddo
    !
    !
    cc = overlap_p + overlap_n
    !
endsubroutine vmat_overlap

!=============================================================
! this subroutine calculates < \tildle{psi}_nk | \tildle{V}_eff | \tildle{psi}_mk >
!=============================================================
subroutine vmat_vlocal_pseudo(cc, ii, jj, WDES, W, GRID, SV, sp)
    !
    complex(q) :: cc
    integer :: ii, jj
    type(wavedes) :: WDES
    type(wavespin) :: W
    type(grid_3d) :: GRID
    complex(q) :: SV(GRID%MPLWV,WDES%NCDIJ)
    !COMPLEX(q) :: SV(:,:)
    integer :: sp
    !
    integer :: isp, isp_, m, mm, mm_
    !
    complex(q), allocatable :: cwork1(:), cwork2(:)
    !
    ! calculate in real space 
    !
    allocate( cwork1( WDES%NRSPINORS * GRID%MPLWV), & 
              cwork2( WDES%NRSPINORS * GRID%MPLWV) )
    cwork1=0;  cwork2=0
    ! 
    do isp=0,WDES%NRSPINORS-1
        !
        call FFTWAV( WDES%NGVECTOR(vmat_k),WDES%NINDPW(1,vmat_k), &
                     cwork1(1+isp*GRID%MPLWV), &
                     W%CPTWFP(1+isp*WDES%NGVECTOR(vmat_k) ,jj,vmat_k,sp), &
                     GRID)
        call FFTWAV( WDES%NGVECTOR(vmat_k),WDES%NINDPW(1,vmat_k), &
                     cwork2(1+isp*GRID%MPLWV), &
                     W%CPTWFP(1+isp*WDES%NGVECTOR(vmat_k) ,ii,vmat_k,sp), &
                     GRID)
        !
    enddo
    !
    cc=0;
    do isp=0, WDES%NRSPINORS-1
        do isp_=0, WDES%NRSPINORS-1
            do m=1, GRID%RL%NP
                mm = m + isp *GRID%MPLWV
                mm_= m + isp_*GRID%MPLWV
                cc = cc + SV( m, 1+isp_+2*isp )*conjg(cwork2(mm))*cwork1(mm_)
            enddo
        enddo
    enddo
    !
    deallocate( cwork1, cwork2 )
    !
    cc=cc/GRID%NPLWV
    !
endsubroutine vmat_vlocal_pseudo

!=============================================================
! this subroutine calculates <\tilde{psi}_nk| V_NL |\tilde{psi}_mk>
!=============================================================
subroutine vmat_nlpot_pseudo(cc, ii, jj, WDES, W, CDIJ,LMDIM, sp)
    !
    complex(q) :: cc
    integer :: ii, jj
    type(wavedes) :: WDES
    type(wavespin) :: W
    complex(q) CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    integer :: LMDIM
    integer :: sp
    !
    integer :: isp, isp_, isp_p, m, mm, mm_
    integer :: NI, NIS, NT, NPRO, NPRO_, LMMAXC
    !
    !
    ! non-local contribution-----------------------------------------
    !
    cc =0
    do isp =0, WDES%NRSPINORS-1
    do isp_=0, WDES%NRSPINORS-1
        NPRO = isp * ( WDES%NPRO/2 )
        NPRO_= isp_* ( WDES%NPRO/2 )
        NIS = 1
        !
        atomtype:DO NT=1, WDES%NTYP
            !
            LMMAXC=WDES%LMMAX(NT)
            !
            IF(LMMAXC==0) GOTO 310
            !
            atom:DO NI = NIS, WDES%NITYP(NT)+ NIS -1
                !
                do mm =1,LMMAXC
                do mm_=1,LMMAXC
                        !
                        ! VASP uses a reverted notation
                        ! D(lm, l'm',alpha+2*alpha') = <alpha'|<l'm'|D|lm>|alpha>
                        !
                        cc = cc + conjg(W%CPROJ(mm+NPRO ,ii, vmat_k, sp)) &
                                * CDIJ(mm_,mm,NI,1+isp_+2*isp) &
                                * W%CPROJ(mm_+NPRO_, jj, vmat_k, sp) 
                        !
                enddo
                enddo
                !
                NPRO = NPRO + LMMAXC
                NPRO_= NPRO_+ LMMAXC
                !
            ENDDO atom
            !
            310 NIS = NIS + WDES%NITYP(NT)
            !
        ENDDO atomtype
        !
    enddo
    enddo
    !
endsubroutine vmat_nlpot_pseudo 

!=============================================================
!                         SET_CPIJ
! prepare the data:
! <phi_i| p |phi_j> - <\tildle{phi}_i| p |\tildle{phi}_j>
! p = -i hbar \nabla
!
! the order of index in CPIJ is same with CDIJ
! the unit of these elements is hbar/bohr
!
!=============================================================
subroutine SET_CPIJ( CPIJ, LMDIM, P, WDES, T_INFO, ivec)
    !
    integer :: LMDIM
    type(wavedes) :: WDES
    TYPE (type_info) T_INFO
    complex(q) :: CPIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    type(potcar), target :: P(T_INFO%NTYP)
    integer :: ivec
    !
    integer :: NI, NT, ISP, LM, LMP, LL, LLP, MM, MMP, CH1, CH2
    integer :: LLMAX, LMMAX, INDLM, INDLMP
    integer :: ii
    real(q) :: cc, dd
    real(q), allocatable :: YLM_NABLA_YLM(:,:,:), YLM_X_YLM(:,:,:)
    real(q), allocatable :: WTMP(:), DWAE2(:), DWPS2(:)
    real(q) :: W12_r, W12_d
    type(potcar), pointer :: PP
    !
    !
    allocate( WTMP( maxval( P(:)%R%NMAX ) ), &
              DWAE2( maxval( P(:)%R%NMAX ) ) , DWPS2( maxval( P(:)%R%NMAX ) ) )
    !
    LLMAX = 0
    do ii=1,T_INFO%NTYP
        LLMAX = max( LLMAX, maxval( P(ii)%LPS(:) ) )
    enddo
    LMMAX = (LLMAX+1)**2
    !
    ! set the nabla elements between spherical harmonics
    !
    allocate( YLM_NABLA_YLM(LMMAX, LMMAX, 0:3), YLM_X_YLM(LMMAX, LMMAX, 0:3) )
    CALL SETYLM_NABLA_YLM(LLMAX, YLM_NABLA_YLM, YLM_X_YLM)
    !
    ! loop
    !
    CPIJ(:,:,:,:) = 0.0_q
    !
    ion: DO NI=1,T_INFO%NIONS
        !
        NT = T_INFO%ITYP(NI)
        PP=>PP_POINTER(P,NI,NT)
        !
        LM=1
        channel1: DO CH1=1,PP%LMAX
        LMP=1
        channel2: DO CH2=1,PP%LMAX
            !
            LL=PP%LPS(CH1)
            LLP=PP%LPS(CH2)
            !
            ! radial partial
            !
            WTMP(:)=PP%WAE(:,CH2)/PP%R%R
            CALL GRAD( PP%R, WTMP, DWAE2 )
            DWAE2 = DWAE2*PP%R%R
            !
            WTMP(:)=PP%WPS(:,CH2)/PP%R%R
            CALL GRAD( PP%R, WTMP, DWPS2 )
            DWPS2 = DWPS2*PP%R%R
            !
            ! radial integral
            !
            W12_r=0; W12_d=0
            !
            DO ii=1,PP%R%NMAX
                !
                W12_r = W12_r + PP%WAE(ii,CH1)*PP%WAE(ii,CH2) /PP%R%R(ii) *PP%R%SI(ii) &
                              - PP%WPS(ii,CH1)*PP%WPS(ii,CH2) /PP%R%R(ii) *PP%R%SI(ii)
                !
                W12_d = W12_d + PP%WAE(ii,CH1)*DWAE2(ii) *PP%R%SI(ii) &
                              - PP%WPS(ii,CH1)*DWPS2(ii) *PP%R%SI(ii)
                !
            ENDDO
            !
            !
            DO MM=1,2*LL+1
            DO MMP=1,2*LLP+1
                !
                INDLM = LL*LL + MM
                INDLMP= LLP*LLP + MMP
                !
                ! check for norm of wave
                ! PP%R%SI(ii) is just measure of dr
                !DO ii=1,PP%R%NMAX
                !    cc = cc + PP%WAE(ii,CH1)*PP%WAE(ii,CH1)*PP%R%SI(ii)
                !    dd = dd + PP%WAE(ii,CH1)*PP%WAE(ii,CH1)*PP%R%R(ii) * PP%R%H
                !ENDDO
                !
                ! check for nabla
                !print*, LL,MM,LLP,MMP
                !write(*,'(4F12.7)'), YLM_NABLA_YLM(INDLM,INDLMP,0), YLM_NABLA_YLM(INDLM,INDLMP,1), &
                !        YLM_NABLA_YLM(INDLM,INDLMP,2), YLM_NABLA_YLM(INDLM,INDLMP,3)
                !
                if ( ABS(YLM_NABLA_YLM(INDLM,INDLMP,ivec))> 1E-8 .or. ABS(YLM_X_YLM(INDLM,INDLMP,ivec))>1E-8 ) then
                    !
                    ! VASP uses a reverted notation
                    ! D(lm, l'm',alpha+2*alpha') = <alpha'|<l'm'|D|lm>|alpha>
                    !
                    if ( LM+MM .le. LMP+MMP ) then
                        CPIJ(LMP+MMP-1,LM+MM-1,NI,1) =  w12_r*YLM_NABLA_YLM(INDLM,INDLMP,ivec) &
                                                      +w12_d*YLM_X_YLM(INDLM,INDLMP,ivec)
                        !
                        if( LM+MM/=LMP+MMP) CPIJ(LM+MM-1,LMP+MMP-1,NI,1) = -CPIJ(LMP+MMP-1,LM+MM-1,NI,1) 
                    endif
                endif
                !
            ENDDO
            ENDDO
            !
            LMP = LMP + 2*LLP+1
        ENDDO channel2
        LM = LM + 2*LL + 1
        ENDDO channel1
        !
    ENDDO ion
    !
    deallocate( YLM_NABLA_YLM, YLM_X_YLM )
    deallocate( WTMP, DWPS2, DWAE2 )
    !
    ! spinors
    !
    if ( WDES%NCDIJ == 2) then
        CPIJ(:,:,:,2) = CPIJ(:,:,:,1)
    elseif ( WDES%NCDIJ == 4) then
        CPIJ(:,:,:,4) = CPIJ(:,:,:,1)
        CPIJ(:,:,:,2) = 0
        CPIJ(:,:,:,3) = 0
    endif
    !
    ! now translate the unit to hbar/bohr
    !
    CPIJ(:,:,:,:) = -imag*AUTOA*CPIJ(:,:,:,:)
    !
endsubroutine

!================================================================
!                           SET_CSOIJ
!   calculates Hso elements
!   can also be used in collinear case
!
!================================================================
subroutine SET_CSOIJ( LMDIM, P, CSOIJ, WDES, T_INFO )
    !
    use relativistic,  only : SPINORB_STRENGTH
    !
    integer :: LMDIM
    TYPE (type_info) :: T_INFO
    type(potcar), target :: P(T_INFO%NTYP)
    type(wavedes) :: WDES
    complex(q) :: CSOIJ(LMDIM, LMDIM, WDES%NIONS, 4)
    !
    integer :: NI, NT
    complex(q) :: CSO(LMDIM, LMDIM, 4)
    real(q) :: alpha, beta
    type(potcar), pointer :: PP
    !
    if ( WDES%NCDIJ/=4) then
        write(*,*) 'Internal error: Hso can only be computed in non-collinear case'
        stop
    endif
    !
    CSOIJ(:,:,:,:) = 0.0_q
    !
    CALL EULER( WDES%SAXIS, alpha, beta)    
    !
    ion: DO NI=1,T_INFO%NIONS
        !
        CSO(:,:,:) = 0.0_q
        !
        NT = T_INFO%ITYP(NI)
        PP=>PP_POINTER(P,NI,NT)
        !
        CALL SPINORB_STRENGTH(   POTAE_all(:,1,1,NI), &
                                 PP%RHOAE, PP%POTAE_XCUPDATED, PP%R, CSO, &
                                 PP%LMAX, PP%LPS, PP%WAE, PP%ZCORE+PP%ZVALF_ORIG, THETA=beta, PHI=alpha )
        !
        CSOIJ(:,:,NI,:) = CSO
        !
    ENDDO ion

endsubroutine

!================================================================
!                           SET_CNABLAVIJ
!   calculates Hso elements
!   can also be used in collinear case
!
!================================================================
subroutine SET_CNABLAVIJ( LMDIM, P, CIJ, WDES, T_INFO, ivec, pauli )
    !
    use relativistic,  only : SET_CNABLAVIJ_
    !
    integer :: LMDIM
    TYPE (type_info) :: T_INFO
    type(potcar), target :: P(T_INFO%NTYP)
    type(wavedes) :: WDES
    complex(q) :: CIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    integer :: ivec
    complex(q), optional :: pauli(0:1,0:1)
    !
    integer :: NI, NT, is, isp, ispp
    complex(q) :: CTMP(LMDIM, LMDIM)
    type(potcar), pointer :: PP
    complex(q) :: sig(0:1,0:1)
    complex(q) :: CIJ_(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    !
    if( present(pauli) .and. WDES%NCDIJ/=4 ) then
        write(*, *) 'Error: pauli matrix can only be computed in non-collinear case'
        stop
    endif
    !
    sig = cmplx( 0.0_q, 0.0_q, kind=q)
    sig(0,0) = 1.0_q 
    sig(1,1) = 1.0_q 
    if( present( pauli ) ) sig = pauli
    !
    CIJ_(:,:,:,:) = 0.0_q
    !
    ion: DO NI=1,T_INFO%NIONS
        !
        CTMP(:,:) = 0.0_q
        !
        NT = T_INFO%ITYP(NI)
        PP=>PP_POINTER(P,NI,NT)
        !
        CALL SET_CNABLAVIJ_(   maxval(PP%LPS(:)), POTAE_all(:,1,1,NI), &
                               PP%RHOAE, PP%POTAE_XCUPDATED, PP%R, CTMP, &
                               PP%LMAX, PP%LPS, PP%WAE, PP%ZCORE+PP%ZVALF_ORIG, ivec) 
        !
        CIJ_(:,:,NI,1) = CTMP
        !
    ENDDO ion
    !
    ! spinors
    !
    if ( WDES%NCDIJ == 2) then
        CIJ(:,:,:,1) = CIJ_(:,:,:,1)
        CIJ(:,:,:,2) = CIJ_(:,:,:,1)
    elseif ( WDES%NCDIJ == 4) then
        CIJ_(:,:,:,4) = CIJ_(:,:,:,1)
        do is=0,1
        do isp=0,1
            !
            ! CIJ(:,:,:,1+is+2*isp) = <isp| sig \nabla V |is>
            !                       = <isp| sig |0> <0| \nabla V |is> + <isp| sig |1> <1| \nabla V |is>
            !
            CIJ(:,:,:,1+is+2*isp) = sig(isp,0) * CIJ_(:,:,:,1+is+2*0) + sig(isp,1) * CIJ_(:,:,:,1+is+2*1)
            !
        enddo
        enddo
    endif
    !
    ! transform unit to Ha/bohr
    !
    CIJ = CIJ*AUTOA/HatoeV
    !
endsubroutine SET_CNABLAVIJ

!====================================================
! SET_CRDVIJ
! <phi_i| x_u dV/dx_v |phi_j>
!
!subroutine SET_CRDVIJ( LMDIM, P, CIJ, WDES, T_INFO )
!    !
!    use relativistic,  only : SET_CRDVIJ_
!    !
!    integer :: LMDIM
!    TYPE (type_info) :: T_INFO
!    type(potcar), target :: P(T_INFO%NTYP)
!    type(wavedes) :: WDES
!    complex(q) :: CIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!    !
!    integer :: NI, NT
!    complex(q) :: CTMP(LMDIM, LMDIM)
!    type(potcar), pointer :: PP
!    !
!    CIJ(:,:,:,:) = 0.0_q
!    !
!    ion: DO NI=1,T_INFO%NIONS
!        !
!        CTMP(:,:) = 0.0_q
!        !
!        NT = T_INFO%ITYP(NI)
!        PP=>PP_POINTER(P,NI,NT)
!        !
!        CALL SET_CRDVIJ_(   maxval(PP%LPS(:)), POTAE_all(:,1,1,NI), &
!                            PP%RHOAE, PP%POTAE_XCUPDATED, PP%R, CTMP, &
!                            PP%LMAX, PP%LPS, PP%WAE, PP%ZCORE+PP%ZVALF_ORIG, ivec, ivec_) 
!        !
!        CIJ(:,:,NI,1) = CTMP
!        !
!    ENDDO ion
!    !
!    ! spinors
!    !
!    if ( WDES%NCDIJ == 2) then
!        CIJ(:,:,:,2) = CIJ(:,:,:,1)
!    elseif ( WDES%NCDIJ == 4) then
!        CIJ(:,:,:,4) = CIJ(:,:,:,1)
!        CIJ(:,:,:,2) = 0
!        CIJ(:,:,:,3) = 0
!    endif
!    !
!    CIJ = CIJ       ! unit of CIJ is eV
!    !
!endsubroutine SET_CRDVIJ

endmodule song_vmat


EOF
)

# new content in paw.F
paw_F=$(cat<<'EOF'
!*******************************************************************
    SUBROUTINE SET_DD_PAW_song(WDES, P , T_INFO, LOVERL, &
         ISPIN, LMDIM, CDIJ, RHOLM_STORE, CRHODE, &
         E, LMETA, LASPH, LCOREL )
#ifdef _OPENACC
      USE mopenacc
#endif
      USE pseudo
      USE asa
      USE poscar
      USE wave
      USE constant
      USE radial
      USE base
      USE relativistic
      USE ldaplusu_module
      USE cl
      USE pawfock_inter
      USE setexm
      USE egrad
      USE meta
      USE setxcmeta
      USE hyperfine
      use song_data     , only : POTAE_all,nosoc_inH    ! songzd 2015/03/25
      USE fock_glb, ONLY : AEXX
! embedding__
      USE mextpot
! embedding__
      IMPLICIT NONE

      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      TYPE (wavedes)  WDES
      TYPE (energy)   E
      INTEGER LMDIM, ISPIN
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q)  RHOLM_STORE(:,:)
      LOGICAL  LOVERL
      LOGICAL  LMETA      !< calculate meta GGA contribution
      LOGICAL  LASPH      !< calculate aspherical corrections to potential
      LOGICAL  LCOREL     !< calculate accurate core level shifts
    ! local variables
      TYPE (potcar),POINTER:: PP
      INTEGER NT,LYMAX,NI,NDIM,LMMAX,NIP,ISP,IBASE,IADD,ISIZE,K,ITMP,NCDIJ,LMAX
      INTEGER, EXTERNAL :: MAXL_AUG,MAXL1
      LOGICAL, EXTERNAL :: USEFOCK_CONTRIBUTION, USEFOCK_AE_ONECENTER
! automatic arrays crash on IBM and SGI (thank's to Lucian Anton NAG, Zhengji Zhao SGI)
!      REAL(q) DDLM(LMDIM*LMDIM),RHOLM(LMDIM*LMDIM),RHOLM_(LMDIM*LMDIM,WDES%NCDIJ)
!      OVERLAP CTMP(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),CSO(LMDIM,LMDIM,WDES%NCDIJ), &
!              CHF(LMDIM,LMDIM,WDES%NCDIJ)
!      OVERLAP COCC(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),COCC_IM(LMDIM,LMDIM)
      REAL(q), ALLOCATABLE :: DDLM(:),RHOLM(:),RHOLM_(:,:)
      OVERLAP, ALLOCATABLE :: CTMP(:,:,:),CSO(:,:,:),CHF(:,:,:)
      OVERLAP, ALLOCATABLE :: COCC(:,:,:),COCC_IM(:,:)
      REAL(q), ALLOCATABLE :: POT(:,:,:), RHO(:,:,:), POTAE(:,:,:), RHOAE(:,:,:)
      REAL(q), ALLOCATABLE :: RHOCOL(:,:,:)
      ! core level shifts
      REAL(q), ALLOCATABLE :: DRHOCORE(:)
      REAL(q) :: DOUBLEC_AE,DOUBLEC_PS
      REAL(q) :: DOUBLEPS,DOUBLEAE
      REAL(q) :: EXCG
      REAL(q) :: DOUBLEC_LDAU, DOUBLEC_HF
      ! euler angles of the global spin quantisation axis
      REAL(q) :: ALPHA,BETA
      REAL(q) :: SPI2
      INTEGER :: RNMAX, RNMAX_CL
      INTEGER, EXTERNAL :: ONE_CENTER_NMAX_FOCKAE
      REAL(q) :: EPAWPSG,EPAWAEG,EPAWCORE
! variables for metaGGA calculations
      REAL(q), ALLOCATABLE :: KINDENSCOL(:,:,:)
      REAL(q), ALLOCATABLE :: TAUAE(:,:,:), TAUPS(:,:,:), MUAE(:,:,:), MUPS(:,:,:)
#ifdef noAugXCmeta
      REAL(q), ALLOCATABLE :: RHOPS_STORE(:,:,:),POTH(:,:,:)
#else
#define RHOPS_STORE RHOCOL
#define POTH POT
#endif
      OVERLAP, ALLOCATABLE :: CMETAGGA(:,:,:)
      REAL(q), POINTER :: NULPOINTER(:)
      REAL(q) :: CVMBJ, GRDR, GRDRSUM
      INTEGER :: LMAX_TAU, LMMAX_TAU
! variables required to store core wavefunctions
      INTEGER MAXNL
      REAL(q), ALLOCATABLE :: W(:,:), EIG(:)
      INTEGER, ALLOCATABLE :: N(:), LC(:)
! needed to distribute over COMM_KINTER
      INTEGER IDONE
      LOGICAL LSKIP

!$    INTEGER IDONP
!$    INTEGER, EXTERNAL :: OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM

      PROFILING_START('set_dd_paw')

      PUSH_ACC_EXEC_ON(.FALSE.)
!=======================================================================
! quick return and allocation of work space
!=======================================================================

      E%PAWPSM=0; E%PAWAEM=0; E%PAWCOREM=0
      E%PAWPSG=0; E%PAWAEG=0; E%PAWCORE =0

      DOUBLEC_AE=0
      DOUBLEC_PS=0

      EPAWPSG=0; EPAWAEG=0; EPAWCORE=0

      GRDRSUM=0
      GRDR=0

      CL_SHIFT= 0

      NULLIFY(NULPOINTER)

      IF (.NOT.LOVERL) THEN
         POP_ACC_EXEC_ON
         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

! mimic US-PP just set the double counting corrections correctly
      IF (MIMIC_US) THEN
         DOUBLEC_AE=DOUBLEC_AE_ATOM
         DOUBLEC_PS=DOUBLEC_PS_ATOM

         E%PAWAE=DOUBLEC_AE
         E%PAWPS=DOUBLEC_PS

         POP_ACC_EXEC_ON

         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

      SPI2= 2*SQRT(PI)

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)

      NDIM=0
      DO NT=1,T_INFO%NTYP
         IF (ASSOCIATED(P(NT)%QPAW)) THEN
            NDIM=MAX(NDIM, P(NT)%R%NMAX)
         END IF
      ENDDO

      IF (NDIM == 0) THEN
         POP_ACC_EXEC_ON
         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

      IF (LUSE_THOMAS_FERMI) CALL PUSH_XC_TYPE(LEXCH, ALDAX, ALDAC, AGGAX, AGGAC, AEXX, 0.0_q)

      ALLOCATE(DDLM(LMDIM*LMDIM),RHOLM(LMDIM*LMDIM))
      ALLOCATE(RHOLM_(LMDIM*LMDIM,WDES%NCDIJ))
      ALLOCATE(CTMP(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),CSO(LMDIM,LMDIM,WDES%NCDIJ),CHF(LMDIM,LMDIM,WDES%NCDIJ))
      ALLOCATE(COCC(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),COCC_IM(LMDIM,LMDIM))

      LMMAX=(LYMAX+1)**2
      NCDIJ = WDES%NCDIJ

      ALLOCATE ( POT( NDIM, LMMAX, NCDIJ ), RHO( NDIM, LMMAX, NCDIJ ), &
     &   POTAE( NDIM, LMMAX, NCDIJ ), RHOAE( NDIM, LMMAX, NCDIJ), DRHOCORE(NDIM))
      ALLOCATE ( POTAE_all( NDIM, LMMAX, NCDIJ, T_INFO%NIONS  ) )       ! songzd 2015/3/25
      ALLOCATE (RHOCOL( NDIM, LMMAX, NCDIJ ))
#ifdef noAugXCmeta
      ALLOCATE(RHOPS_STORE(NDIM,LMMAX,NCDIJ),POTH(NDIM,LMMAX,NCDIJ))
#endif
! for metagga
      ALLOCATE (CMETAGGA(LMDIM,LMDIM,NCDIJ))

! for spin orbit coupling set the euler angles
      IF ( WDES%LSORBIT ) CALL EULER(WDES%SAXIS, ALPHA, BETA)

!=======================================================================
! cycle all ions and add corrections to pseudopotential strength CDIJ
!=======================================================================
      IBASE=1; IDONE=0; CVMBJ=CMBJ

!$    CALL SET_RSGF_ALL(P)
!$    IDONP=0

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP PRIVATE(NI, NIP, NT, LSKIP, ISP, PP, LYMAX, RNMAX, LMAX_TAU, LMMAX_TAU, LMMAX, &
!$OMP TAUAE, TAUPS, MUAE, MUPS, KINDENSCOL, RHOLM, RHOLM_, COCC, COCC_IM, ISIZE, IADD, ITMP, &
!$OMP RHOAE, RHO, RHOCOL, &
!$OMP CSO, CHF, POT, DOUBLEPS, EXCG, POTAE, DOUBLEAE, DOUBLEC_HF, DOUBLEC_LDAU, &
#ifdef noAugXCmeta
!$OMP RHOPS_STORE, POTH, &
#endif
!$OMP DRHOCORE, RNMAX_CL, MAXNL, W, N, LC, EIG, CTMP, DDLM, CMETAGGA) &
!$OMP SHARED(VTUTOR, T_INFO,WDES,DO_LOCAL,P,NCDIJ,CDIJ,NDIM,LMAX_MIX,CRHODE,RHOLM_STORE,METRIC,LMDIM, &
!$OMP ISPIN,E,LASPH,ALPHA,BETA,NULPOINTER,LCOREL,CL_SHIFT,SPI2,GRHO_OVER_RHO_AUG,ID_METAGGA) &
!$OMP FIRSTPRIVATE(IBASE,IDONE,IDONP,CVMBJ,GRDR) &
!$OMP REDUCTION(+:DOUBLEC_PS,DOUBLEC_AE,EPAWPSG,EPAWAEG,EPAWCORE,GRDRSUM) &
!$OMP IF(.NOT.EGRAD_CALC_EFG().AND..NOT.LHYPERFINE().AND..NOT.EXTPT_LEXTPOT().AND..NOT.WDES%LSORBIT)
      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)

         LSKIP=.FALSE.
#ifdef MPI
         ! DO_LOCAL represents a distribution of the work on the
         ! one-center terms over the procs in COMM_INB and COMM_INTER (=COMM_KIN).
         ! The following allows an additional round-robin distribution over COMM_KINTER as well.
         IF (DO_LOCAL(NI)) THEN
            IDONE=IDONE+1; LSKIP=(MOD(IDONE,WDES%COMM_KINTER%NCPU)+1/=WDES%COMM_KINTER%NODE_ME)
         ENDIF
#endif
         ! if this element is not treated locally CYCLE
         IF (.NOT. DO_LOCAL(NI).OR.LSKIP) THEN
            ! for PAW, set CDIJ to zero if it resides on local node
            ! and if the element is not treated locally
            IF (ASSOCIATED(P(NT)%QPAW)) THEN
               IF (NIP /= 0) THEN
!$OMP SINGLE
                  DO ISP=1,NCDIJ
                     CDIJ(:,:,NIP,ISP)=0
                  ENDDO
!$OMP END SINGLE NOWAIT
               ENDIF
            ELSE
            ! US PP: initialize to zero if we are not on first node in COMM_INTER
            ! (at the end, we use a global sum over COMM_INTER)
#ifdef MPI
               IF (WDES%COMM_INTER%NODE_ME*WDES%COMM_KINTER%NODE_ME /=1 .AND. NIP /=0 ) THEN
!$OMP SINGLE
                  DO ISP=1,NCDIJ
                     CDIJ(:,:,NIP,ISP)=0
                  ENDDO
!$OMP END SINGLE NOWAIT
               ENDIF
#endif
            ENDIF
#ifdef _OPENMP
         ENDIF
         ! and to have another round-robin distribution over threads ...
         IF (DO_LOCAL(NI).AND.(.NOT.LSKIP)) THEN
            IDONP=IDONP+1; LSKIP=(MOD(IDONP,OMP_GET_NUM_THREADS())/=OMP_GET_THREAD_NUM())
         ENDIF
         IF (.NOT. DO_LOCAL(NI).OR.LSKIP) THEN
#endif
            ! in case we skip work on this ion, we still have to advance IBASE
            IF (LSKIP) THEN
               PP=> PP_POINTER(P, NI, NT)
               IBASE=IBASE+COUNT_RHO_PAW_ELEMENTS(PP)
            ENDIF
            CYCLE ion
         ENDIF

         IF (ID_METAGGA==30) THEN
            !atom type for mBJ
            CALL GET_CMBJ_RAD(NT,CVMBJ)
         ELSEIF (ID_METAGGA==31) THEN
            !atom index for local mBJ
            IF (LDO_METAGGA()) GRHO_OVER_RHO_AUG(NI)=0._q
            CALL GET_CMBJ_RAD(NI,CVMBJ)
         ENDIF

         PP=> PP_POINTER(P, NI, NT)
         CALL SET_RSGF_TYPE(NT)
!        CALL SET_RSGF_SIMPLE(PP)

         LYMAX =MAXL1(PP)*2
         RNMAX =PP%R%NMAX

         LMAX_TAU=LYMAX+2; LMMAX_TAU=(LMAX_TAU+1)**2
         ALLOCATE (TAUAE(NDIM,LMMAX_TAU,NCDIJ),TAUPS(NDIM,LMMAX_TAU,NCDIJ), &
        &   MUAE(NDIM,LMMAX_TAU,NCDIJ),MUPS(NDIM,LMMAX_TAU,NCDIJ),KINDENSCOL(NDIM,LMMAX_TAU,NCDIJ))
  !-----------------------------------------------------------------------
  ! first set RHOLM (i.e. the on site occupancy matrix)
  ! and then the lm dependent charge densities RHO and RHOAE
  ! (excluding augmentation charges yet)
  !-----------------------------------------------------------------------
         RHOLM_=0
  !      WRITE(*,'("RHOMIX",6F10.6)') RHOLM_STORE
         COCC=0
         ISIZE=UBOUND(RHOLM_STORE,1)

         DO ISP=1,NCDIJ
    ! retrieve the one center on site charge densities to RHOLM_
            IF ( LMAX_MIX < PP%LMAX_CALC) &
                 CALL TRANS_RHOLM( CRHODE(:,:,NIP,ISP), RHOLM_(:,ISP), PP )
    ! retrieve mixed elements from RHOLM_STORE and overwrite them in RHOLM_
            CALL RETRIEVE_RHOLM( RHOLM_(:,ISP), RHOLM_STORE(IBASE:,ISP), &
                           METRIC(IBASE:), IADD, PP, .FALSE.,  ITMP)

    ! calculate the total radial angular decomposed charge distributions
            LMMAX=(LYMAX+1)**2
            RHOAE(:,:,ISP)=0; RHO(:,:,ISP)=0
            CALL RAD_CHARGE( RHOAE(:,:,ISP), PP%R,RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WAE )
            CALL RAD_CHARGE( RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WPS )
#ifdef noAugXCmeta
    ! We'll need the pseudo charge density without augmentation in the metagga
            RHOPS_STORE(:,:,ISP)=RHO(:,:,ISP)
#endif
    ! for LDA+U or Hartree fock we need the mixed occupancy matrix COCC
            IF (USELDApU() .OR. LDO_METAGGA() .OR. &
           &   USEFOCK_CONTRIBUTION() .OR. USEFOCK_AE_ONECENTER() .OR. &
           &   LHYPERFINE()) THEN
               ! calculate the occupancy matrix COCC from RHOLM_(:,ISP)
               CALL TRANS_RHOLMI( COCC(:,:,ISP), RHOLM_(:,ISP), PP )
#ifndef realmode
               ! retrieve imaginary part from CRHODE and store in RHOLM
               CALL TRANS_RHOLM_IM( CRHODE(:,:,NIP,ISP), RHOLM, PP )
               ! orverwrite by elements passed through mixer
#ifdef RHOLM_complex
               CALL RETRIEVE_RHOLM( RHOLM, RHOLM_STORE(IBASE+ISIZE/2:,ISP), &
                                 METRIC(IBASE+ISIZE/2:), IADD, PP, .FALSE.,  ITMP)
#endif
               COCC_IM=0
               CALL TRANS_RHOLMI_IM( COCC_IM, RHOLM, PP )
               ! join imaginary and real parts
               COCC(:,:,ISP)=COCC(:,:,ISP)+(0._q,1._q)*COCC_IM(:,:)
!              WRITE(*,*) 'spin',ISP
!              CALL DUMP_DLLMM_IM(COCC(:,:,ISP),PP)
!              CALL DUMP_DLLMM_IM(CRHODE(:,:,NIP,ISP),PP)
#endif
            ENDIF
!              WRITE(*,*) 'spin',ISP
!              CALL DUMP_DLLMM("cocc",COCC(:,:,ISP),PP)
!              CALL DUMP_DLLMM("crhode",CRHODE(:,:,NIP,ISP),PP)
         ENDDO

         ! bring COCC to (up,down)
         IF (WDES%ISPIN==2) CALL OCC_FLIP2(COCC,LMDIM)

         CALL RAD_KINEDEN(PP,PP%WAE,LMAX_TAU,NCDIJ,COCC,TAUAE)
         CALL RAD_KINEDEN(PP,PP%WPS,LMAX_TAU,NCDIJ,COCC,TAUPS)

         ! bring COCC to spinor representation
         IF (WDES%LNONCOLLINEAR) CALL OCC_FLIP4(COCC,LMDIM)

  !-----------------------------------------------------------------------
  ! add augmentation charges now
  !-----------------------------------------------------------------------
         DO ISP=1,NCDIJ
            CALL RAD_AUG_CHARGE(  RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS,  &
                  LYMAX, PP%AUG, PP%QPAW )
            IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
               CALL RAD_AUG_CHARGE_FOCK(  RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS, &
                    LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK)
            ENDIF
            CALL RAD_INT( PP%R,  LYMAX, RHO(:,:,ISP), RHOAE(:,:,ISP) )
         ENDDO
         IBASE=IBASE+IADD

         CALL EGRAD_EFG_RAD_HAR_ONLY(T_INFO,NI,PP,RHO,RHOAE)
  !-----------------------------------------------------------------------
  ! calculate the local radial potential
  ! mind in the non-collinear case the potential V(r) = d E(r) / d rho (r)
  ! and the potential vec mu(r) = d E(r) / d vec m (r) are stored in
  ! POT and POTAE (potentials need to be real), whereas
  ! in the collinear case the spin up and down potentials
  ! are stored in POT and POTAE
  ! probably one should rewrite this in the collinear case
  !-----------------------------------------------------------------------
    ! initialise the spin orbit contributions to D_ij to 0
         CSO=0
    ! Hartree Fock contribution set to zero
         CHF=0

         IF ( WDES%LNONCOLLINEAR ) THEN
            ! bring KINDENSPS from spinor to 2 component spin up and spin down presentation
#ifndef noAugXCmeta
            CALL RAD_MAG_DENSITY_KINDENS(RHO, RHOCOL, TAUPS, KINDENSCOL, LYMAX, LMAX_TAU, PP%R)
#else
            CALL RAD_MAG_DENSITY_KINDENS(RHO, RHOCOL, TAUPS, KINDENSCOL, LYMAX, LMAX_TAU, PP%R, RHOPS_STORE)
#endif
            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            IF (L_NO_LSDA()) RHOCOL(:,:,2:WDES%NCDIJ)=0
#ifdef noAugXCmeta
            IF (L_NO_LSDA()) RHOPS_STORE(:,:,2:WDES%NCDIJ)=0
#endif
            CALL RAD_POT( PP%R, 2, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOPS, PP%POTPS, POT, DOUBLEPS, EXCG)

            CALL RAD_POT_METAGGA( PP%R, 2, PP%LMAX_CALC, LMAX_TAU, LASPH, CVMBJ,  &
               RHOCOL, RHOPS_STORE, PP%RHOPS, PP%POTPS, KINDENSCOL, POTH, POT, DOUBLEPS, EXCG, GRDR, MUPS, NI, -1._q, PP%TAUPS)

            GRDRSUM=GRDRSUM-GRDR
            EPAWPSG=EPAWPSG-EXCG

            CALL RAD_MAG_DIRECTION( RHO, RHOCOL, POT, LYMAX, PP%R)
#ifdef noAugXCmeta
            CALL RAD_MAG_DIRECTION( RHO, RHOCOL, POTH, LYMAX, PP%R)
#endif
            CALL RAD_MAG_DIRECTION_KINDENS( RHO, TAUPS, LMAX_TAU, POT, MUPS, PP%R)

            ! bring KINDENSAE from spinor to 2 component spin up and spin down presentation
            CALL RAD_MAG_DENSITY_KINDENS(RHOAE, RHOCOL, TAUAE, KINDENSCOL, LYMAX, LMAX_TAU, PP%R)

            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            IF (L_NO_LSDA()) RHOCOL(:,:,2:WDES%NCDIJ)=0

            CALL APPLY_ONE_CENTER_AEXX()

            CALL RAD_POT( PP%R, 2, LYMAX, PP%LMAX_CALC, LASPH,  &
                 RHOCOL, PP%RHOAE, PP%POTAE_XCUPDATED,  POTAE, DOUBLEAE, EXCG)

            CALL RAD_POT_METAGGA( PP%R, 2, PP%LMAX_CALC, LMAX_TAU, LASPH, CVMBJ, &
                 RHOCOL, RHOCOL, PP%RHOAE, PP%POTAE_XCUPDATED, KINDENSCOL, POTAE, POTAE, DOUBLEAE, EXCG, GRDR, MUAE, NI, 1._q, PP%TAUAE)

            CALL RESTORE_ONE_CENTER_AEXX

            GRDRSUM=GRDRSUM+GRDR
            EPAWAEG=EPAWAEG+EXCG-PP%DEXCCORE
            EPAWCORE=EPAWCORE+PP%DEXCCORE

            CALL RAD_MAG_DIRECTION( RHOAE, RHOCOL, POTAE, LYMAX, PP%R)

            CALL RAD_MAG_DIRECTION_KINDENS( RHOAE, TAUAE, LMAX_TAU, POTAE, MUAE, PP%R)

            IF (WDES%LSORBIT .and. (.not. nosoc_inH))  &
              CALL SPINORB_STRENGTH(POTAE(:,1,1), PP%RHOAE, PP%POTAE_XCUPDATED, PP%R, CSO, &
                PP%LMAX, PP%LPS ,PP%WAE, PP%ZCORE+PP%ZVALF_ORIG, THETA=BETA, PHI=ALPHA)

         ELSE
! collinear case
            DRHOCORE=0

            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            RHOCOL=RHO
            IF (L_NO_LSDA()) RHOCOL(:,:,2)=0
#ifdef noAugXCmeta
            IF (L_NO_LSDA()) RHOPS_STORE(:,:,2)=0
#endif
            ! cl shifts DRHOCORE is zero here, since for the pseudo terms
            ! we do not include the core electron in the exchange correlation term
            ! but only in the Hartree term
            CALL RAD_POT( PP%R, ISPIN, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOPS-DRHOCORE(1:RNMAX), PP%POTPS, POT, DOUBLEPS, EXCG)

            CALL RAD_POT_METAGGA( PP%R, ISPIN, PP%LMAX_CALC, LMAX_TAU, LASPH, CVMBJ, &
               RHOCOL, RHOPS_STORE, PP%RHOPS-DRHOCORE(1:RNMAX), PP%POTPS, TAUPS, POTH, POT, DOUBLEPS, EXCG, GRDR, MUPS, NI, -1._q, PP%TAUPS)

            GRDRSUM=GRDRSUM-GRDR
            EPAWPSG=EPAWPSG-EXCG

            CALL SET_CL_DRHOCORE_PS(DRHOCORE, NT, PP%R, PP%AUG)
            CALL ADD_CL_HARTREE_POT(DRHOCORE, NT, ISPIN, POT, PP%R)
            !CALL RAD_POT_HAR_ONLY( PP%R, ISPIN, LYMAX, PP%LMAX_CALC,RHOCOL,  POT, DOUBLEPS)

            DRHOCORE=0

            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            RHOCOL=RHOAE
            IF (L_NO_LSDA()) RHOCOL(:,:,2)=0

            CALL SET_CL_DRHOCORE_AE(DRHOCORE, NT, PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG )

            CALL APPLY_ONE_CENTER_AEXX()

            CALL RAD_POT( PP%R, ISPIN, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOAE-DRHOCORE(1:RNMAX), PP%POTAE_XCUPDATED,  POTAE, DOUBLEAE,EXCG)

            CALL RAD_POT_METAGGA( PP%R, ISPIN, PP%LMAX_CALC, LMAX_TAU, LASPH, CVMBJ, &
               RHOCOL, RHOCOL, PP%RHOAE-DRHOCORE(1:RNMAX), PP%POTAE_XCUPDATED, TAUAE, POTAE, POTAE, DOUBLEAE, EXCG, GRDR, MUAE, NI, 1._q, PP%TAUAE)

            GRDRSUM=GRDRSUM+GRDR
            EPAWAEG=EPAWAEG+EXCG-PP%DEXCCORE
            EPAWCORE=EPAWCORE+PP%DEXCCORE

            CALL RESTORE_ONE_CENTER_AEXX

            CALL ADD_CL_HARTREE_POT(DRHOCORE, NT, ISPIN, POTAE, PP%R)

            !CALL RAD_POT_HAR_ONLY( PP%R, ISPIN, LYMAX, PP%LMAX_CALC,RHOCOL,  POTAE, DOUBLEAE)
         ENDIF

         POTAE_all(:,:,:,NIP) = POTAE(:,:,:)     ! songzd 2015/03/25

         DOUBLEC_PS= DOUBLEC_PS-DOUBLEPS*T_INFO%VCA(NT)
         ! the core-core exchange correlation energy is included
         ! up to this point in \int dr (e_xc(rho_c+rho_v) - v_xc(rho_c+rho_v) rho_v(r)
         ! subtract it now
         DOUBLEC_AE= DOUBLEC_AE+DOUBLEAE*T_INFO%VCA(NT)-PP%DEXCCORE*T_INFO%VCA(NT)

         CALL HYPERFINE_RAD(T_INFO,NI,PP,RHO,RHOAE,POTAE,COCC)
  !-----------------------------------------------------------------------
  ! calculate core level shift for averaged up and down potential
  ! or total potential (in non collinear case stored in POTAE(..,..,1))
  !-----------------------------------------------------------------------
         DO RNMAX_CL=1,RNMAX
           IF (PP%RDEP>0 .AND.  PP%R%R(RNMAX_CL)-PP%RDEP > -5E-3) EXIT
         ENDDO

         IF (.NOT. LCOREL) THEN
           CL_SHIFT(1,NI)=0
           IF (NCDIJ==2) THEN
              CALL RAD_CL_SHIFT( (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/SPI2/2, &
                      (POTAE(1:RNMAX,1,1)+POTAE(1:RNMAX,1,2))/SPI2/2, PP%R, CL_SHIFT(1,NI), PP%AUG(:,0))
           ELSE
              CALL RAD_CL_SHIFT( POT(1:RNMAX,1,1)/SPI2, POTAE(1:RNMAX,1,1)/SPI2, PP%R, CL_SHIFT(1,NI), PP%AUG(:,0))
           ENDIF
         ELSE

           CALL CL_INIT_CORE_CONF(PP,MAXNL)
           ALLOCATE( W(RNMAX,MAXNL), N(MAXNL), LC(MAXNL), EIG(MAXNL))

           ! first version for cl-shifts
           ! ---------------------------
           ! calculate core wavefunctions in PAW sphere for atomic reference potential

           CL_SHIFT(1:MAXNL,NI) =0
           CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
             W, N, LC, EIG)
           IF (MAXNL > SIZE(CL_SHIFT,DIM=1)) THEN
              CALL vtutor%bug("CL_MAXNL too small (cl_shift.F) " // str(MAXNL) // " " &
                 // str(SIZE(CL_SHIFT,DIM=1)),__FILE__,__LINE__)
           ENDIF

           ! now calculate the first order change caused by the current potential
           ! and subtract the pseudo contribution
           IF (NCDIJ==2) THEN
             CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/2/SPI2 , &
                   PP%R, CL_SHIFT(:,NI), &
                   PP%AUG(:,0),  MAXNL, W, EIG, (POTAE(:,1,1)+POTAE(:,1,2))/SPI2/2)
           ELSE
             CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ POT(1:RNMAX,1,1)/SPI2 , &
                   PP%R, CL_SHIFT(:,NI), &
                   PP%AUG(:,0),  MAXNL, W, EIG, POTAE(:,1,1)/SPI2)
           ENDIF
          !  WRITE(0,*) CL_SHIFT(1:MAXNL,NI)

          ! CL_SHIFT(1:MAXNL,NI) =0
           ! version for cl-shifts that uses exact potential
           ! ------------------------------------------------
           ! solve radial Schroedinger equation for *current* potential
          ! IF (NCDIJ==2) THEN
          !    CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
          !    W, N, LC, EIG, (POTAE(1:RNMAX,1,1)+POTAE(1:RNMAX,1,2))/2 , NMAX= RNMAX_CL)

           ! subtract only the pseudo contribution
          !    CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/2/SPI2 , &
          !         PP%R, CL_SHIFT(:,NI), &
          !         PP%AUG(:,0),  MAXNL, W, EIG)
          ! ELSE
          !    CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
          !    W, N, LC, EIG, POTAE(1:RNMAX,1,1),  NMAX= RNMAX_CL)

           ! subtract only the pseudo contribution
          !    CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ POT(1:RNMAX,1,1)/SPI2 , &
          !         PP%R, CL_SHIFT(:,NI), &
          !         PP%AUG(:,0),  MAXNL, W, EIG)
          ! ENDIF
           ! WRITE(0,*)  CL_SHIFT(1:MAXNL,NI)
           DEALLOCATE(W, N, LC, EIG)
           CALL CL_CLEAR_CORE_CONF
         ENDIF

  !-----------------------------------------------------------------------
  ! calculate the PAW correction terms to the pseudopotential strength D
  ! I have defined the PAW contribution in a way that in the limit of
  ! atomic occupancies no contributions are added
  !-----------------------------------------------------------------------
! embedding__
         IF (EXTPT_LEXTPOT()) CALL EXTPT_EXTERNAL_POT_ADD_PAW(POTAE, POT, NDIM, NCDIJ, LMMAX, NIP)
! embedding__
         ! multiply potentials by simpson weights
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POTAE)
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POT)
#ifdef noAugXCmeta
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POTH)
#endif
         CTMP=0
         DO ISP=1,NCDIJ
            DDLM=0
            CALL RAD_PROJ(  POTAE(:,:,ISP), PP%R, 1._q, DDLM, PP%LMAX, PP%LPS, PP%WAE )
            CALL RAD_PROJ(  POT(:,:,ISP)  , PP%R,-1._q, DDLM, PP%LMAX, PP%LPS, PP%WPS )
#ifndef noAugXCmeta
            CALL RAD_AUG_PROJ( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                  LYMAX, PP%AUG, PP%QPAW )
#else
            IF (LDO_METAGGA()) THEN
               CALL RAD_AUG_PROJ( POTH(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                     LYMAX, PP%AUG, PP%QPAW )
            ELSE
               CALL RAD_AUG_PROJ( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                     LYMAX, PP%AUG, PP%QPAW )
            ENDIF
#endif
            IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
               CALL RAD_AUG_PROJ_FOCK( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                    LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK )
            ENDIF
         ! transform them using Clebsch Gordan coefficients and add to CDIJ
            CALL TRANS_DLM( CTMP(:,:,ISP), DDLM , PP )
         ENDDO


         CMETAGGA=0
         CALL RAD_PROJ_METAGGA(PP,PP%WAE,LMAX_TAU,MUAE, 1._q,CMETAGGA)
         CALL RAD_PROJ_METAGGA(PP,PP%WPS,LMAX_TAU,MUPS,-1._q,CMETAGGA)

         ! non-collinear case: strength parameters need to go to the spinor presentation now
         IF (WDES%LNONCOLLINEAR) THEN
            CALL DIJ_FLIP(CTMP,LMDIM)
            CALL DIJ_FLIP(CMETAGGA,LMDIM)
         ENDIF

         IF (USELDApU() .OR. USEFOCK_CONTRIBUTION() .OR. USEFOCK_AE_ONECENTER()) THEN
            IF (WDES%ISPIN==1.AND.(.NOT.WDES%LNONCOLLINEAR)) THEN
               COCC(:,:,1)=COCC(:,:,1)*0.5_q
               ! LDA+U requires up and down density
               COCC(:,:,2)=COCC(:,:,1)
            ENDIF

          !---------------------------------------------------------------
            IF (USEFOCK_AE_ONECENTER()) THEN
               CALL SETUP_PAWFOCK_AE(NT, PP)  ! this initializes AE part only
               CALL CALC_PAWFOCK(NT, PP, COCC, CHF, DOUBLEC_HF)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_HF*T_INFO%VCA(NT)
            ELSE IF (USEFOCK_CONTRIBUTION()) THEN
               CALL CALC_PAWFOCK(NT, PP, COCC, CHF, DOUBLEC_HF)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_HF*T_INFO%VCA(NT)
            ENDIF

           ! correction terms from LDA+U
            IF (USELDApU()) THEN
          !---------------------------------------------------------------
               CALL LDAPLUSU(LMDIM, SIZE(COCC,3), NI, NT, COCC, CTMP, PP, DOUBLEC_LDAU)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_LDAU*T_INFO%VCA(NT)
            ENDIF
         ENDIF

         IF (LCALC_ORBITAL_MOMENT().AND.WDES%LNONCOLLINEAR) THEN
            COCC=CRHODE(:,:,NIP,:)
            CALL OCC_FLIP4(COCC,LMDIM) ! go to spinor representation
            CALL CALC_ORBITAL_MOMENT(LMDIM, SIZE(COCC,3), NI, NT, COCC, PP, 0._q, 0._q)
         ENDIF

         IF (WDES%LSORBIT) THEN
            COCC=CRHODE(:,:,NIP,:)
            CALL OCC_FLIP4(COCC,LMDIM) ! go to spinor representation
            CALL CALC_SPINORB_MATRIX_ELEMENTS(WDES,PP,T_INFO,NI,CSO,COCC)
         ENDIF

         CDIJ(:,:,NIP,:)=CDIJ(:,:,NIP,:)+(CTMP(:,:,1:NCDIJ)+CSO+CHF+CMETAGGA)*T_INFO%VCA(NT)

         DEALLOCATE(TAUAE,TAUPS,MUAE,MUPS,KINDENSCOL)
         CALL UNSET_RSGF_TYPE
         ! we can deallocate the calculated range-separated Greens functions here,
         ! which means that these kernels are recalculated for each ion
#ifndef _OPENMP
         CALL DEALLOCATE_RSGF
#endif
      ENDDO ion
!$OMP END PARALLEL
      ! or we could deallocate here, which means that the Greens functions are
      ! only thrown away after SET_DD_PAW has finished completely
!$    CALL DEALLOCATE_RSGF
!=======================================================================
! now distribute the DIJ to all nodes which hold DIJ (using global sum)
!=======================================================================
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ))
#else
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ*2))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ*2))
#endif
      CALLMPI( M_sum_d(WDES%COMM, CL_SHIFT, SIZE(CL_SHIFT)))

      CALLMPI( M_sum_d(WDES%COMM, DOUBLEC_AE, 1))
      CALLMPI( M_sum_d(WDES%COMM, DOUBLEC_PS, 1))
      CALLMPI( M_sum_d(WDES%COMM, EPAWPSG, 1))
      CALLMPI( M_sum_d(WDES%COMM, EPAWAEG, 1))
      CALLMPI( M_sum_d(WDES%COMM, EPAWCORE, 1))

      IF (LscMBJ()) THEN
         CALLMPI( M_sum_d(WDES%COMM, GRDRSUM, 1))
      ENDIF

#ifdef debug
      DO K=1,WDES%COMM%NCPU
      IF (WDES%COMM%NODE_ME == K) THEN
         DO NI=1,WDES%NIONS
            CALL DUMP_DLLMM( "CDIJ",CDIJ(:,:,NI,1), PP)
            IF (ISPIN==2) CALL DUMP_DLLMM( "CDIJ",CDIJ(:,:,NI,2), PP)
         ENDDO
         WRITE(*,*)
      ENDIF
      CALL MPI_barrier( WDES%COMM%MPI_COMM, k )
      ENDDO
#endif
      DEALLOCATE(RHOCOL)
#ifdef noAugXCmeta
      DEALLOCATE(RHOPS_STORE,POTH)
#endif
      DEALLOCATE(POTAE, RHOAE, POT, RHO, DRHOCORE )
      DEALLOCATE(DDLM, RHOLM, RHOLM_)
      DEALLOCATE(CTMP,CSO,CHF,CMETAGGA)
      DEALLOCATE(COCC,COCC_IM)

      E%PAWAE=DOUBLEC_AE
      E%PAWPS=DOUBLEC_PS

      E%PAWPSG  =EPAWPSG
      E%PAWAEG  =EPAWAEG
      E%PAWCORE =EPAWCORE

      GRHO_OVER_RHO_ONE_CENTER=GRDRSUM

      IF (LUSE_THOMAS_FERMI) CALL POP_XC_TYPE

      CALL RELEASE_PAWFOCK

      POP_ACC_EXEC_ON
!$ACC UPDATE DEVICE(CDIJ) IF_PRESENT IF(ACC_EXEC_ON)

      PROFILING_STOP('set_dd_paw')

    END SUBROUTINE SET_DD_PAW_song
EOF
)

main_1=$(cat<<"EOF"
      GOTO 54322
      !    
!=======================================================
! postprocess by songzd 2015
!=======================================================
     
54321 IF(IO%IU0>=0)  write(IO%IU0, '(1X," postprocess ...")')
      IF(IO%IU6>=0 .and. comm_world%ncpu /=1 ) THEN 
        write(IO%IU6, *) 'vmat can not run parallelly for now !!!!!!'
        stop 
      ENDIF
      !    
      ! recalculate potentials
      !    
      !IF (.false.) THEN

          CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES,  &
                      INFO,P,T_INFO,E,LATT_CUR, &
                      CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)

          CALL POTLOK_METAGGA(KINEDEN, &
                      GRID,GRIDC,GRID_SOFT,WDES%COMM_INTER,WDES,INFO,P,T_INFO,E,LATT_CUR, &
                      CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT_TO_C,XCSIF)

          CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                      LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

          CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                      LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)

          CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)
          !
          ! the modified SET_DD_PAW writes POTAE_all defined in sonf_data.f90,
          ! which will be used later
          !
          CALL SET_DD_PAW_song(WDES, P , T_INFO, INFO%LOVERL, &
             WDES%NCDIJ, LMDIM, CDIJ, RHOLM, CRHODE, &
             E, LMETA=.FALSE., LASPH =INFO%LASPH, LCOREL=.FALSE.)

          !CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
          !   WDES%NCDIJ, LMDIM, CDIJ, RHOLM, CRHODE, &
          !   E, LMETA=.FALSE., LASPH =INFO%LASPH, LCOREL=.FALSE.)

          CALL UPDATE_CMBJ(P,GRIDC,T_INFO,LATT_CUR,IO%IU6)
          !
      !ENDIF
      !
      !CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
      !       LMDIM,CDIJ,CQIJ, 0,SV,T_INFO,P,IO%IU0,E%EXHF)
      !
      ! pp
      !
      call vmat_top(IO, T_INFO, LATT_CUR, KPOINTS, WDES, GRID, W, LMDIM,  &
      CDIJ, CQIJ, SV, P, NONLR_S, NONL_S)
      !call vmat_top(IO,INFO, KPOINTS, WDES, NONLR_S, NONL_S, GRID, W, LMDIM, &
      !                   CDIJ, CQIJ, SV)

      IF(IO%IU0>=0) write(IO%IU0, '(1X," done")')
      stop
54322 continue
!=======================================================
! end of postprocess by songzd 2015
!=======================================================
EOF
)

asa_F=$(cat<<"EOF"
    SUBROUTINE SETYLM_XX_YLM(LYDIM,YLM_XX_YLM)
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L

!      REAL(q) YLM_NABLA_YLM((LYDIM+1)*(LYDIM+1),(LYDIM+1)*(LYDIM+1),0:3)
      REAL(q) YLM_XX_YLM(:,:,:,:)
    ! local
      INTEGER LLMAX, LMMAX, PHPTS, THPTS, NPTS
      INTEGER I, J, NP, IFAIL
      REAL(q) DELTAPHI, SIM_FAKT, TMP(3)
      REAL(q), ALLOCATABLE ::  RADPTS(:,:), XYZPTS(:,:),YLM(:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)
      INTEGER LM, LMP, L, LP, M, MP
      EXTERNAL GAUSSI2

      IF (SIZE(YLM_XX_YLM,1) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_XX_YLM,2) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_XX_YLM,3) <3 .OR. &
          SIZE(YLM_XX_YLM,4) <3 ) THEN
         WRITE(0,*)'internal ERROR: SETYLM_XX_YLM, insufficient L workspace'
         STOP
      ENDIF

      LLMAX=LYDIM
      LMMAX=(LLMAX+1)**2
! number of theta and phi pivot points
! the value below perform the integration exactly without error
! less is not possible more is not required
      PHPTS=(LLMAX+1)*2
      THPTS=FLOOR(REAL(LLMAX/2+1,KIND=q))*2
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX),XYZPTS(NPTS,3),RADPTS(NPTS,2))
      ALLOCATE(WEIGHT(THPTS),ABSCIS(THPTS))

      RADPTS=0; WEIGHT=0; ABSCIS=0
      ! set phi positions, equally spaces
      DO I=1,PHPTS
         DO J=1,THPTS
            RADPTS((J-1)*PHPTS+I,2)=(I-1)*DELTAPHI
         ENDDO
      ENDDO
     ! get theta positions (actually get cos(theta)) (Gauss integration)
      CALL GAUSSI(GAUSSI2,-1._q,1._q,0,THPTS,WEIGHT,ABSCIS,IFAIL)
      DO I=1,THPTS
         RADPTS((I-1)*PHPTS+1:I*PHPTS,1)=ABSCIS(I)
      ENDDO
      ! convert radial to cartesian coordinates
      DO I=1,NPTS
         XYZPTS(I,1)=COS(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! x
         XYZPTS(I,2)=SIN(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! y
         XYZPTS(I,3)=RADPTS(I,1)                                 ! z
      ENDDO
    ! get |r| Y_lm on a unit sphere and its derivatives
      YLM=0 ;

      !CALL SETYLM_GRAD2(LLMAX,NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))
      CALL SETYLM(LLMAX,NPTS,YLM, XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

    ! loop over all points in the angular grid

      YLM_XX_YLM=0

      points: DO NP=1,NPTS
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))
         
         DO LM=1,(LYDIM+1)*(LYDIM+1)
            DO LMP=1,(LYDIM+1)*(LYDIM+1)
               DO I=1,3
               DO j=1,3
                YLM_XX_YLM(LM,LMP,I,J)    =YLM_XX_YLM(LM,LMP,I,J)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,I)*XYZPTS(NP,J)
               ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO points
!      WRITE(0,*) 'YLM YLM'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(:,:,0)
!      WRITE(0,*) 'YLM YLM1'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,1)
!      WRITE(0,*) 'YLM YLM2'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,2)
!      WRITE(0,*) 'YLM YLM3'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,3)

      DEALLOCATE(YLM,XYZPTS,RADPTS)
      DEALLOCATE(WEIGHT,ABSCIS)
      !
#if .false.
      do L=0,LLMAX
      do LP=0,LLMAX
          do M=1,2*L+1
          do MP=1,2*LP+1
              write(*,'( 2("(",2I4,")"), F12.6 )') L,M,LP,MP, YLM_XX_YLM(L*L+M, LP*LP+MP, 1,2)
          enddo
          enddo
      enddo
      enddo
#endif
      !
    ENDSUBROUTINE SETYLM_XX_YLM
EOF
)

relativistic_F=$(cat<<"EOF"
      SUBROUTINE SET_CNABLAVIJ_(LMAX, POT, RHOC, POTVAL, R, DLLMM, CHANNELS, L, W, Z, ivec)
      USE prec
      USE constant
      USE radial
      use asa,              only : SETYLM_NABLA_YLM
      !
      IMPLICIT NONE
      !
      integer :: LMAX        ! maximum L quantum number
      REAL(q) POT(:)         ! spherical contribution to potential w.r.t reference potential
      REAL(q) RHOC(:)        ! electronic core charge
      REAL(q) POTVAL(:)      ! minus potential of atom
      TYPE(rgrid) :: R
      REAL(q) W(:,:)         ! wavefunctions phi(r,l)
      OVERLAP DLLMM(:,:)   ! contribution to H from so-coupling
      INTEGER CHANNELS, L(:) 
      REAL(q) Z              ! charge of the nucleus
      integer :: ivec        ! 1-x, 2-y, 3-z 
! local
      INTEGER I,J,LM,LMP,M,MP,CH1,CH2,LL,LLP
      REAL(q) APOT(R%NMAX)   ! average potential (up down)
      REAL(q) DPOT(R%NMAX)   ! radial derivative   of potential APOT
      REAL(q) RHOT(R%NMAX)   ! charge density
      INTEGER  LMMAX
      real(q), allocatable :: YLM_NABLA_YLM(:,:,:), YLM_X_YLM(:,:,:)
      REAL(q) SUM, SCALE
      !
      LMMAX=(LMAX+1)**2
      !
!     thats just Y_00
      SCALE=1/(2*SQRT(PI))
      !
!     unfortunately the PAW method operates usually only with
!     difference potentials (compared to isolated atom)
!     we need to evaluate a couple of terms

!     lets first calculate the Hatree potential of the core electrons
      CALL RAD_POT_HAR(0, R, APOT, RHOC, SUM)
!     add the potential of the nucleus (delta like charge Z at origin)
      APOT=APOT*SCALE - FELECT/R%R*Z
!     subtract reference potential POTVAL (previously added to POT(:,:) (see RAD_POT)
!     this one contains essentially valence only contributions
      APOT=APOT-POTVAL
!     finally add the current potential (average spin up and down)
      APOT=APOT+POT* SCALE
!     gradient
      CALL GRAD(R,APOT,DPOT)   ! now DPOT are in unit eV/A
      !
      ! prepare YLM
      !
      ALLOCATE(YLM_NABLA_YLM(LMMAX,LMMAX,0:3), YLM_X_YLM(LMMAX, LMMAX,0:3) )
      CALL SETYLM_NABLA_YLM(LMAX, YLM_NABLA_YLM, YLM_X_YLM)
      !
!     calculates the integral
!      \int dr  w_ln(r)  DPOT(r)  w_ln'(r) *\int dOmega  Y_lm x_i/|x| Y_l'm'
      !
      DLLMM(:,:)=0.0_q
      !
      LM =1
      DO CH1=1,CHANNELS
      LMP=1
      DO CH2=1,CHANNELS
        DO I=1,R%NMAX
           RHOT(I)=W(I,CH1)*W(I,CH2)
        END DO
        LL = L(CH1)
        LLP= L(CH2)
!     calculation is made only for l=0,1,2,3,4 orbitals
!     a spherical potential is assumed
          SUM=0.0_q
          DO I=1,R%NMAX 
!      The integral is made only inside the augmentation sphere
!            IF(R%R(I) <= R%RMAX) THEN
              SUM= SUM+DPOT(I)*RHOT(I)*R%SI(I)
!            ENDIF
          END DO
          SUM=SUM
!
! VASP uses a reverted notation (for efficiency reason)
!  D(lm,l'm') =  < y_l'm' | D | y_lm>  
! this is a spin diagnal operator
! therefore we need a little bit of reindexing (not too complicated)
          DO M =1,2*LL+1
          DO MP=1,2*LLP+1
             !
             if ( LM+M .le. LMP+MP ) then
                !
                if( ABS( YLM_X_YLM(LL*LL+M, LLP*LLP+MP, ivec) ) > 1E-8_q ) then
                    DLLMM(LMP+MP-1,LM+M-1)= SUM*YLM_X_YLM( LL*LL+M, LLP*LLP+MP, ivec)
                    !SUM*LS(M,MP,I+2*J+1,LL)
                endif
                !
                if( LM+M/=LMP+MP) DLLMM(LM+M-1,LMP+MP-1) =  DLLMM(LMP+MP-1,LM+M-1)
                !
             endif
             !
            !
          END DO
          END DO

      LMP=LMP+(2*LLP+1)
      ENDDO
      LM= LM+ (2*LL+1)
      ENDDO
      !
      DEALLOCATE( YLM_NABLA_YLM, YLM_X_YLM )
      !
      END SUBROUTINE SET_CNABLAVIJ_

!*******************************************************************

      SUBROUTINE SET_CRDVIJ_(LMAX, POT, RHOC, POTVAL, R, DLLMM, CHANNELS, L, W, Z, ivec, ivec_)

      USE prec
      USE constant
      USE radial
      use song_data,        only: socfactor, cfactor
      use asa,              only : SETYLM_XX_YLM
      !
      IMPLICIT NONE
      !
      integer :: LMAX        ! maximum L quantum number
      REAL(q) POT(:)         ! spherical contribution to potential w.r.t reference potential
      REAL(q) RHOC(:)        ! electronic core charge
      REAL(q) POTVAL(:)      ! minus potential of atom
      TYPE(rgrid) :: R
      REAL(q) W(:,:)         ! wavefunctions phi(r,l)
      OVERLAP DLLMM(:,:)   ! contribution to H from so-coupling
      INTEGER CHANNELS, L(:) 
      REAL(q) Z              ! charge of the nucleus
      integer :: ivec, ivec_ ! 1-x, 2-y, 3-z 
! local
      INTEGER I,J,LM,LMP,M,MP,CH1,CH2,LL,LLP
      REAL(q) APOT(R%NMAX)   ! average potential (up down)
      REAL(q) DPOT(R%NMAX)   ! radial derivative   of potential APOT
      REAL(q) RHOT(R%NMAX)   ! charge density
      INTEGER  LMMAX
      real(q), allocatable :: YLM_XX_YLM(:,:,:,:)
      REAL(q) SUM, SUM1, SCALE
      !
      LMMAX=(LMAX+1)**2
      !
!     thats just Y_00
      SCALE=1/(2*SQRT(PI))
      !
!     unfortunately the PAW method operates usually only with
!     difference potentials (compared to isolated atom)
!     we need to evaluate a couple of terms

!     lets first calculate the Hatree potential of the core electrons
      CALL RAD_POT_HAR(0, R, APOT, RHOC, SUM)
!     add the potential of the nucleus (delta like charge Z at origin)
      APOT=APOT*SCALE - FELECT/R%R*Z
!     subtract reference potential POTVAL (previously added to POT(:,:) (see RAD_POT)
!     this one contains essentially valence only contributions
      APOT=APOT-POTVAL
!     finally add the current potential (average spin up and down)
      APOT=APOT+POT* SCALE
!     gradient
      CALL GRAD(R,APOT,DPOT)   ! now DPOT are in unit eV/A
      !
      ! prepare YLM
      !
      ALLOCATE(YLM_XX_YLM(LMMAX, LMMAX,3,3) )
      CALL SETYLM_XX_YLM(LMAX, YLM_XX_YLM)
      !
!     calculates the integral
!      \int dr  w_ln(r)*r*DPOT(r)*w_ln'(r) * \int dOmega  Y_lm x_i/|x| Y_l'm'
      !
      DLLMM(:,:)=0.0_q
      !
      LM =1
      DO CH1=1,CHANNELS
      LMP=1
      DO CH2=1,CHANNELS
        DO I=1,R%NMAX
           RHOT(I)=W(I,CH1)*W(I,CH2)
        END DO
        LL = L(CH1)
        LLP= L(CH2)
!     calculation is made only for l=0,1,2,3,4 orbitals
!     a spherical potential is assumed
          SUM=0.0_q
          DO I=1,R%NMAX 
!      The integral is made only inside the augmentation sphere
!            IF(R%R(I) <= R%RMAX) THEN
              SUM= SUM+DPOT(I)*RHOT(I)*R%R(I)*R%SI(I)
!            ENDIF
          END DO
          SUM=SUM           ! now SUM in unit of eV
#if .false.
          SUM1=0.0_q
          DO I=1,R%NMAX
            SUM1=SUM1+DPOT(I)*RHOT(I)/R%R(I)*R%SI(I)  ! SUM1 in unit of eV/A^2
          ENDDO
          !! compare these numbers in atomic units
          write(*,*) SUM/2/RYTOEV,  SUM1/2/RYTOEV*AUTOA*AUTOA   
          !
#endif
!
! VASP uses a reverted notation (for efficiency reason)
!  D(lm,l'm') =  < y_l'm' | D | y_lm>  
! this is a spin diagnal operator
! therefore we need a little bit of reindexing (not too complicated)
          DO M =1,2*LL+1
          DO MP=1,2*LLP+1
             !
             if( ABS( YLM_XX_YLM(LL*LL+M, LLP*LLP+MP, ivec, ivec_) ) > 1E-8_q ) then
                DLLMM(LMP+MP-1,LM+M-1)= SUM*YLM_XX_YLM( LL*LL+M, LLP*LLP+MP, ivec,ivec_)
                !SUM*LS(M,MP,I+2*J+1,LL)
             endif
             !
            !
          END DO
          END DO

      LMP=LMP+(2*LLP+1)
      ENDDO
      LM= LM+ (2*LL+1)
      ENDDO
      !
      DEALLOCATE( YLM_XX_YLM )
      !
      END SUBROUTINE SET_CRDVIJ_
!**********************************************************************
EOF
)



##############################################################################
# begin

# judge if vasp2mat.6.4 already exists
if [ -d "vasp2mat.6.4" ]; then
    echo "The file folder 'vasp2mat.6.4' already exists, if you continue the installation, this folder will be deleted. Do you want to continue? [y/n]"
    read flag
    while [ 0 ]; do
        if [ "$flag" == "n" ] || [ "$flag" == "N" ]; then
            exit
        elif [ "$flag" == "y" ] || [ "$flag" == "Y" ]; then
            rm -rf vasp2mat.6.4
            break
        else
            read flag
        fi
    done
fi

# prepare the files
mkdir vasp2mat.6.4
cd vasp2mat.6.4

mkdir bin
mkdir build
cp ../vasp.6.4/makefile.include .
cp -r ../vasp.6.4/src .

# revise main.F
cd src
line_number=$(grep -n "USE hyperfine" "main.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
#line_number=$(expr $line_number + 1)
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in main.F!!!"
    exit
fi
sed -i "${line_number}r /dev/stdin" main.F <<<"      USE song_data,  only: song_getinp, vmat  ! added by songzd 2015/03/18  "
line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" main.F <<<"      USE song_vmat,  only: vmat_top           ! added by songzd 2015/03/18 "

line_number=$(grep -n "open Files" "main.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')

if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in main.F!!!"
    exit
fi

line_number=$(expr $line_number  + 1)
sed -i "${line_number}r /dev/stdin" main.F <<<"      call song_getinp(IO)      ! added by songzd 2015/03/18"

line_number=$(grep -n "IF ((.NOT. LJ_ONLY .AND. .NOT. LCHI) .AND. (LDO_AB_INITIO)) THEN" "main.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in main.F!!!"
    exit
fi

#line_number=$(expr $line_number - 2)
sed -i "${line_number}r /dev/stdin" main.F <<<"      IF (vmat==-1) GOTO 54322      ! calculations using readed wavefunction, like wannier"
line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" main.F <<<"      IF (vmat/=0) GOTO 54321       ! by songzd"

line_number=$(grep -n 'CALL STOP_TIMING("G",IO%IU6,"ORTHCH",XMLTAG="orth")' "main.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in main.F!!!"
    exit
fi

line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" main.F <<< "$main_1"

# revise paw.F
line_number=$(grep -n "END MODULE pawm" "paw.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')

if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in paw.F!!!"
    exit
fi

line_number=$(expr $line_number - 2)

sed -i "${line_number}r /dev/stdin" paw.F <<< "$paw_F"

# revise asa.F
line_number=$(grep -n "END SUBROUTINE SETYLM_NABLA_YLM" "asa.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in asa.F!!!"
    exit
fi

line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" asa.F <<< "$asa_F"

# revise relativistic.F

line_number=$(grep -n "MODULE RELATIVISTIC" "relativistic.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in main.F!!!"
    exit
fi

line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" relativistic.F <<< "      USE constant, only : AUTOA, RYTOEV"

line_number=$(grep -n "USE radial" "relativistic.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in relativistic.F!!!"
    exit
fi
sed -i "${line_number}r /dev/stdin" relativistic.F <<< "      use song_data,        only: socfactor, cfactor"

line_number=$(grep -n "INTEGER, PARAMETER :: LMAX=3, MMAX=LMAX\*2+1" "relativistic.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p') # notice
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in relativistic.F!!!"
    exit
fi

sed -i "${line_number}r /dev/stdin" relativistic.F <<< "      real(q) :: C_, INVMC2_ ! songzd"

line_number=$(grep -n "KSI(:)" "relativistic.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in relativistic.F!!!"
    exit
fi


sed -i "${line_number}r /dev/stdin" relativistic.F <<< "      INVMC2_ = INVMC2/cfactor/cfactor*socfactor"
line_number=$(expr $line_number + 1)
sed -i "${line_number}r /dev/stdin" relativistic.F <<< "      C_ = CLIGHT*cfactor"
line_number=$(expr $line_number + 8)

sed "${line_number}s/CLIGHT/C_/g" relativistic.F > t && mv t relativistic.F
sed "${line_number}s/INVMC2/INVMC2_/g" relativistic.F > t && mv t relativistic.F

line_number=$(grep -n "END SUBROUTINE SPINORB_STRENGTH" "relativistic.F" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in relativistic.F!!!"
    exit
fi

line_number=$(expr $line_number + 3)
sed -i "${line_number}r /dev/stdin" relativistic.F <<< "$relativistic_F"

# create song_vmat.F
echo "$song_vmat_F" >> song_vmat.F

# create song_data.F
echo "$song_data_F" >> song_data.F

# revise .object
line_number=$(grep -n "constant.o" ".objects" | grep -v 'ml_ff_constant.o'| sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in .objects!!!"
    exit
fi


line_number=$(expr $line_number - 1)
sed -i "${line_number}r /dev/stdin" .objects <<< "        song_data.o \\"

line_number=$(grep -n "rpa_high.o" ".objects" | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in .objects!!!"
    exit
fi

sed -i "${line_number}r /dev/stdin" .objects <<< "        song_vmat.o"
sed "${line_number}s/$/ \\\/" .objects > temp.txt && mv temp.txt .objects

cd ..

# revise makefile
if [ -e "../vasp.6.4/makefile" ]; then
  makefile="makefile"
elif [ -e "../vasp.6.4/Makefile" ]; then
  makefile="Makefile"
else
  echo "File 'Makefile' not exists in vasp.6.4!"
  exit
fi

cp  ../vasp.6.4/$makefile ./


line_number=$(grep -n -E '^[[:space:]]*VERSION' $makefile | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in ${makefile}!!!"
    exit
fi

sed -i "${line_number}s/.*/VERSIONS = ncl/" $makefile

line_number=$(grep -n -E '^[[:space:]]*all[[:space:]]*:' $makefile | sed -n 's/^\([0-9]\+\):.*$/\1/p')
if [ -z "$line_number" ]; then
    echo "Error: cannot find the right line in ${makefile}!!!"
    exit
fi

sed -i "${line_number}s/.*/all: ncl/" $makefile

echo "Finishing modifying vasp into vasp2mat!"
echo "vasp2mat source file is in the file folder vasp2mat.6.4, do you want to compile it right now? [y/n]"

# ask if the user want to compile vasp2mat right now
read flag_2
while [ 0 ]; do
    if [ "$flag_2" == "n" ] || [ "$flag_2" == "N" ]; then
        cd ..
        exit
    elif [ "$flag_2" == "y" ] || [ "$flag_2" == "Y" ]; then
        echo "Begin compiling vasp2mat!"
        break
    else
        read flag_2
    fi
done

# begin compiling
make

# rename vasp to vasp2mat
cd build/ncl
mv vasp vasp2mat
cd ../../bin
mv vasp_ncl vasp2mat

# path of the file folder of vasp2mat
current_path=$(pwd)

cd ../..

# print the path
echo " "
echo "Finishing installing vasp2mat!"
echo "The path of vasp2mat: $current_path/vasp2mat"
