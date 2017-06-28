module jacobi_mod
  use mpi
  implicit none

  ! Boundary value at the LHS of the pipe
  real(kind=8), parameter :: LEFT_VALUE = 1.0
  ! Boundary value at the RHS of the pipe
  real(kind=8), parameter :: RIGHT_VALUE = 10.0
  ! The convergence to terminate at
  real(kind=8), parameter :: CONVERGENCE_ACCURACY = 1e-4
  ! The maximum number of iterations
  integer, parameter :: MAX_ITERATIONS = 500000
  ! How often to report the norm
  integer, parameter :: REPORT_NORM_PERIOD = 1000

contains

  subroutine run_solver()
    real(kind=8), dimension(:,:), allocatable :: u_k, u_kp1, temp
    real(kind=8) :: bnorm, rnorm, norm, tmpnorm
    integer :: i, j, k, ierr, size, myrank, local_nx, nx, ny, requests(4)
    character(len=32) :: arg
    double precision :: start_time
    
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, myrank, ierr)
    start_time=MPI_Wtime()

    if (command_argument_count() /= 2) then
      print *, &
           "You must provide two command line arguments, the global size in X and the global size in Y"
      return
    end if

    call get_command_argument(1, arg)
    read(arg,*) nx
    call get_command_argument(2, arg)
    read(arg,*) ny

    if (myrank==0) then
      print *, "Global size in X=", nx, "Global size in Y=", ny
    end if
    
    local_nx=nx/size
    if (local_nx * size .lt. nx) then
      if (myrank .lt. nx - local_nx * size) local_nx=local_nx+1
    end if

    allocate(u_k(ny, 0:local_nx+1), u_kp1(ny, 0:local_nx+1), temp(ny, 0:local_nx+1))

    bnorm=0.0
    tmpnorm=0.0
    rnorm=0.0
    
    call initialise_values(u_k, u_kp1, local_nx, ny, myrank, size)

    do j=1, local_nx
      do i=1, ny
        tmpnorm=tmpnorm+((u_k(i,j)*4-u_k(i-1,j)-u_k(i+1,j)-u_k(i,j-1)-u_k(i,j+1)) ** 2)
      end do
    end do
    call mpi_allreduce(tmpnorm, bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
    bnorm=sqrt(bnorm)

    requests=MPI_REQUEST_NULL

    do k=0, MAX_ITERATIONS
      if (myrank .gt. 0) then
         call mpi_isend(u_k(1,1), ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(1), ierr)
         call mpi_irecv(u_k(1,0), ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(2), ierr)
      end if
      if (myrank .lt. size-1) then
         call mpi_isend(u_k(1,local_nx), ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(3), ierr)
         call mpi_irecv(u_k(1,local_nx+1), ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(4), ierr)
      end if
      call mpi_waitall(4, requests, MPI_STATUSES_IGNORE, ierr)
      tmpnorm=0.0
      do j=1, local_nx
        do i=1, ny
          tmpnorm=tmpnorm+((u_k(i,j)*4-u_k(i-1,j)-u_k(i+1,j)-u_k(i,j-1)-u_k(i,j+1)) ** 2)
        end do      
      end do
      call mpi_allreduce(tmpnorm, rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
      norm=sqrt(rnorm)/bnorm
      if (norm .lt. CONVERGENCE_ACCURACY) exit

      do j=1, local_nx
        do i=1, ny
          u_kp1(i,j)=0.25 * (u_k(i-1,j) + u_k(i+1,j) + u_k(i,j-1) + u_k(i,j+1))
        end do
      end do
      temp=u_kp1
      u_kp1=u_k
      u_k=temp

      if (mod(k, REPORT_NORM_PERIOD)==0 .and. myrank==0) print *, "Iteration=",k," Relative Norm=",norm
    end do
    if (myrank==0) print *, "Terminated on ",k," iterations, Relative Norm=", norm, " runtime=",MPI_Wtime()-start_time," sec"
    deallocate(u_k, u_kp1, temp)
    call mpi_finalize(ierr)
  end subroutine run_solver

  ! Initialises the arrays, such that u_k contains the boundary conditions at the start and end points and all other
  ! points are zero. u_kp1 is set to equal u_k
  subroutine initialise_values(u_k, u_kp1, local_nx, ny, myrank, size)
    real(kind=8), intent(inout) :: u_k(ny, 0:local_nx+1), u_kp1(ny, 0:local_nx+1)
    integer, intent(in) :: local_nx, ny, myrank, size

    u_k(:,0)=merge(LEFT_VALUE, 0.0_8, myrank==0)
    u_k(:,local_nx+1)=merge(RIGHT_VALUE, 0.0_8, myrank==size-1)
    u_k(:,1:local_nx)=0.0_8
    u_kp1=u_k
  end subroutine initialise_values  
end module jacobi_mod

program jacobi
  use jacobi_mod
  implicit none

  call run_solver()
end program jacobi
