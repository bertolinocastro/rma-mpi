#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Boundary value at the top of the domain
#define TOP_VALUES 1.0
// Boundary value at the bottom of the domain
#define BOTTOM_VALUES 10.0
// The maximum number of iterations
#define MAX_ITERATIONS 5000000
// The convergence to terminate at
#define CONVERGENCE_ACCURACY 1e-4
// How often to report the norm
#define REPORT_NORM_PERIOD 1000

int nx, ny;

void initialise( double*, double*, int, int, int );

int main(int argc, char **argv){

    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if( argc != 3 ){
	   if(!rank) fprintf(stderr, "You must provide the size in X and in Y as arguments to this code.\n");
	   return -1;
    }

    nx = atoi( argv[1] );
    ny = atoi( argv[2] );

    if(!rank) printf( "Solving to accuracy of %.0e, global system size is x=%d y=%d\n", CONVERGENCE_ACCURACY, nx, ny );
// printf("Rank %d size %d\n",rank, size );
    int lnx = nx/size; // Computing local nx
    if( lnx*size < nx )
	   if( rank < nx - lnx  * size ) lnx++;

    double * u_k = malloc( (lnx + 2)*ny * sizeof(*u_k) );
    double * u_kp1 = malloc( (lnx + 2)*ny * sizeof(*u_kp1) );
    double * tmp = malloc( (lnx + 2)*ny * sizeof(*tmp) );
    double start_time, end_time;

    MPI_Win ghostWinL, ghostWinR, normWin;
    int leftRank, rightRank;
    double *leftBound, *rightBound; long leftSize, rightSize;

    // Left Rank and Border's address and size
    leftRank = rank ? rank - 1 : MPI_PROC_NULL;
    // leftBound = rank ? &u_k[ny] : NULL; // Exposing truecells
    leftBound = rank ? &u_k[0] : NULL; // Exposing ghostcells
    leftSize = rank ? ny*sizeof(double) : 0;

    // Right Rank and Border's address and size
    rightRank = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    // rightBound = rank < size - 1 ? &u_k[ny*lnx] : NULL; // Exposing truecells
    rightBound = rank < size - 1 ? &u_k[ny*(lnx+1)] : NULL; // Exposing ghostcells
    rightSize = rank < size - 1 ? ny*sizeof(double) : 0;

    // Criando janela para as truecells à esquerda
    MPI_Win_create( leftBound, leftSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghostWinL );
    // Criando janela para as true cells à direita
    MPI_Win_create( rightBound, rightSize, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghostWinR );


    // MPI_Win_create( &normWin, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &normWin ); // Criando janela para a variavel de norma

    initialise( u_k, u_kp1, lnx, rank, size );
    // return 0;

    double rnorm = 0.0f, bnorm = 0.0f, norm, tmpnorm = 0.0f;
    MPI_Request requests[] = {MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};

    int i,j,k;

    for( i = 1; i <= lnx; ++i )
        for( j = 0; j < ny; ++j )
            tmpnorm +=  pow(
			    u_k[j   + i	    *ny]*4  -
			    u_k[j-1 + i	    *ny]    -
			    u_k[j+1 + i	    *ny]    -
			    u_k[j   + (i-1) *ny]    -
			    u_k[j   + (i+1) *ny],
			    2
            );

    MPI_Allreduce( &tmpnorm, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    bnorm = sqrt( bnorm );


    start_time = MPI_Wtime();

    for( k = 0; k < MAX_ITERATIONS; ++k ){

        // // Populando as próprias ghostcells à esquerda com as truecells à direita dos leftRank
        // MPI_Win_fence( 0, ghostWinR );
        //     MPI_Put( &u_k[ny], ny, MPI_DOUBLE,
        //              leftRank, 0, ny, MPI_DOUBLE,
        //              ghostWinR );
        //     // MPI_Get(
        //     //     &u_k[0], ny, MPI_DOUBLE,
        //     //     leftRank, 0, ny, MPI_DOUBLE,
        //     //     ghostWinR
        //     // );
        // MPI_Win_fence( 0, ghostWinR );
        //
        // // printf("Comuniquei! %d winR\n", rank);
        //
        // // Populando as próprias ghostcells à direita com as truecells à esquerda dos rightRank
        // MPI_Win_fence( 0, ghostWinL );
        //     MPI_Put( &u_k[ny*lnx], ny, MPI_DOUBLE,
        //          rightRank, 0, ny, MPI_DOUBLE,
        //          ghostWinL );
        //     // MPI_Get(
        //     //     &u_k[ny*(lnx+1)], ny, MPI_DOUBLE,
        //     //     rightRank, 0, ny, MPI_DOUBLE,
        //     //     ghostWinL
        //     // );
        // MPI_Win_fence( 0, ghostWinL );

        // printf("Comuniquei! %d winL\n", rank);




        if( rank )
    	    MPI_Isend( &u_k[ny], ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[0] ),
    	    MPI_Irecv( &u_k[0], ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[1] );
    	if( rank < size - 1 )
    	    MPI_Isend( &u_k[lnx*ny], ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[2] ),
    	    MPI_Irecv( &u_k[(lnx+1)*ny], ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[3] );

    	MPI_Waitall( 4, requests, MPI_STATUSES_IGNORE );

    	tmpnorm = 0.0f;
    	for( i = 1; i <= lnx; ++i )
    	    for( j = 0; j < ny; ++j )
    		tmpnorm +=  pow(
    				u_k[j   + i	    *ny]*4  -
    				u_k[j-1 + i	    *ny]    -
    				u_k[j+1 + i	    *ny]    -
    				u_k[j   + (i-1) *ny]	    -
    				u_k[j   + (i+1) *ny],
    				2
    			    );

    	MPI_Allreduce( &tmpnorm, &rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    	norm = sqrt( rnorm )/bnorm;

    	if( norm < CONVERGENCE_ACCURACY ) break;

    	for( i = 1; i <= lnx; ++i )
    	    for( j = 0; j < ny; ++j )
    		u_kp1[j + i*ny] =   0.25f * (
    				    u_k[(j-1)	+ i*ny]	    +
    				    u_k[(j+1)	+ i*ny]	    +
    				    u_k[j	+ (i-1)*ny] +
    				    u_k[j	+ (i+1)*ny]
    				    );

    	memcpy( tmp, u_kp1, ny * (lnx + 2) * sizeof(double) );
    	memcpy( u_kp1, u_k, ny * (lnx + 2) * sizeof(double) );
    	memcpy( u_k,   tmp, ny * (lnx + 2) * sizeof(double) );

    	if( !(k%REPORT_NORM_PERIOD) && !rank ) printf( "Iteration=%d Relative Norm=%e\n", k, norm);

    }
    end_time = MPI_Wtime();

    if( !rank ) printf( "\nTerminated on %d iterations, Relative Norm=%e, Total time=%e s\n", k, norm, end_time - start_time );

    free( u_k );
    free( u_kp1 );
    free( tmp );

    MPI_Finalize();

    return 0;
}

void initialise( double *u_k, double *u_kp1, int lnx, int rank, int size ){
    int i, j, llnx = lnx + 1;

    for( j = 0; j < ny; ++j ){
	    u_kp1[j] = u_k[j] = !rank ? TOP_VALUES : 0;
        //for( j = 0; j < ny; ++j )
        u_kp1[j+ny*llnx] = u_k[j+ny*llnx] = rank == size-1 ? BOTTOM_VALUES : 0;

        for( i = 1; i <= lnx; ++i )
	       u_kp1[j+i*ny] = u_k[j+i*ny] = 0;
    }
}
