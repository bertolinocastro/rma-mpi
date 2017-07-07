#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int rank, size;
    int k;
    MPI_Win win;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if( size != 2 )
	return -1;

    k = !rank ? 1 : 2;

    MPI_Win_create( &k, sizeof(k), sizeof(k), MPI_INFO_NULL, MPI_COMM_WORLD, &win );

    if( !rank ){
	MPI_Win_lock( MPI_LOCK_EXCLUSIVE, 1, 0, win  );

	MPI_Put(
	    &k, 1, MPI_INT,
	    1, 0, 1, MPI_INT,
	    win
	);

	MPI_Win_unlock( 1, win );

    }
    MPI_Barrier( MPI_COMM_WORLD );

    printf( "Oi, sou o proc %d e meu k vale: %d\n", rank, k );

    MPI_Finalize();
    return 0;
}
