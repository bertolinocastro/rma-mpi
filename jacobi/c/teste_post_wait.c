#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int vars[2];

    int rank, size;
    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if(size != 2) return -1;

    MPI_Win win;
    MPI_Group worldGroup, subGroup;
    int ranks[] = { 0, 1 };

    MPI_Comm_group( MPI_COMM_WORLD, &worldGroup );

    if( rank ){
        MPI_Win_create(
            vars, sizeof(int) * 2, sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD,
            &win
        );
    }else{
        MPI_Win_create(
            NULL, 0, sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD,
            &win
        );
    }





    if( !rank ){
        vars[0] = 1, vars[1] = 2;

        MPI_Group_incl( worldGroup, 1, &ranks[1], &subGroup );

        MPI_Win_start( subGroup, 0, win );

        MPI_Put(
            vars, 2, MPI_INT,
            ranks[1], 0, 2, MPI_INT,
            win
        );

        MPI_Win_complete( win );

    }else{

        vars[0] = 3, vars[1] = 4;

        MPI_Group_incl( worldGroup, 1, &ranks[0], &subGroup );



        MPI_Win_post( subGroup, 0, win );

        MPI_Win_wait( win );
    }



    printf( "Oi, sou o proc %d e meu vars[0]=%d e vars[1]=%d\n", rank, vars[0], vars[1] );

    MPI_Win_free( &win );
    MPI_Group_free( &worldGroup );
    MPI_Group_free( &subGroup );

    MPI_Finalize();
    return 0;
}
