LFLAGS=-lm
FTN=ftn
FFLAGS=-O3

all: 
	$(FTN) $(FFLAGS) -o jacobi jacobi.F90 $(LFLAGS)

clean:
	rm -f jacobi *.o *.mod
