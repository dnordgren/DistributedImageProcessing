CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
MPIFLAGS = -I/usr/include/openmpi-x86_64 -pthread -Wl,-rpath -Wl,/usr/lib64/openmpi/lib -Wl,--enable-new-dtags -L/usr/lib64/openmpi/lib -lmpi_cxx -lmpi

% : %.cpp
	g++ $(CFLAGS) $(MPIFLAGS) -o $@.out $< $(LIBS)
