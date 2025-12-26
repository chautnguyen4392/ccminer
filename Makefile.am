# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS = compat

bin_PROGRAMS = ccminer

# Simplified sources - only scrypt-jane algorithm support
ccminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  ccminer.cpp pools.cpp util.cpp bench.cpp bignum.cpp \
			  api.cpp hashlog.cpp nvml.cpp stats.cpp sysinfos.cpp cuda.cpp \
			  nvsettings.cpp \
			  cuda_checkhash.cu \
			  sph/sha2.c \
			  scrypt-jane.cpp \
			  scrypt/blake.cu scrypt/keccak.cu scrypt/sha256.cu \
			  scrypt/salsa_kernel.cu scrypt/test_kernel.cu \
			  scrypt/fermi_kernel.cu scrypt/kepler_kernel.cu \
			  scrypt/nv_kernel.cu scrypt/nv_kernel2.cu scrypt/titan_kernel.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

if ARCH_ARM64
ccminer_CPPFLAGS += -DARM64
ccminer_LDFLAGS += -L@CUDA_LOCAL_PATH@/targets/aarch64-linux/lib
endif

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

ccminer_LDADD += -lcuda

# Default architecture (Pascal - GTX 1060/1070/1080)
nvcc_ARCH := -gencode=arch=compute_61,code=\"sm_61,compute_61\"

# Optional: Turing (RTX 2000 series) - requires CUDA 10+
if WANT_GEN75
nvcc_ARCH += -gencode=arch=compute_75,code=\"sm_75,compute_75\"
endif

# Optional: Ampere (RTX 3000 series) - requires CUDA 11+
if WANT_GEN86
nvcc_ARCH += -gencode=arch=compute_86,code=\"sm_86,compute_86\"
endif

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

# Scrypt kernel build rules
scrypt/salsa_kernel.o: scrypt/salsa_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

scrypt/test_kernel.o: scrypt/test_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

scrypt/fermi_kernel.o: scrypt/fermi_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

scrypt/kepler_kernel.o: scrypt/kepler_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

scrypt/nv_kernel.o: scrypt/nv_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

scrypt/titan_kernel.o: scrypt/titan_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

# Custom target to clean only scrypt-related object files for faster rebuilds
clean-scrypt:
	rm -f scrypt/*.o scrypt-jane.o
