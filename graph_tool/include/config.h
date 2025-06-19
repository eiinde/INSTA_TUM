/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* program author(s) */
#define AUTHOR "Tiago de Paula Peixoto <tiago@skewed.de>"

/* Suppress Boost warning */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS 1

/* Stack size in bytes */
#define BOOST_COROUTINE_STACK_SIZE 5242880

/* copyright info */
#define COPYRIGHT "Copyright (C) 2006-2024 Tiago de Paula Peixoto\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."

/* c++ preprocessor compilation options */
#define CPPFLAGS "-DBOOST_ALLOW_DEPRECATED_HEADERS -DNDEBUG -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/lab/yilu/conda/envs/py311_from_lab/include -I/home/lab/yilu/conda/envs/py311_from_lab/include"

/* c++ compilation options */
#define CXXFLAGS "-fopenmp -O3 -fvisibility=default -fvisibility-inlines-hidden -Wno-deprecated -Wall -Wextra -ftemplate-backtrace-limit=0 -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/lab/yilu/conda/envs/py311_from_lab/include -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/graph-tool-suite_1716855807735/work=/usr/local/src/conda/graph-tool-base-2.68 -fdebug-prefix-map=/home/lab/yilu/conda/envs/py311_from_lab=/usr/local/src/conda-prefix -I/home/lab/yilu/conda/envs/py311_from_lab/include -std=c++17 -O3"

/* compile debug info */
/* #undef DEBUG */

/* GCC version value */
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

/* git HEAD commit hash */
#define GIT_COMMIT "318a7c03"

/* git HEAD commit date */
#define GIT_COMMIT_DATE ""

/* define if the Boost library is available */
#define HAVE_BOOST /**/

/* define if the Boost::Context library is available */
#define HAVE_BOOST_CONTEXT /**/

/* define if the Boost::Coroutine library is available */
#define HAVE_BOOST_COROUTINE /**/

/* define if the Boost::Graph library is available */
#define HAVE_BOOST_GRAPH /**/

/* define if the Boost::IOStreams library is available */
#define HAVE_BOOST_IOSTREAMS /**/

/* "We know boost-python is available in this conda-forge recipe" */
#define HAVE_BOOST_PYTHON 1

/* define if the Boost::Regex library is available */
#define HAVE_BOOST_REGEX /**/

/* define if the Boost::Thread library is available */
#define HAVE_BOOST_THREAD /**/

/* Cairomm is available */
#define HAVE_CAIROMM 1

/* Cairomm 1.16 is available */
#define HAVE_CAIROMM_1_16 1

/* Indicates presence of CGAL library */
#define HAVE_CGAL 1

/* define if the compiler supports basic C++17 syntax */
#define HAVE_CXX17 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `gmp' library (-lgmp). */
#define HAVE_LIBGMP 1

/* Define to 1 if you have the <minix/config.h> header file. */
/* #undef HAVE_MINIX_CONFIG_H */

/* Define if OpenMP is enabled */
#define HAVE_OPENMP 1

/* If available, contains the Python version number currently in use. */
/* #undef HAVE_PYTHON */

/* Using google's sparsehash */
#define HAVE_SPARSEHASH 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <wchar.h> header file. */
#define HAVE_WCHAR_H 1

/* python prefix */
#define INSTALL_PREFIX "/home/lab/yilu/conda/envs/py311_from_lab"

/* linker options */
#define LDFLAGS "-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/lab/yilu/conda/envs/py311_from_lab/lib -Wl,-rpath-link,/home/lab/yilu/conda/envs/py311_from_lab/lib -L/home/lab/yilu/conda/envs/py311_from_lab/lib -L/home/lab/yilu/conda/envs/py311_from_lab/lib"

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "graph-tool"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "http://graph-tool.skewed.de/issues"

/* package data dir */
#define PACKAGE_DATA_DIR "/home/lab/yilu/conda/envs/py311_from_lab/share/graph-tool"

/* package doc dir */
#define PACKAGE_DOC_DIR "${datarootdir}/doc/${PACKAGE_TARNAME}"

/* Define to the full name of this package. */
#define PACKAGE_NAME "graph-tool"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "graph-tool 2.68"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "graph-tool"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://graph-tool.skewed.de"

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.68"

/* pycairo header file */
#define PYCAIRO_HEADER <py3cairo.h>

/* The directory name for the site-packages subdirectory of the standard
   Python install tree. */
#define PYTHON_DIR "/home/lab/yilu/conda/envs/py311_from_lab/lib/python3.11/site-packages"

/* Sparsehash include macro */
#define SPARSEHASH_INCLUDE(f) <sparsehash/f>

/* Sparsehash include prefix */
#define SPARSEHASH_PREFIX sparsehash

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable general extensions on macOS.  */
#ifndef _DARWIN_C_SOURCE
# define _DARWIN_C_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable X/Open compliant socket functions that do not require linking
   with -lxnet on HP-UX 11.11.  */
#ifndef _HPUX_ALT_XOPEN_SOCKET_API
# define _HPUX_ALT_XOPEN_SOCKET_API 1
#endif
/* Identify the host operating system as Minix.
   This macro does not affect the system headers' behavior.
   A future release of Autoconf may stop defining this macro.  */
#ifndef _MINIX
/* # undef _MINIX */
#endif
/* Enable general extensions on NetBSD.
   Enable NetBSD compatibility extensions on Minix.  */
#ifndef _NETBSD_SOURCE
# define _NETBSD_SOURCE 1
#endif
/* Enable OpenBSD compatibility extensions on NetBSD.
   Oddly enough, this does nothing on OpenBSD.  */
#ifndef _OPENBSD_SOURCE
# define _OPENBSD_SOURCE 1
#endif
/* Define to 1 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_SOURCE
/* # undef _POSIX_SOURCE */
#endif
/* Define to 2 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_1_SOURCE
/* # undef _POSIX_1_SOURCE */
#endif
/* Enable POSIX-compatible threading on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-5:2014.  */
#ifndef __STDC_WANT_IEC_60559_ATTRIBS_EXT__
# define __STDC_WANT_IEC_60559_ATTRIBS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-1:2014.  */
#ifndef __STDC_WANT_IEC_60559_BFP_EXT__
# define __STDC_WANT_IEC_60559_BFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-2:2015.  */
#ifndef __STDC_WANT_IEC_60559_DFP_EXT__
# define __STDC_WANT_IEC_60559_DFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-4:2015.  */
#ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
# define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-3:2015.  */
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
# define __STDC_WANT_IEC_60559_TYPES_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TR 24731-2:2010.  */
#ifndef __STDC_WANT_LIB_EXT2__
# define __STDC_WANT_LIB_EXT2__ 1
#endif
/* Enable extensions specified by ISO/IEC 24747:2009.  */
#ifndef __STDC_WANT_MATH_SPEC_FUNCS__
# define __STDC_WANT_MATH_SPEC_FUNCS__ 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable X/Open extensions.  Define to 500 only if necessary
   to make mbstate_t available.  */
#ifndef _XOPEN_SOURCE
/* # undef _XOPEN_SOURCE */
#endif


/* Version number of package */
#define VERSION "2.68"
