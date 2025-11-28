import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "agents.mcts.cython.sampler.mcts_sampler",
        ["agents/mcts/cython/sampler/mcts_sampler.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        define_macros=[('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')],
        extra_compile_args=["-O3", "-march=native", "-std=c++11", "-ffast-math", "-funroll-loops"],
    ), 
    Extension(
        "agents.mcts.cython.node.mcts_node",
        ["agents/mcts/cython/node/mcts_node.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        define_macros=[('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')],
        extra_compile_args=["-O3", "-march=native", "-std=c++11", "-ffast-math", "-funroll-loops"],
    ),
    Extension(
        "agents.mcts.cython.env.mcts_env",
        ["agents/mcts/cython/env/mcts_env.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        define_macros=[('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')],
        extra_compile_args=["-O3", "-march=native", "-std=c++11", "-ffast-math", "-funroll-loops"],
    ),
    Extension(
        "cython_lib.cyhanabi",
        ["cython_lib/cyhanabi.pyx"],
        include_dirs=[
            numpy.get_include(),
            ".",           # For finding pyhanabi.h
            "hanabi_lib",  # For finding hanabi library headers
        ],
        libraries=["pyhanabi"],  # Link against the existing C++ library
        library_dirs=["."],      # Where to find the library
        language="c++",
        extra_compile_args=["-O3", "-march=native", "-std=c++11", "-ffast-math", "-funroll-loops"],
    ),
]

setup(
    name="MCTS",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "profile": False,
            "linetrace": False,
            "binding": False,
            "infer_types": True,
        },
    ),
    packages=["cython_lib", "agents.mcts.cython.sampler", "agents.mcts.cython.node", "agents.mcts.cython.env"],
    zip_safe=False,
)
