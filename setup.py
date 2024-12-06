import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "agents.mcts.cython.sampler.mcts_sampler",
        ["agents/mcts/cython/sampler/mcts_sampler.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
    ), 
    Extension(
        "agents.mcts.cython.node.mcts_node",
        ["agents/mcts/cython/node/mcts_node.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
    ),
    Extension(
        "cython_lib.cyhanabi",
        ["cython_lib/cyhanabi.pyx"],
        include_dirs=[
            numpy.get_include(),
            ".",  # For finding pyhanabi.h
            "hanabi_lib",  # For finding hanabi library headers
        ],
        libraries=["pyhanabi"],  # Link against the existing C++ library
        library_dirs=["."],  # Where to find the library
        language="c++",  # Specify that we're using C++
        extra_compile_args=["-O3", "-march=native", "-std=c++11","-ffast-math", "-funroll-loops"],
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
            "infer_types": True,
        },
    ),
    packages=["cython_lib", "agents.mcts.cython.sampler", "agents.mcts.cython.node"],
    zip_safe=False,
)
