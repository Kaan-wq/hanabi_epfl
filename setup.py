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
    zip_safe=False,
)
