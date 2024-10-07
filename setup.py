import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "agents.mcts.cython.sampler.mcts_sampler",
        ["agents/mcts/cython/sampler/mcts_sampler.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="MCTS Sampler",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    zip_safe=False,
)
