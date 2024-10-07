from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "agents.mcts.cython.sampler.mcts_sampler",
        ["agents/mcts/cython/sampler/mcts_sampler.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name='MCTS Sampler',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
