from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "engine_core", 
        sources=["engine.pyx"],
        libraries=["GLESv2", "log","SDL2"],
        extra_compile_args=['-O3'],
        language="c++",
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="pydroid_3d",
    version="1.1",
    author="stndstnd",
    ext_modules=cythonize(ext_modules, language_level=3),
    zip_safe=False,
)

