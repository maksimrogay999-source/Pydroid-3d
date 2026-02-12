from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "engine_core", 
        sources=["engine.pyx"],
        libraries=["GLESv2", "log"],
        extra_compile_args=['-O3'],
    )
]

setup(
    name="pydroid_3d",
    version="1.0",
    author="stndstnd",
    ext_modules=cythonize(ext_modules, language_level=3),
    zip_safe=False,
)

