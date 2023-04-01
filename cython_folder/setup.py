from setuptools import setup
from Cython.Build import cythonize
import subprocess
subprocess.call(["cython","-a","exp.pyx"])

setup(
    ext_modules = cythonize("expv2.pyx", annotate=True),
    
)