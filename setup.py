from setuptools import setup, find_packages
import glob

setup(
    name='PR2L',
    author="_ianmi",
    version='0.0.1',
    packages=find_packages(where="PR2L", include='*'),
    data_files=[('Lib/site-packages/PR2L', glob.glob('PR2L/*.py'))]
)
