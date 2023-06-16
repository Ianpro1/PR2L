from setuptools import setup, find_packages
import glob
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        print("NOTE: Some modules may utilize classes found in either the gymnasium or gym modules (openai). The following setup does not require their installation, therefore it is up to the user to install them.")
        install.run(self)

setup(
    name='PR2L',
    author="_ianmi",
    version='0.0.1',
    packages=find_packages(where="PR2L", include='*'),
    data_files=[('Lib/site-packages/PR2L', glob.glob('PR2L/*.py'))],
    cmdclass={'install': CustomInstallCommand},
)

