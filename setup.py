from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py
import subprocess as sub
import sys
import os

class MyInstall(install):
    def run(self):
        install.run(self)
        for ii in range(20):
            print('-'*10)
        if sys.platform == 'darwin':
            julia = '/Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia'
        if sys.platform == 'linux2':
            julia = 'julia'
        if sys.platform == 'win32':
            julia = os.path.expanduser('~') + '\\AppData\\Local\\Programs\\Julia-1.9.3\\bin\\julia.exe'
        sub.call([julia, 'InstallPkg.jl'])


setup(name='pyLLEHelper',
      version='1.0.0',
      description='LLE Solver',
      url='https://github.com/noiclas/pyLLEHelper.git',
      author='Greg Moille',
      author_email='gmoille@umed.gov',
      license='Open',
      long_description='User friendly LLE solver. For more information, visit our github repository page',
      packages=['pyLLEHelper'],
      install_requires=[
          'scipy',
          'plotly',
          'numpy',
          'matplotlib',
          'h5py',
          'prettytable',
          'matplotlib',
          'ipdb',
      ],
      package_data={'': ['*.jl']},
      include_package_data=True,
      zip_safe=False,
      cmdclass={'install': MyInstall},
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ),)
