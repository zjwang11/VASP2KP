# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:54:57 2023
Last modified on Wedn Dec 6 18:45:00 2023

@author: Sheng Zhang, Institute of Physics, Chinese Academy of Sciences

Set up VASP2KP
"""


import sys

if sys.version_info < (3, 6):
    raise 'must use Python version 3.6 or higher'
    
from setuptools import setup

README = """A tool for computing k.p effective Hamiltonians and Zeeman's coupling under given symmetry constraints from VASP calculations."""


setup(name='VASP2KP',
      python_requires=">=3.6",
      version = "1.0.2",
      long_description=README,
      install_requires=[
        'sympy', 'numpy', 'scipy', 'kdotp_generator'
        ],
      packages=['VASP2KP'],
      author = 'Sheng Zhang, Haohao Sheng, Zhi-Da Song, Chenhao Liang, Zhijun Wang',
      author_email='zhangsheng221@mails.ucas.ac.cn, songzd@pku.edu.cn, wzj@iphy.ac.cn',
      include_package_data=True,
      zip_safe=False,
      license='GPLv3',
      package_data={'.': ['mat2kp', 'mat2kp.in','vasp2mat.5.3-patch-1.0.1.sh','vasp2mat.6.4-patch-1.0.1.sh']},
      url='https://github.com/zjwang11/VASP2KP',
      classifiers=[
      	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      	'Natural Language :: English',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
	'Programming Language :: Python :: 3.9',
	'Programming Language :: Python :: 3.10',
	'Programming Language :: Python :: 3.11',
	'Programming Language :: Python :: 3.12',
	'Intended Audience :: Science/Research',
	'Topic :: Scientific/Engineering :: Physics',
	'Development Status :: 4 - Beta'
      ]
      )


