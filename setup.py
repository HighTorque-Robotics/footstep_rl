from setuptools import find_packages
from distutils.core import setup

setup(
    name='gpuGym',
    version='1.0.1',
    author='Biomimetic Robotics Lab',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'matplotlib',
                      'pandas',
                      'tensorboard',
                      'setuptools==59.5.0',
		              'torch>=1.4.0',
		              'torchvision>=0.5.0',
		              'numpy>=1.16.4']
)
# inference 0.13.0 requires cython<=3.0.0, which is not installed.
# supervision 0.21.0 requires opencv-python-headless>=4.5.5.64, which is not installed.
# inference 0.13.0 requires networkx>=3.1, but you have networkx 2.2 which is incompatible.
# inference 0.13.0 requires opencv-python<=4.8.0.76, but you have opencv-python 4.10.0.84 which is incompatible.
# inference 0.13.0 requires setuptools<70.0.0,>=65.5.1, but you have setuptools 59.5.0 which is incompatible.
# supervision 0.21.0 requires matplotlib>=3.6.0, but you have matplotlib 3.5.3 which is incompatible.
