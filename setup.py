from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='dqn_cpp',
      ext_modules=[CppExtension('dqn', ['Trainer.cpp'])],
      cmdclass={'build_ext': BuildExtension})