import sys
from setuptools import setup
from setuptools.command.egg_info import egg_info
from torch.utils.cpp_extension import BuildExtension, CppExtension, IS_WINDOWS, IS_MACOS

extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
elif IS_MACOS:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp=libomp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')

class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)

setup(
    name='pycorlimbo',
    author='Christoph Neuhauser',
    ext_modules=[
        CppExtension(
            'pycorlimbo',
            [
                'src/Random/Random.cpp',
                'src/Random/Xorshift.cpp',
                'src/Correlation.cpp',
                'src/MutualInformation.cpp',
                'src/PyCorLimbo.cpp',
            ],
            libraries=['nlopt'],
            extra_compile_args=extra_compile_args,
        )
    ],
    data_files=[
        ( '.', ['src/pycorlimbo.pyi'] )
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'egg_info': EggInfoInstallLicense
    },
    license_files = ('LICENSE',),
    include_dirs=['third_party/limbo/src', 'third_party/eigen']
)
