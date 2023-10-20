# PyCorLimbo

This library implements Bayesian optimal sampling for correlation fields in Python via the C++ library
[Limbo](https://github.com/resibots/limbo). It is recommended to use this library in conjunction with
[PyCoriander](https://github.com/chrismile/PyCoriander), which implements the evaluation of different correlation
metrics. Both libraries have a Python interface relying on PyTorch.


## Install with setuptools

A prerequisites for installing this library is installing [PyTorch](https://pytorch.org/) in the currently active Python
environment. Information on how to install PyTorch can be found on the website of the project.

Additionally, the libraries nlopt and Boost must be installed somewhere on the system where setup.py can find it.
On Linux distributions, this can for example be done with one of the following commands.

```shell
# Ubuntu or other Debian-based distributions
sudo apt install libnlopt-cxx-dev libboost-dev
# Arch Linux-based distributions (e.g., Arch, Manjaro)
sudo pacman -S nlopt boost
# Fedora
sudo yum install -y NLopt boost-devel
```

To download all submodules of this library, the following command can be used.

```shell
git submodule update --init
```

To finally install the library as a Python module, the following command must be called in the repository directory.

```sh
python setup.py install
```

If it should be installed in a Conda environment, activate the corresponding environment first as follows.

```sh
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate <env-name>
```
