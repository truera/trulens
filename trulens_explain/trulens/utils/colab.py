# Some utilities to help with colab/jupyter notebook demos.

import importlib
from pathlib import Path
import subprocess
import sys


def install_if_not_installed(packages):
    """
    Install the given packages if they are not already installed.
    """

    for package in packages:
        if isinstance(package, tuple):
            package_name, package_package = package
        else:
            package_name = package
            package_package = package

        print(f'{package_name} ... ', end='')

        try:
            importlib.import_module(package_name)
            print('already installed')

        except:
            print(f'installing from {package_package}')
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package_package]
            )


def load_or_make(
    filename: Path, loader, maker=None, saver=None, downloader=None
):
    """
    Load something from a `filename` using `loader` if the file exists, otherwise
    make it using `maker` or download it using `downloader`, save it using
    `saver`, and return it.
    """

    print(f'loading {filename} ... ', end='')

    if filename.exists():
        print('from file')
        return loader(filename)

    if maker is not None:
        print('using maker')
        thing = maker()
        saver(filename, thing)
        return thing

    if downloader is not None:
        print('using downloader')
        downloader(filename)
        return loader(filename)

    raise ValueError('provide a maker/saver or downloader.')
