from pathlib import Path
import pkg_resources as pkg
from setuptools import find_packages, setup

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]

VERSION = '0.2.0'

setup(
    name='HSC3D',  # name of pypi package
    version=VERSION,  # version of pypi package
    python_requires='>=3.10',
    license='MIT',
    description=('HSC3D: a Python package to quantify three-dimensional habitat structural complexity'),
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    keywords='3D point clouds,')