from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="cmvc",
    version="0.1.0",
    license="MIT license",
    description="パッケージの説明を追加",
    author="Kenya Maejima",
    url="https://github.com/kjun1/CMVC",
    packages=find_packages("cmvc"),
    package_dir={"": "cmvc"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)