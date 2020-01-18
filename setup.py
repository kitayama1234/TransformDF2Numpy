from setuptools import setup, find_packages

readme = './README.md'
license = './LICENSE'

setup(
    name="transformdf2numpy",
    version="1.0.0",
    description="A simple tool for quick transformation from pandas.DataFrame to numpy.array dataset containing some utilities",
    long_description=readme,
    author="Masaki Kitayama",
    author_email="kitayama-masaki@ed.tmu.ac.jp",
    install_requires=['numpy', 'pandas'],
    url='https://github.com/kitayama1234/TransformDF2Numpy',
    license=license,
    package=find_packages()
)


