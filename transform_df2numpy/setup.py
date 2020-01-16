from setuptools import setup, find_packages

readme = './README.md'
license = './LICENSE'

setup(
    name="transform_df2numpy",
    version="0.0.1",
    description="A simple tool for quick transformation from pandas.DataFrame to numpy.array dataset containing some utilities",
    long_description=readme,
    author="Masaki Kitayama",
    author_email="kitayama-masaki@ed.tmu.ac.jp",
    install_requires=['numpy', 'pandas'],
    url='',
    license=license,
    package=find_packages()
)


