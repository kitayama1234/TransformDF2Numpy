from setuptools import setup, find_packages


setup(
    name="df2numpy",
    version="0.1.0",
    description="A simple tool for quick transformation from pandas.DataFrame to numpy.array dataset containing some utilities",
    long_description='./README.md',
    author="Masaki Kitayama",
    author_email="kitayama-masaki@ed.tmu.ac.jp",
    install_requires=['numpy', 'pandas'],
    tests_require=["pytest"],
#     package_dir={"": "df2numpy"},
    url='https://github.com/kitayama1234/TransformDF2Numpy',
    license='./LICENSE',
    packages=["df2numpy"]
)




