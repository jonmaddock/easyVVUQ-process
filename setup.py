from setuptools import setup, find_packages

setup(
    name="infeas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "easyvvuq",
        "dask",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
    ],
)
