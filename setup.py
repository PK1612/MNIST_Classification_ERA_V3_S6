from setuptools import setup, find_packages

setup(
    name="mnist_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'numpy>=1.19.2',
        'matplotlib>=3.3.2',
        'pytest>=6.0.0',
    ],
) 