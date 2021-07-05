from setuptools import find_packages, setup

setup(
    name='daseg',
    license='Apache-2.0 License',
    version='0.1',
    packages=find_packages(),
    scripts=['daseg/bin/dasg']
)
