from setuptools import setup, find_packages
setup(
    name='daseg',
    version='0.1',
    packages=find_packages(),
    scripts=['daseg/bin/dasg', 'daseg/bin/dasg_TrueCasing']
)
