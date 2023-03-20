from setuptools import setup

setup(
    name='gpuaffman_networks',
    version='0.1.0',
    description='GPU accelerated Kauffman networks',
    url='https://github.com/trevormccrt/gpuaffman_networks',
    author='Trevor McCourt',
    author_email='tmccourt@mit.edu',
    license='BSD 2-clause',
    packages=['gpuaffman_networks'],
    install_requires=['numpy'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: MIT',
        'Programming Language :: Python :: 3',
    ],
)
