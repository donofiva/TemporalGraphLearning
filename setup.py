from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    packages=['temporal_graph_learning'],
    package_dir={'': 'src'},
    install_requires=[
        'pandas'
    ],
)