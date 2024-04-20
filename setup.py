from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=['temporal_graph_learning'],
    install_requires=[
        'pandas'
    ],
)