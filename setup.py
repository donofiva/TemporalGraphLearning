from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    packages=['src'],
    package_dir={'src': 'temporal_graph_learning'},
    install_requires=[
        'pandas'
    ],
)