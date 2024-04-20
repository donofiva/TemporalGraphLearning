from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    package_dir={'src': 'temporal_graph_learning'},
    packages=find_packages(include=['src', 'src*']),
    install_requires=[
        'pandas'
    ],
)