from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    package_dir={'temporal_graph_learning': 'src'},
    packages=find_packages(where=['src', 'src*']),
    install_requires=[
        'pandas'
    ],
)