from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'temporal_graph_learning': 'src'},
    install_requires=[
        'pandas'
    ],
)