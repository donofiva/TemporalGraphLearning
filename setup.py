from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.0.1',
    author='Ivan D\'Onofrio',
    author_email='s269504@studenti.polito.com',
    description='Right now this package doesn\'t do much TBH',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'matplotlib',
        'seaborn'
    ]
)