from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of what your package does',
    package_dir={'': 'src'},  # Points to the top-level folder where packages are
    packages=find_packages(where='src'),  # Automatically find all packages in src/
    install_requires=[
        # dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)