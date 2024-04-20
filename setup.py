from setuptools import setup, find_packages

setup(
    name='temporal_graph_learning',  # This is the name of your package
    version='0.1.0',                 # The initial release version
    author='Your Name',              # Optional: your name or your organization’s name
    author_email='your.email@example.com',  # Optional: your email address
    description='A brief description of what your package does',  # Optional but recommended
    packages=find_packages(where='src'),  # This finds all modules in 'src' directory
    package_dir={'': 'src'},  # Tells distutils that everything under 'src' is part of the package
    install_requires=[
        # List all packages that your package depends on:
        # 'numpy',
        # 'pandas',
        # etc.
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',    # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',    # Define your audience
        'Natural Language :: English',        # Optional: main language used in your package
        'Operating System :: OS Independent', # Should work on any OS
        'Programming Language :: Python :: 3', # Supported Python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)