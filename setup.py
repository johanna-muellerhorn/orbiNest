from setuptools import setup, find_packages

setup(
    name='orbiNest',
    version='0.1.0',
    description='Bayesian orbital fitting for binary stars using nested sampling',
    author='J. Mueller-Horn',
    author_email='you@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'ultranest',
        'astropy'
    ],
    entry_points={
        'console_scripts': [
            'orbinest-example=examples.run_mock_example:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    python_requires='>=3.7',
)
