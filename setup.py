import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearn_seco',
    version='0.0',
    packages=setuptools.find_packages(),
    url='https://github.com/azrdev/sklearn-seco',
    license='BSD',
    author='Jonathan Biegert',
    author_email='azrdev@qrdn.de',
    description='Implementation of the *Separate and Conquer* / '
                '*Covering*-Algorithm for scikit-learn.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=[
        'scikit_learn >= 0.20',
        'numpy',
    ],
    extras_require={
        'tests': ['matplotlib', 'pytest >= 3.5'],
        'numba': ['numba'],
    },
)
