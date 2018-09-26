import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearn_seco',
    version='0.0',
    packages=setuptools.find_packages('sklearn_seco'),
    package_dir={'': 'sklearn_seco'},
    url='https://github.com/azrdev/sklearn-seco',
    license='BSD',
    author='Jonathan Biegert',
    author_email='azrdev@qrdn.de',
    description='Implementation of the *Separate and Conquer* / '
                '*Covering*-Algorithm for scikit-learn.',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
