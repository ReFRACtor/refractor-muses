from setuptools import setup, find_namespace_packages

# Version moved to python/refractor/muses/version.py so we have one place it is defined.
exec(open("python/refractor/muses/version.py").read())

# Namespace packages are a bit on the new side. If you haven't seen
# this before, look at https://packaging.python.org/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages for a description of
# this.
setup(
    name='refractor-muses',
    version=__version__,
    description='ReFRACtor MUSE integration',
    author='James McDuffie',
    author_email='James.McDuffie@jpl.nasa.gov',
    packages=find_namespace_packages(include=["refractor.*"],
                                     where="python"),
    package_dir={"": "python"},
    package_data={"*" : ["py.typed", "*.pyi"]},
    install_requires=[
        'numpy', 'refractor-framework', 
    ],
    scripts=["bin/refractor-retrieve",]
)
