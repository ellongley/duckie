from setuptools import setup

# make the README into the long description
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="duckie",
    version="0.0.1",
    description="test machine learning model",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="github.com/ellongley/duckie",
    author="Emily Phillips Longley",
    license="MIT",
    packages=["duckie"],
    package_dir={"duckie": "duckie"},
    install_requires=[
        'numpy>=1.21',
        'scikit-learn>=0.24',
    ],
    zip_safe=False
)
