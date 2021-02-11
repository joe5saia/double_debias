import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="double_debias",  # Replace with your own username
    version="0.0.5",
    author="Joe Saia",
    author_email="joe5saia@gmail.com",
    description="Implements Double Debias Estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joe5saia/double_debias",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'sklearn',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    python_requires='>=3.6',
)
