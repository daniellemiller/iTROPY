import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="itropy",
    version="0.0.1",
    author="Danielle Miller",
    author_email="danimillers10@gmail.com",
    description="A package for finding information hotspots in a given string",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniellemiller/iTROPY",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)