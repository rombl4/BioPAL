import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biopal",
    version="0.0.1",
    author="the BioPAL team",
    author_email="biopal@esa.int",
    description="BIOMASS Product Algorithm Laboratory",
    long_description=long_description,
    long_description_content_type="text/x-md",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'biopal': ['conf/*.xml']},
    entry_points = {'console_scripts': ['biopal=biopal.__main__:main',]},
)