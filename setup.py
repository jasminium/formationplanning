import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="formationplanning",
    version="1.0.0",
    author="John Hartley",
    author_email="john.hartley@northumbria.ac.uk",
    description="Package to generate trajectories for UAVs with performance constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['formationplanning', 'formationplanning.plotting'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'toppra', 'matplotlib', 'tqdm'],
    python_requires='>=3.6',
)