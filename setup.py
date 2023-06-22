from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    description = fh.read()

header_idx = description.find("##")
if header_idx != -1:
    description = description[:header_idx]
else:
    description = description

setup(
    name="cuSPREC",
    version="0.0.1",
    author="Camille Dunning",
    description="Sparse recovery playground with PyCUDA.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/camille-004/cusprec",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: CUDA",
        "Operating System :: OS Independent",
    ],
)
