import setuptools

__version__ = "0.0.0"
SRC_REPO = "ImageCaptioning"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author="vyash",
    description="A small python package for Image Captioning app",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
