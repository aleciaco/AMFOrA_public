"""
Setup file for AMACFA+ package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amacfa-plus",
    version="1.0.0",
    author="Alec Iacobucci",
    author_email="aleciaco@uw.edu",
    description="Automated Macroscopic Fabric and Orientation Analysis for ceramic sherds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "pillow>=8.0.0",
    ],
    keywords="ceramic analysis, computer vision, archaeological image processing, petrography",
    project_urls={
        "Source": "https://github.com/aleciaco/AMFOrA_public",
        "Tracker": "https://github.com/aleciaco/AMFOrA_public/issues",
    },
)