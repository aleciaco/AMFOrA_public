"""
Setup file for AMACFA+ package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amacfa-plus",
    version="1.0.0",
    author="Enhanced by Claude Code",
    description="Automatic Macroscopic Fabric and Orientation Analysis (Enhanced)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "pillow>=8.0.0",
    ],
    keywords="ceramic analysis, computer vision, archaeological image processing",
    project_urls={
        "Documentation": "https://github.com/your-username/amacfa-plus",
        "Source": "https://github.com/your-username/amacfa-plus",
        "Tracker": "https://github.com/your-username/amacfa-plus/issues",
    },
)