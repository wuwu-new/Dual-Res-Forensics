#!/usr/bin/env python
"""Setup configuration for DRF v2."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drf-forensics",
    version="2.0.0",
    author="wuwu-new",
    description="Dual-Res-Forensics v2: Remote Deepfake Detection with ViT-L/14",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wuwu-new/Dual-Res-Forensics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "flake8>=4.0",
            "black>=22.0",
            "mypy>=0.950",
            "sphinx>=4.5",
        ],
        "vis": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "drf-eval=tools.test:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
