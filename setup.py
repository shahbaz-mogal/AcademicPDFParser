"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.

Used by Shahbaz Mogal (shahbazmogal@outlook.com) 
for educational purposes.
"""
import os
from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "academicpdfparser", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]

setup(
    name="academic-pdf-parser",
    version=read_version(),
    description="Academic PDF Parser. Based off of Nougat: Neural Optical Understanding for Academic Documents",
    author="Lukas Blecher extended by Shahbaz Mogal",
    author_email="lblecher@meta.com, shahbazmogal@outlook.com",
    url="https://github.com/facebookresearch/nougat",
    license="MIT",
    packages=find_packages(
        exclude=[
            "result",
        ]
    ),
    install_requires=[
        "transformers>=4.25.1",
        "pypdf>=3.1.0",
        "pypdfium2",
        "Pillow",
        "nltk",
        "python-Levenshtein",
        "opencv-python-headless",
        "albumentations>=1.0.0",
        "timm"
    ],
     extras_require={
       "rasterize": [
           "tqdm",
       ]  
     },
    python_requires=">=3.7, <3.12",
)