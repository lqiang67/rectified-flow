from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rectified-flow",
    version="1.0.1",
    description="A PyTorch implementation of Rectified Flow and its variants.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Qiang Liu, Runlong Liao, Bo Liu, Xixi Hu",
    author_email="rectifiedflow@gmail.com",
    url="https://github.com/lqiang67/rectified-flow",
    packages=find_packages(),
    python_requires=">=3.10",
    keywords="pytorch, rectified-flow, generative-modeling",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "accelerate>=1.2.0",
        "openai-clip==1.0.1",
        "diffusers>=0.31.0",
        "matplotlib>=3.9.3",
        "numpy",
        "packaging",
        "Pillow",
        "plotly",
        "scikit_learn",
        "scipy",
        "sympy",
        "timm>=1.0.12",
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "tqdm"
    ]
)
