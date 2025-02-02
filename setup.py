from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vaganboost",
    version="0.7.7",
    author="Ali Bavarchee",
    author_email="ali.bavarchee@gmail.com",
    description="A hybrid model combining VAE, GAN, and LightGBM for boosting performance in high-energy physics or data analysis tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AliBavarchee/vaganboost", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.8.0",
        "lightgbm>=3.2.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "sphinx>=4.0",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "vaganboost-train=vaganboost.train:main",
        ],
    },
)
