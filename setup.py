from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vaganboostktf",
    version="0.9.1",
    author="Ali Bavarchee",
    author_email="ali.bavarchee@gmail.com",
    description="Hybrid VAE-GAN with LightGBM for class-imbalanced classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AliBavarchee/vaganboost",
    packages=find_packages(include=["vaganboostktf", "vaganboostktf.*"]),
    package_data={
        "vaganboostktf": [
            "config/*.json",
            "best_models/*/*",
            "best_models/*/*/*"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    install_requires=[
        "dill>=0.3.9",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tensorflow>=2.8.0",
        "keras>=3.8.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "tqdm>=4.64.0"
    ],
    entry_points={
        "console_scripts": [
            "vaganboostktf=vaganboostktf.cli:main",
        ],
    },
    keywords=[
        "machine-learning",
        "deep-learning",
        "data-augmentation",
        "class-imbalance",
        "vae",
        "gan",
        "lightgbm"
    ],
    license="MIT",
)