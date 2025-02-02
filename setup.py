from setuptools import setup, find_packages

# Read the contents of the README file for long description (optional)
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vaganboost",  # Name of your package
    version="0.7.6",  # Initial version of your package
    author="Ali Bavarchee",  # Your name
    author_email="ali.bavarchee@gmail.com",  # Your email
    description="A hybrid model combining VAE, GAN, and LightGBM for boosting performance in high-energy physics or data analysis tasks.",
    long_description=long_description,  # Read long description from README.md
    long_description_content_type="text/markdown",  # Format of the long description
    url="https://github.com/AliBavarchee/vaganboost",  # URL to the project (if available)
    packages=find_packages(),  # Automatically find all packages in the current directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",  # License type (adjust as needed)
        "Operating System :: OS Independent",  # Supports all major operating systems
    ],
    python_requires=">=3.6",  # Minimum Python version required
    install_requires=[  # List of dependencies
        "torch>=1.8.0",
        "lightgbm>=3.2.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
    ],
    extras_require={  # Optional dependencies for extra features
        "dev": [
            "pytest>=6.0",  # For testing during development
            "sphinx>=4.0",  # For documentation generation
            "black",  # For code formatting
        ],
    },
    entry_points={  # Entry points for command-line tools (if needed)
        "console_scripts": [
            "vaganboost-train=vaganboost.train:main",  # Example of CLI tool entry point
        ],
    },
)
