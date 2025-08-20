#!/usr/bin/env python3
"""
Setup script for QEmbed package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "QEmbed: Quantum-Enhanced Embeddings for Natural Language Processing"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="qembed",
    version="0.1.0",
    author="QEmbed Team",
    author_email="contact@qembed.ai",
    description="Quantum-Enhanced Embeddings for Natural Language Processing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/qembed/qembed",
    project_urls={
        "Bug Tracker": "https://github.com/qembed/qembed/issues",
        "Documentation": "https://qembed.readthedocs.io/",
        "Source Code": "https://github.com/qembed/qembed",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
            "kaleido>=0.2.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "qembed-demo=qembed.examples.basic_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qembed": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "quantum",
        "embeddings",
        "nlp",
        "natural-language-processing",
        "machine-learning",
        "deep-learning",
        "pytorch",
        "transformer",
        "bert",
        "polysemy",
        "uncertainty",
        "superposition",
        "entanglement",
    ],
)
