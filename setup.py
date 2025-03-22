from setuptools import setup, find_packages

setup(
    name="renrag-colbert",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.8.0",
        "colbert-ir>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "pre-commit>=2.13.0",
        ],
    },
    author="Javier Martin",
    author_email="hello@renaiss.ai",
    description="A Python library for semantic document indexing and searching with ColBERT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/renrag-ai/renrag-colbert",
    project_urls={
        "Bug Tracker": "https://github.com/renaiss-ai/renrag-colbert/issues",
        "Documentation": "https://github.com/renaiss-ai/renrag-colbert",
        "Source Code": "https://github.com/renaiss-ai/renrag-colbert",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    keywords="colbert, search, retrieval, semantic-search, nlp, transformers",
    python_requires=">=3.7",
)