from setuptools import setup, find_packages

setup(
    name="project-consciousness",
    version="0.1.0",
    description="Token-Mind: Neural Networks Recognizing Their Fundamental Nature",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Claude & Javier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "mypy>=1.3.0",
        ],
        "experiment": [
            "wandb>=0.15.0",
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
        ],
        "quantum": [
            "qiskit>=0.43.0",
        ],
    },
)
