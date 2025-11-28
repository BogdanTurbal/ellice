from setuptools import setup, find_packages

setup(
    name="ellice",
    version="0.1.0",
    description="ElliCE: Actionability-aware Robust Counterfactual Explanations",
    author="Bohdan Turbal",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "pyyaml",
        "matplotlib",
    ],
)

