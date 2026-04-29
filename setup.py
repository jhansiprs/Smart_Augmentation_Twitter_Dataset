from setuptools import setup, find_packages

setup(
    name="sentiment_analysis_research",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.1",
        "transformers>=4.24.0",
        "nltk>=3.7",
        "scikit-learn>=1.1.3",
        "pandas>=1.5.1",
        "numpy>=1.23.4",
        "openpyxl>=3.0.10",
        "tqdm>=4.64.1",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "googletrans==4.0.0-rc1",
    ],
    author="Prashant Upadhyaya",
    description="Comprehensive Smart Data Augmentation with Multiple Transformer Models for Imbalanced Sentiment Classification",
)
