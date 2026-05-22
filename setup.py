from setuptools import setup, find_packages

setup(
    name="latam_eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "langchain>=0.1.0",
        "google-genai>=0.1.0",
        "pyyaml>=6.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.1",
    ],
    entry_points={"console_scripts": ["eval-llm=latam_eval.cli:main"]},
)
