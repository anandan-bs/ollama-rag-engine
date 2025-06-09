"""
Setup script for ragify-docs package.
"""

from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

setup(
    name="ragify-docs",
    version="0.1.0-beta",
    description="A powerful RAG engine powered by Ollama, ChromaDB, and Gradio",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anandan B S",
    author_email="anandanklnce@gmail.com",
    url="https://github.com/anandan-bs/ragify-docs",
    packages=find_packages(),
    install_requires=[
        req for req in parse_requirements('requirements.txt')
        if not any(dev_req in req.lower() for dev_req in ['pytest', 'pytest-cov', 'flake8', 'black'])
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'flake8>=6.1.0',
            'black>=23.11.0',
            'pre-commit>=3.5.0',
        ]
    },
    entry_points={
        "console_scripts": [
            "ragify-docs = ragify_docs.main:launch"
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
    ],
    keywords=["rag", "ollama", "llm", "chromadb", "embeddings", "chatbot"],
)
