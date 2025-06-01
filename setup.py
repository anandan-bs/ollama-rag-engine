"""
Setup script for ollama-rag-engine package.
"""

from setuptools import setup, find_packages

setup(
    name="ollama-rag-engine",
    version="0.1.0-beta",
    description="A powerful RAG engine powered by Ollama, ChromaDB, and Gradio",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anandan B S",
    author_email="anandanklnce@gmail.com",
    url="https://github.com/anandan-bs/ollama-rag-engine",
    packages=find_packages(),
    install_requires=[
        "gradio>=4.19.2",
        "chromadb>=0.4.15",
        "sentence-transformers>=2.2.2",
        "requests>=2.31.0",
        "PyMuPDF>=1.23.8",
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "flake8>=7.0.0",
            "black>=24.1.1",
            "pre-commit>=3.5.0",
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
