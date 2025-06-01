# 🚀 Ollama RAG Engine (Beta)

**A powerful RAG (Retrieval-Augmented Generation) engine powered by Ollama, ChromaDB, and Gradio. Query your documents with local LLMs.**

[![CI Status](https://github.com/anandan-bs/ollama-rag-engine/workflows/Ollama%20RAG%20Engine%20CI/badge.svg)](https://github.com/anandan-bs/ollama-rag-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🔍 **Advanced RAG Implementation**
  - ChromaDB for efficient vector storage and similarity search
  - Sentence Transformers for high-quality embeddings
  - Configurable chunk size and overlap
  - Supports multiple document formats (PDF, TXT, MD, RST)

- 🤖 **Local LLM Integration**
  - Seamless integration with Ollama
  - Support for multiple models (Mistral, Llama, etc.)
  - Configurable model parameters (temperature, context window)
  - Rate limiting and caching for optimal performance

- 🎯 **Modern UI with Gradio**
  - Clean and intuitive interface
  - Real-time indexing progress
  - Chat history management
  - Session persistence
  
[Screenshot placeholder: Main chat interface]

- 📊 **Production-Ready Features**
  - Comprehensive logging
  - Health checks for all components
  - Error handling and recovery
  - Environment-based configuration

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- (Optional) Git for version control

### Installation

```bash
# Clone the repository
git clone https://github.com/anandan-bs/ollama-rag-engine.git
cd ollama-rag-engine

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

1. Start Ollama with your preferred model:
```bash
ollama run mistral  # or any other supported model
```

2. Run the application:
```bash
python main.py
```

3. Open your browser and navigate to:
```
http://localhost:7860
```

[Screenshot placeholder: Document upload interface]

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
OLLAMA_API_URL=http://localhost:11434/api/generate
MODEL_NAME=mistral
TEMPERATURE=0.7

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database Configuration
DB_PATH=./data/chromadb
COLLECTION_NAME=documents

# Storage Configuration
UPLOAD_DIR=./data/uploads
CHATLOG_DIR=./data/chatlogs

# Processing Configuration
MIN_CHUNK_SIZE=100

# Environment
ENV=development
DEBUG=true
```

### Changing Models

1. **LLM Model**: 
   - Install desired model in Ollama: `ollama pull <model_name>`
   - Update `MODEL_NAME` in `.env`

2. **Embedding Model**:
   - Update `EMBEDDING_MODEL` in `.env`
   - Choose from [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)

## 📚 API Reference

### RAG Configuration

```python
from app.config import OLLAMA_CONFIG, EMBEDDING_CONFIG

# Customize LLM parameters
OLLAMA_CONFIG.update({
    "temperature": 0.5,
    "top_p": 0.9
})

# Customize embedding parameters
EMBEDDING_CONFIG.update({
    "normalize_embeddings": True
})
```

## 🧪 Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
make test
```

### Code Quality

```bash
# Run linting
make lint

# Format code
black app/ tests/
```

### Clean up

```bash
make clean
```

### Run

```bash
make run
```

## 🗺️ Roadmap

- [ ] Support for more document formats (DOCX, EPUB)
- [ ] Multi-model comparison interface
- [ ] Batch processing for large document sets
- [ ] Advanced RAG techniques (Hypothetical Document Embeddings)
- [ ] API endpoint for headless operation
- [ ] Docker support
- [ ] Performance optimization for large collections
- [ ] Web crawling support
- [ ] Custom embedding model training
- [ ] Integration with other vector stores

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for functions and classes
- Add tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Anandan B S**
- Email: anandanklnce@gmail.com
- GitHub: [@anandan-bs](https://github.com/anandan-bs)

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the amazing local LLM runtime
- [ChromaDB](https://www.trychroma.com/) for the vector store
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Gradio](https://gradio.app/) for the UI framework

---

*Note: This is a beta version. Please report any issues on GitHub.*
