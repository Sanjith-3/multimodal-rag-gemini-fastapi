# 🤖 Multimodal RAG System with Gemini and FastAPI

A practical implementation of a Multimodal Retrieval-Augmented Generation (RAG) system using Google's Gemini model, FastAPI, and Streamlit. This project demonstrates basic multimodal RAG capabilities with PDF document processing and chat functionality.

## ✨ What Can It Do?

- 📄 Extract text and images from PDF documents
- 🔍 Store and retrieve content using ChromaDB vector database
- 💬 Process natural language queries
- 🖼️ Handle image-based queries using Gemini multimodal
- 📊 Evaluate response quality with basic metrics

## 🗂️ Project Structure

```
├── API/
│   ├── evaluation.py          # Response evaluation logic
│   ├── rag_chat.py           # Main RAG implementation
│   ├── Streamlit_Chat_app.py # Chat interface
│   ├── figures/              # Stores extracted images
│   ├── uploads/              # Stores uploaded PDFs
│   └── chromadb_storage/     # Vector database files
├── Test_Cases/              # Example test scenarios
└── Notebooks/               # Development notebooks
```

## 🛠️ How Each Component Works

### 1. RAG Implementation (`rag_chat.py`) 📋
```python
# Core functionality:
- Uses UnstructuredPDFLoader to extract text and images from PDFs
- Processes text through TextProcessor class for summarization
- Handles images through ImageProcessor class
- Stores embeddings in ChromaDB for retrieval
- Uses Gemini multimodal for processing queries with images
```

Key Functions:
- `process_document()`: Extracts content from PDFs
- `MultiModalRAG.query()`: Processes user queries using stored context
- `ChromaMultiModalRetriever`: Handles vector storage and retrieval

### 2. Evaluation System (`evaluation.py`) 📊
```python
# Implements basic metrics:
- BLEU and ROUGE scores for text similarity
- Semantic similarity using sentence-transformers
- Image similarity through Gemini vision model
- Response relevancy scoring
```

### 3. Streamlit Interface (`Streamlit_Chat_app.py`) 🖥️
```python
# Features:
- PDF document upload interface
- Chat interface with response display
- Shows retrieved source images
- Displays context sources in expandable sections
```

## 🚀 Prerequisites

- Python 3.8+
- Poppler (for PDF processing)
- Tesseract OCR

## 📦 Installation

1. Clone the repository:
```bash
[git clone https://github.com/yourusername/multimodal-rag.git](https://github.com/Sanjith-3/multimodal-rag-gemini-fastapi.git)
cd multimodal-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create `.env` file:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## 🔄 How It Works

### Document Processing Flow
1. When a PDF is uploaded:
   - `UnstructuredPDFLoader` extracts text and images
   - Text is chunked and summarized
   - Images are processed and described
   - All content is embedded and stored in ChromaDB

### Query Processing
1. User sends a query through Streamlit interface
2. System:
   - Embeds the query
   - Retrieves relevant text and images from ChromaDB
   - Formats prompt with retrieved context:
   ```python
   messages = [
       {"type": "image_url", "image_url": {...}},  # For retrieved images
       {"type": "text", "text": "Context: ... Query: ..."}
   ]
   ```
   - Sends to Gemini multimodal for response generation
   - Displays response with source attribution

## 🌟 Why These Choices?

### Gemini Multimodal & API Limits
- While API limits (12 calls/day) restrict extensive use, this approach:
  - Avoids heavy computational requirements of local models
  - Provides reliable multimodal capabilities without infrastructure overhead
  - Offers good response quality for demonstration purposes

#### Why Not Other Services?
- Services like Groq and SambaNova:
  - Limited to single image per input
  - May have higher latency

## 🚧 Limitations & Future Improvements

- API call limits restrict production use
- Basic evaluation metrics could be enhanced
- Document processing could be optimized
- More sophisticated vector storage implementations possible
- Response generation could be more nuanced

## 🎯 Usage

1. Start FastAPI server:
```bash
cd API
uvicorn rag_chat:app --reload
```

2. Launch Streamlit:
```bash
streamlit run Streamlit_Chat_app.py
```

### API Endpoints 🔌

```http
POST /upload-doc  # Upload and process PDFs
POST /chat       # Process queries
POST /evaluate   # Evaluate responses
```

## 📈 Testing

The `Test_Cases` directory contains example scenarios for testing response quality and system behavior.

## 📚 Dependencies

Main packages:
- `fastapi`: API framework
- `streamlit`: UI interface
- `google-generativeai`: Gemini integration
- `chromadb`: Vector storage
- `sentence-transformers`: Text embedding
- `pdf2image`: PDF processing
- `pillow`: Image handling
- `nltk`: Text processing

## 🤝 Contributing

Feel free to fork, improve, and create pull requests. This is a basic implementation that can benefit from community enhancements.
