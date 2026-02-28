# Multimodal RAG Q&A System (Gemini + FAISS + OCR + Streamlit)

An AI-powered multimodal document question-answering system that allows users to upload **any file type** PDF, Word, PowerPoint, Excel, or Images and ask questions about their content. The system uses a tiered image processing pipeline (OCR first, Gemini Vision as fallback), semantic search (FAISS), and Google Gemini models to provide accurate, context-aware answers.

---

## Key Features

- Upload and process PDF, Word, PowerPoint, Excel, and Image files
- Automatic image extraction from all document types
- Tiered image processing: Disk Cache > Duplicate Filter > Tesseract OCR (free) > Gemini Vision (fallback)
- Smart text chunking with sliding window overlap
- Semantic search using FAISS vector store
- Local embeddings with Sentence Transformers (no API cost)
- Answer generation using Google Gemini
- Persistent caption cache — same image never processed twice
- In-memory answer cache — repeated questions cost zero API calls
- Live API quota dashboard with daily usage tracker
- Proactive rate limiter to stay within free tier limits
- Confidence scoring for all responses
- Interactive UI built with Streamlit
- Debug view for retrieved context and image previews

---

## Project Structure

```bash
multimodal_rag/
├── app.py                              # Main Streamlit entry point
├── config.py                           # All config from environment variables
├── requirements.txt                    # Dependencies
├── .env.example                        # Environment variable template
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── interfaces.py                   # Abstract base classes (SOLID)
│   └── models.py                       # DocumentChunk, QAResponse, RetrievalResult
│
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py                     # Orchestrator (Facade pattern)
│   ├── extractor_registry.py           # Plugin registry (Open/Closed Principle)
│   ├── chunker.py                      # Sliding-window text chunker
│   ├── image_captioner.py              # Tiered captioning: cache > OCR > Vision
│   ├── image_filter.py                 # Skip tiny, blank, duplicate images
│   ├── ocr_engine.py                   # Tesseract OCR with preprocessing
│   └── extractors/
│       ├── __init__.py
│       ├── base_extractor.py           # Shared factory helpers (DRY)
│       ├── pdf_extractor.py            # pdfplumber + PyMuPDF
│       ├── word_extractor.py           # python-docx
│       ├── pptx_extractor.py           # python-pptx
│       ├── excel_extractor.py          # openpyxl
│       └── image_extractor.py          # Pillow raw bytes
│
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py                     # SentenceTransformer (cached, local)
│   └── vector_store.py                 # FAISS index (swappable via interface)
│
├── generation/
│   ├── __init__.py
│   ├── prompt_builder.py               # Multimodal prompt construction
│   └── gemini_generator.py             # Gemini API call and QAResponse
│
├── services/
│   ├── __init__.py
│   ├── rag_service.py                  # Main orchestrator facade
│   ├── cache_service.py                # In-memory answer cache with TTL
│   ├── rate_limiter.py                 # Dual-window rate limiting
│   └── caption_cache_service.py        # Persistent disk cache for image captions
│
├── ui/
│   ├── __init__.py
│   ├── components.py                   # Sidebar, upload, QA sections
│   └── helpers.py                      # Pure display utilities
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py                   # Temp file handling and validation
│   └── logger.py                       # Centralized stdout logging
│
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_extractor_registry.py
    └── test_models.py
```

---

## System Architecture

```
User Uploads File
       |
       v
+----------------+
|  Extractor     |  PDF / Word / PPT / Excel / Image
|  Registry      |  Picks the right extractor automatically
+----------------+
       |
       v
+----------------+
|  Image         |  Disk Cache -> Filter -> OCR -> Gemini Vision
|  Captioner     |  Free options always tried first
+----------------+
       |
       v
+----------------+
|  Text          |  Sliding window with overlap
|  Chunker       |  1000 chars, 200 char overlap
+----------------+
       |
       v
+----------------+
|  Embedder      |  all-MiniLM-L6-v2 (local, free)
|                |  Converts chunks to 384-dim vectors
+----------------+
       |
       v
+----------------+
|  FAISS Index   |  IndexFlatL2
|                |  Exact similarity search
+----------------+
       |
  User Query
       |
       v
+----------------+
|  Retriever     |  Top-K most similar chunks
|                |  Similarity threshold filter
+----------------+
       |
       v
+----------------+
|  Prompt        |  Text + base64 images interleaved
|  Builder       |  Multimodal context construction
+----------------+
       |
       v
+----------------+
|  Gemini API    |  Answer generation
|                |  gemini-2.0-flash (default)
+----------------+
       |
       v
   Final Answer
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Interactive web interface |
| LLM | Google Gemini (`2.0-flash`, `2.5-flash`, `2.5-pro`) | Answer generation |
| Vision | Google Gemini Vision | Chart and diagram understanding |
| OCR | Tesseract + pytesseract | Free local text extraction from images |
| RAG Pipeline | Custom implementation | Full control over retrieval logic |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | Local embeddings, zero API cost |
| Vector Store | FAISS (`IndexFlatL2`) | Exact search, no external dependency |
| PDF Parsing | pdfplumber + PyMuPDF | Text, table, and image extraction |
| Word Parsing | python-docx | Paragraphs, tables, embedded images |
| PPT Parsing | python-pptx | Slide text, tables, embedded images |
| Excel Parsing | openpyxl | Sheet data as readable text |
| Image Processing | Pillow | Preprocessing for better OCR accuracy |
| Numerical | NumPy | Efficient vector operations |
| API Client | google-genai | Gemini integration |
| Caching | In-memory + disk JSON | Answer and caption caching |
| State Management | Streamlit session state | Maintain UI state across reruns |
| Language | Python 3.12+ | Ecosystem support |

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
```

### 2. Create Virtual Environment

```bash
python -m venv myenv

source myenv/bin/activate       # Mac / Linux
myenv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR Engine

```bash
# Windows — download and run installer:
# https://github.com/UB-Mannheim/tesseract/wiki

# Mac
brew install tesseract

# Linux
sudo apt install tesseract-ocr
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here

# Windows only — path to Tesseract executable
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional
DEBUG=false
TEMP_DIR=/tmp/multimodal_rag
```


### 6. Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```


---

## Confidence Scoring

Every answer includes a confidence score based on the average similarity of retrieved chunks:

| Score | Meaning |
|---|---|
| Above 0.5 | High confidence — strong context match |
| 0.3 to 0.5 | Medium confidence — partial context match |
| Below 0.3 | Low confidence — weak or no context match |

---
