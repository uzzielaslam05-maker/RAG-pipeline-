# 🧠 DocuMind AI
Live Here: https://uzzielaslam05-maker-rag-pipeline--app-6kwz9s.streamlit.app/

> **Ask questions. Get grounded answers.**
>
> DocuMind AI is a Retrieval-Augmented Generation (RAG) application that lets users interact with document content through an AI-powered interface. Instead of relying only on an LLM's general knowledge, the system is designed to retrieve relevant information from the supplied document and use that context to generate a more grounded response.

---

## ✨ Overview

DocuMind AI brings together document processing, semantic search, vector storage, and large-language-model generation into a single application.

The project was built around a simple idea:

**Your documents should become the source of truth for the conversation.**

The initial pipeline was designed with:

- **Python** for the application and RAG pipeline
- **PyPDF** for PDF loading and processing
- **Sentence Transformers** for semantic embeddings
- **ChromaDB** for local vector storage
- **Groq** for LLM-powered response generation
- **Streamlit** for the interactive web interface
- **Google Colab** as a convenient development environment during the early pipeline work

The application also includes user-facing handling for LLM rate limits, including an attempt to determine the actual retry interval and present the expected recovery time in the visitor's own local timezone.

---

## 🎯 Why RAG?

A normal LLM can answer questions from its pretrained knowledge, but that does not make it automatically knowledgeable about a private PDF.

RAG adds a retrieval layer:

```text
                 ┌──────────────────┐
                 │   User's PDF     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Document Parsing │
                 │     (PyPDF)      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text → Embedding │
                 │ Sentence        │
                 │ Transformers    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    ChromaDB      │
                 │  Vector Store    │
                 └────────┬─────────┘
                          │
                  Relevant context
                          │
                          ▼
┌──────────────┐   ┌──────────────────┐
│ User Query   │──▶│ Retrieval + LLM  │
└──────────────┘   │     (Groq)        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Grounded Answer  │
                   └──────────────────┘
```

This separation is important: **retrieval finds the relevant evidence; generation turns that evidence into a useful answer.**

---

## 🚀 Core Capabilities

### 📄 Document Interaction
Load a PDF and prepare its content for downstream retrieval and question answering.

### 🔎 Semantic Retrieval
The system is designed around embeddings rather than simple keyword matching, allowing queries to retrieve conceptually relevant passages.

### 🗄️ Vector Storage
ChromaDB provides the vector database layer for storing and searching document embeddings locally during development.

### 🤖 LLM Generation
Groq is used as the generation provider for producing responses from the retrieved context.

### 🖥️ Streamlit Interface
The project exposes the functionality through a Streamlit application rather than requiring users to interact directly with Python code.

### 🌍 Timezone-Aware Rate-Limit Feedback
When Groq returns a rate-limit error, the application attempts to determine the provider's retry interval. If successful, the interface can calculate the recovery time in the visitor's browser, allowing the displayed time to correspond to the user's local timezone.

If the provider's retry information cannot be determined, the application falls back to a more general wait message.

### 🌑 Dark UI
The project includes Streamlit configuration for a darker visual presentation through `.streamlit/config.toml`.

---

## 🧩 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| PDF handling | PyPDF |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| LLM provider | Groq |
| Development environment | Google Colab / Local Python |
| Source control | Git + GitHub |

---

## 📁 Project Structure

A typical structure for the project is:

```text
DocuMind-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── your_document.pdf
│
└── README.md
```

> The exact repository structure may evolve as additional RAG stages and application components are added.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the initial PDF/RAG pipeline, the core dependencies included:

```bash
pip install pypdf chromadb sentence-transformers
```

### 4. Configure the LLM provider

The application uses Groq for generation.

Create a Groq API key and provide it through your environment or the secret-management mechanism expected by the application.

**Do not hard-code API keys inside the repository.**

For local development, an environment variable can be used:

```bash
GROQ_API_KEY=your_api_key_here
```

On Streamlit deployment, store the key using Streamlit's secrets mechanism rather than committing it to Git.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be opened in your browser.

---

## ☁️ Google Colab Development

The early RAG pipeline was also designed to run conveniently in Google Colab.

Install the core dependencies:

```python
!pip install pypdf chromadb sentence-transformers -q
```

For a quick PDF upload:

```python
from google.colab import files

uploaded = files.upload()
```

The uploaded document can then be accessed from `/content/`.

For persistent files, Google Drive can be mounted:

```python
from google.colab import drive

drive.mount('/content/drive')
```

---

## 🔐 Security

Never commit secrets.

Make sure files such as these are ignored when appropriate:

```text
.env
.streamlit/secrets.toml
__pycache__/
.venv/
*.pyc
```

If an API key has accidentally been committed, **do not simply delete it from the latest commit and assume it is safe**. Treat the key as compromised and rotate/revoke it.

---

## 🛡️ Rate-Limit Handling

External LLM APIs can impose request limits.

DocuMind AI includes handling intended to make this failure mode less confusing for users.

The application attempts to:

1. Detect a rate-limit response.
2. Read the provider's retry interval when available.
3. Parse the retry duration when necessary.
4. Convert that duration into a recovery time.
5. Calculate the displayed time in the browser so it follows the visitor's local timezone.
6. Fall back to a generic wait message if the provider does not expose usable retry information.

This is deliberately better than hard-coding a statement such as "try again in one minute," because provider-side retry windows are not guaranteed to be exactly one minute.

---

## 🧠 RAG Pipeline

The intended pipeline can be summarized as:

```text
PDF
 │
 ▼
Load / Parse
 │
 ▼
Extract Text
 │
 ▼
Chunk Document
 │
 ▼
Generate Embeddings
 │
 ▼
Store in ChromaDB
 │
 ▼
User Question
 │
 ▼
Embed Question
 │
 ▼
Similarity Search
 │
 ▼
Retrieve Relevant Context
 │
 ▼
Send Context + Question to LLM
 │
 ▼
Generate Answer
```

The critical principle is that the model should receive relevant document context rather than being expected to answer from general model knowledge alone.

---

## 📌 Design Principles

### Ground the model
Retrieved document context should be the foundation of answers about the uploaded material.

### Separate retrieval from generation
Vector search and LLM generation solve different problems. Keeping them conceptually separate makes the system easier to debug and improve.

### Fail clearly
API failures and rate limits should produce useful feedback rather than cryptic exceptions.

### Keep secrets out of source control
Credentials belong in environment variables or platform-managed secrets.

### Build incrementally
The project was developed step-by-step, beginning with PDF loading before moving toward the complete RAG workflow and deployed application.

---

## 🔭 Future Improvements

The architecture leaves room for several meaningful upgrades:

- Support for multiple PDFs and document collections
- Better document chunking strategies
- Metadata-aware retrieval
- Source/page citations in generated answers
- Retrieval-quality evaluation
- Hybrid keyword + semantic search
- Reranking retrieved chunks
- Conversation memory
- Streaming LLM responses
- Document-level access controls
- Improved observability and error logging
- Automated tests for retrieval and generation
- More robust handling of large documents

The biggest quality improvement would be **measuring retrieval quality instead of assuming that the top-k chunks are good enough**. A RAG system is only as reliable as the context it retrieves.

---

## ⚠️ Limitations

DocuMind AI depends on several external components and therefore is not magically immune to failure.

Potential limitations include:

- LLM provider rate limits
- Embedding and retrieval quality depending on document structure
- PDFs with complex layouts or scanned pages may require additional processing
- Incorrect or incomplete retrieval can lead to weak generated answers
- Provider availability can affect generation
- Local ChromaDB storage is not the same thing as a production-grade distributed vector database

**RAG reduces the model's dependence on unsupported knowledge; it does not guarantee factual correctness.**

---

## 🤝 Contributing

Contributions are welcome.

A clean contribution workflow is:

```text
Fork
  ↓
Create a feature branch
  ↓
Make changes
  ↓
Test locally
  ↓
Commit
  ↓
Open a Pull Request
```

For substantial changes, explain:

- What changed
- Why it changed
- How it was tested
- Any new dependencies or configuration required

---

## 📜 License

Add your preferred open-source license here before publishing the repository publicly.

For example:

```text
MIT License
```

Do not claim a license that has not actually been added to the repository.

---

## 👨‍💻 Project Status

**Status:** Active development

DocuMind AI is being developed incrementally from a foundational PDF/RAG pipeline into a polished document-question-answering application.

---

<div align="center">

### DocuMind AI

**Turn documents into conversations.**

Built with Python · Streamlit · ChromaDB · Sentence Transformers · Groq

</div>
