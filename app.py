import uuid

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 Ask Questions About Your PDF")
st.caption("Upload a PDF, then ask questions about its content. Powered by ChromaDB + Groq.")


# ---------- Cached resources (loaded once per app instance) ----------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Add it in Streamlit Cloud under "
            "App settings → Secrets (see README.md)."
        )
        st.stop()
    return Groq(api_key=api_key)


embedding_model = load_embedding_model()
groq_client = get_groq_client()


# ---------- RAG pipeline functions ----------

def extract_text(reader: PdfReader) -> str:
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_index(chunks: list[str]):
    """Create a fresh in-memory Chroma collection for this uploaded PDF."""
    client = chromadb.Client()  # in-memory client, isolated per session
    collection = client.create_collection(name=f"pdf_{uuid.uuid4().hex}")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False, batch_size=32)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=chunks)
    return collection


def retrieve_relevant_chunks(collection, query: str, top_k: int = 3) -> list[str]:
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]


def generate_answer(collection, query: str, top_k: int = 3) -> str:
    retrieved_chunks = retrieve_relevant_chunks(collection, query, top_k=top_k)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the information in the context below, taken from a document.
Give a complete, explanatory answer. If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------- Session state ----------

if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ---------- UI ----------

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.current_file != uploaded_file.name:
    with st.spinner("Reading and indexing PDF... this may take a minute"):
        reader = PdfReader(uploaded_file)
        text = extract_text(reader)
        chunks = chunk_text(text)
        st.session_state.collection = build_index(chunks)
        st.session_state.current_file = uploaded_file.name
        st.session_state.chat_history = []
    st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

if st.session_state.collection is not None:
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    query = st.chat_input("Ask a question about the PDF...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.spinner("Thinking..."):
            answer = generate_answer(st.session_state.collection, query)
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat_history.append((query, answer))
else:
    st.info("Upload a PDF above to get started.")
