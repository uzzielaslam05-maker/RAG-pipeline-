import re
import uuid

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #0D1326 0%, #080B14 55%) !important;
}

/* ---------- Hero ---------- */
.dm-hero {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 8px 0 28px 0;
    border-bottom: 1px solid #1B2340;
    margin-bottom: 28px;
}
.dm-hero-icon {
    flex-shrink: 0;
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    background: linear-gradient(135deg, #12356B 0%, #0D1B33 100%);
    border: 1px solid #2A5CB8;
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.25), inset 0 0 12px rgba(61, 139, 255, 0.15);
}
.dm-hero-text h1 {
    font-family: 'Sora', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #EDF1FC;
    margin: 0;
    letter-spacing: -0.02em;
}
.dm-hero-text p {
    font-family: 'Inter', sans-serif;
    color: #8892B0;
    font-size: 0.92rem;
    margin: 4px 0 0 0;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #0C1122 !important;
    border: 1.5px dashed #2C3660 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #3D8BFF !important;
    background: #0E1530 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #3D8BFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stWidgetLabel"] p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    color: #6E7A9C;
}

/* ---------- Alerts ---------- */
div[data-testid="stAlertContainer"] {
    background: #0E1530 !important;
    border: 1px solid #223061 !important;
    border-left: 3px solid #3D8BFF !important;
    border-radius: 10px !important;
    color: #C7D2F0 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: #0F1528 !important;
    border: 1px solid #1E2748 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
}
[data-testid="stChatInput"] textarea {
    background: #0C1122 !important;
    border: 1px solid #2C3660 !important;
    border-radius: 10px !important;
    color: #E7ECFB !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="dm-hero">
        <div class="dm-hero-icon">🧠</div>
        <div class="dm-hero-text">
            <h1>DocuMind AI</h1>
            <p>Upload a PDF and get grounded answers, pulled straight from the source.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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


# ---------- Rate-limit handling ----------

class RateLimitWait(Exception):
    """Raised when Groq's API rate limit is hit. Carries the wait time, if known."""

    def __init__(self, retry_seconds: float | None):
        self.retry_seconds = retry_seconds
        super().__init__("Rate limit hit")


class GenerationError(Exception):
    """Raised for any other failure while generating an answer."""


def parse_retry_after_seconds(exc: Exception) -> float | None:
    """Pull the wait time out of a Groq rate-limit error, in seconds."""
    # Prefer the raw HTTP header if the SDK exposes it — it's authoritative.
    response = getattr(exc, "response", None)
    if response is not None:
        header_val = response.headers.get("retry-after")
        if header_val:
            try:
                return float(header_val)
            except ValueError:
                pass

    # Fall back to parsing Groq's error message, e.g. "try again in 6m11.52s"
    match = re.search(r"try again in\s+(?:([\d.]+)m)?([\d.]+)s", str(exc))
    if match:
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    return None


def render_rate_limit_banner(retry_seconds: float | None) -> None:
    """Show a rate-limit notice with the recovery time in the visitor's own
    local timezone, computed client-side since the server doesn't know it."""
    if retry_seconds is None:
        st.error(
            "DocuMind AI is getting a lot of questions right now and hit its "
            "rate limit. Please wait about a minute and try again."
        )
        return

    components.html(
        f"""
        <div style="
            background:#0E1530;
            border:1px solid #223061;
            border-left:3px solid #E2574C;
            border-radius:10px;
            padding:14px 16px;
            font-family:'JetBrains Mono', monospace;
            font-size:0.85rem;
            color:#C7D2F0;
            line-height:1.5;
        ">
            DocuMind AI hit its rate limit. It'll be ready again at
            <strong id="dm-retry-time">…</strong> your time.
        </div>
        <script>
            const retryAfterSeconds = {retry_seconds};
            const readyTime = new Date(Date.now() + retryAfterSeconds * 1000);
            const formatted = readyTime.toLocaleTimeString([], {{
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            }});
            document.getElementById('dm-retry-time').innerText = formatted;
        </script>
        """,
        height=70,
    )


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

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
    except Exception as e:
        error_text = str(e).lower()
        if "rate_limit" in error_text or "429" in error_text:
            raise RateLimitWait(parse_retry_after_seconds(e)) from e
        raise GenerationError(
            "Something went wrong generating an answer. Please try again."
        ) from e

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
    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

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
        try:
            with st.spinner("Thinking..."):
                answer = generate_answer(st.session_state.collection, query)
        except RateLimitWait as e:
            render_rate_limit_banner(e.retry_seconds)
        except GenerationError as e:
            st.error(str(e))
        else:
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.chat_history.append((query, answer))
else:
    st.info("Drop a PDF above to start asking questions.")
