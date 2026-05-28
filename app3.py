import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

# ── Helper: seconds → MM:SS ──────────────────────────────────────────────────
def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# ── Helper: extract YouTube video ID ────────────────────────────────────────
def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11:
        return url_or_id
    pattern = r"(?:v=|/v/|embed/|youtu\.be/|/shorts/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else None

# ── Helper: sliding-window text splitter ────────────────────────────────────
def split_text_manually(text, chunk_size=800, overlap=200):
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if len(text) - start < overlap:
            break
    return chunks

# ── Helper: format retrieved docs with timestamps ───────────────────────────
def format_docs(retrieved_docs):
    return "\n\n".join(
        f"[Timestamp: {seconds_to_timestamp(doc.metadata['start'])} to {seconds_to_timestamp(doc.metadata['end'])}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

# ── Embeddings model (cached) ────────────────────────────────────────────────
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

# ── Load cookies from Streamlit secrets ──────────────────────────────────────
def load_cookies_from_secrets() -> str | None:
    """
    Writes cookie content from st.secrets to a temp file and returns its path.
    Returns None if the secret isn't configured.
    """
    try:
        cookie_content = st.secrets["youtube"]["cookies"]
        cookie_path = "/tmp/yt_cookies.txt"
        with open(cookie_path, "w") as f:
            f.write(cookie_content)
        return cookie_path
    except (KeyError, FileNotFoundError):
        return None

# ── Fetch transcript with optional cookie path ───────────────────────────────
def fetch_transcript(vid_id: str, cookie_path: str | None = None):
    """
    Attempts to fetch the transcript.
    - If cookie_path is provided, passes it to the API for authenticated requests.
    - Falls back to unauthenticated if cookies aren't set.
    """
    if cookie_path and os.path.exists(cookie_path):
        api = YouTubeTranscriptApi(cookies=cookie_path)
    else:
        api = YouTubeTranscriptApi()
    return api.fetch(vid_id, languages=['en'])

# ════════════════════════════════════════════════════════════════════════════
# APP UI
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")
st.title("YouTube Video RAG Assistant")

if not hf_token:
    st.error("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("vector_store", None),
    ("current_video_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# CHANGED: Load cookie_path from secrets instead of defaulting to None
if "cookie_path" not in st.session_state:
    st.session_state.cookie_path = load_cookies_from_secrets()

# ── Cookie status indicator ───────────────────────────────────────────────────
if st.session_state.cookie_path:
    st.success("🍪 YouTube cookies loaded from secrets.")
else:
    st.warning("⚠️ No YouTube cookies configured. Add `[youtube] cookies = ...` to your Streamlit secrets.")

# ── Video URL input ───────────────────────────────────────────────────────────
st.markdown("---")
video_input = st.text_input("Enter YouTube Video URL or 11-character Video ID:")

input_vid_id = extract_video_id(video_input)
if (
    input_vid_id
    and st.session_state.current_video_id
    and st.session_state.current_video_id != "Manual Input"
    and input_vid_id != st.session_state.current_video_id
):
    st.warning("⚠️ Video ID changed. Click **Process Video** to load the new video.")

if st.button("Process Video"):
    vid_id = extract_video_id(video_input)
    if not vid_id:
        st.error("Could not parse a valid YouTube Video ID. Please check the URL.")
    else:
        with st.spinner("Fetching transcript and processing chunks..."):
            try:
                # Pass cookie path from session state
                transcript_list = fetch_transcript(vid_id, st.session_state.cookie_path)

                transcript_with_timestamps = [
                    {"text": chunk.text, "start": chunk.start, "end": chunk.start + chunk.duration}
                    for chunk in transcript_list
                ]

                # Sliding-window chunking preserving timestamps
                CHUNK_CHAR_LIMIT = 800
                CHUNK_OVERLAP = 200
                documents = []
                current_text = ""
                current_start = None

                for item in transcript_with_timestamps:
                    if current_start is None:
                        current_start = item["start"]
                    current_text += " " + item["text"]
                    if len(current_text) >= CHUNK_CHAR_LIMIT:
                        documents.append(Document(
                            page_content=current_text.strip(),
                            metadata={"start": current_start, "end": item["end"]}
                        ))
                        current_text = current_text[-CHUNK_OVERLAP:]
                        current_start = item["start"]

                if current_text.strip():
                    documents.append(Document(
                        page_content=current_text.strip(),
                        metadata={"start": current_start, "end": transcript_with_timestamps[-1]["end"]}
                    ))

                embeddings = get_embeddings_model()
                st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
                st.session_state.current_video_id = vid_id
                st.success(f"✅ Successfully processed video: `{vid_id}`!")

            except TranscriptsDisabled:
                st.error("Captions are disabled/unavailable for this video.")
            except Exception as e:
                err = str(e)
                if "blocking requests" in err or "cloud provider" in err or "429" in err:
                    st.error("⚠️ YouTube is blocking requests from this server's IP.")
                    st.info(
                        "**Fix:** Add your `cookies.txt` content to Streamlit secrets under "
                        "`[youtube] cookies = ...`, then reboot the app. "
                        "Or use the Manual Transcript fallback below."
                    )
                else:
                    st.error(f"Failed to process video: {e}")

# ── Manual transcript fallback ────────────────────────────────────────────────
st.markdown("---")
with st.expander("📝 Manual Transcript Fallback"):
    st.write("Copy the transcript from YouTube's '...' → 'Show transcript' panel and paste below.")
    manual_text = st.text_area("Paste raw transcript text here:", height=200)
    if st.button("Process Manual Text"):
        if not manual_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Processing manual text..."):
                try:
                    chunks = split_text_manually(manual_text, chunk_size=800, overlap=200)
                    documents = [
                        Document(page_content=chunk, metadata={"start": 0, "end": 0})
                        for chunk in chunks
                    ]
                    embeddings = get_embeddings_model()
                    st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
                    st.session_state.current_video_id = "Manual Input"
                    st.success(f"✅ Processed manual text into {len(documents)} chunks.")
                except Exception as e:
                    st.error(f"Failed to process manual text: {e}")

# ── Query section ─────────────────────────────────────────────────────────────
if st.session_state.vector_store is not None:
    st.divider()
    if st.session_state.current_video_id == "Manual Input":
        st.info("👉 **Currently Querying:** Manually pasted transcript.")
    else:
        st.info(f"👉 **Currently Querying Video:** `https://youtube.com/watch?v={st.session_state.current_video_id}`")

    question = st.text_input("Ask a question about the active video:")

    if st.button("Ask Assistant"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating response..."):
                try:
                    llm = HuggingFaceEndpoint(
                        repo_id="Qwen/Qwen2.5-7B-Instruct",
                        task="text-generation",
                        max_new_tokens=512,
                        temperature=0.5,
                        huggingfacehub_api_token=hf_token
                    )
                    chat_model = ChatHuggingFace(llm=llm)

                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    retrieved_docs = retriever.invoke(question)

                    with st.expander("🔍 Retrieved Context Chunks (With Timestamps)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(
                                f"**Chunk {i+1}** "
                                f"[{seconds_to_timestamp(doc.metadata['start'])} - {seconds_to_timestamp(doc.metadata['end'])}]"
                            )
                            st.write(doc.page_content)
                            st.divider()

                    prompt = PromptTemplate(
                        template="""
You are a video transcript analysis assistant.

Answer STRICTLY using the transcript context below.
Do NOT use outside knowledge.

TASK:
1. Decide whether the user's topic is discussed in the video.
2. If YES:
   - Clearly explain what is discussed
   - Give a short summary relevant to the question
   - List the EXACT video timestamps where it is discussed
3. If NO:
   - Say only: "NO. Sorry, the topic is not discussed in this video."

RULES:
- Start your answer with YES or NO
- Timestamps MUST be in MM:SS format
- Use ONLY timestamps present in the context

Transcript context:
{context}

Question:
{question}
""",
                        input_variables=["context", "question"]
                    )

                    context_text = format_docs(retrieved_docs)
                    final_prompt = prompt.format(context=context_text, question=question)
                    response = chat_model.invoke(final_prompt)

                    st.markdown("### Answer:")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
