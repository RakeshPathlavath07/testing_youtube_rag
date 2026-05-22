import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

# Helper to format seconds into MM:SS
def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# Helper to extract YouTube ID
def extract_video_id(url_or_id):
    if len(url_or_id) == 11:
        return url_or_id
    match = re.search(r"(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/|^)([^#\&\?]{11})", url_or_id)
    return match.group(1) if match else None

# Format documents with timestamps
def format_docs(retrieved_docs):
    return "\n\n".join(
        f"[Timestamp: {seconds_to_timestamp(doc.metadata['start'])} to {seconds_to_timestamp(doc.metadata['end'])}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

# Use API-based Embeddings to avoid local PyTorch dependencies
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

# App UI
st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")
st.title("YouTube Video RAG Assistant")

if not hf_token:
    st.error("Hugging Face API token not found. Please ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file.")
    st.stop()

# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# Input Section
video_input = st.text_input("Enter YouTube Video URL or 11-character Video ID:")

if st.button("Process Video"):
    vid_id = extract_video_id(video_input)
    if not vid_id:
        st.error("Invalid YouTube URL or Video ID.")
    else:
        with st.spinner("Fetching transcript and processing chunks..."):
            try:
                api = YouTubeTranscriptApi()
                transcript_list = api.fetch(vid_id, languages=['en'])
                
                # Use object dot notation instead of dictionary subscripts
                transcript_with_timestamps = []
                for chunk in transcript_list:
                    transcript_with_timestamps.append({
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.start + chunk.duration
                    })
                
                # Chunking logic
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
                        documents.append(
                            Document(
                                page_content=current_text.strip(),
                                metadata={
                                    "start": current_start,
                                    "end": item["end"]
                                }
                            )
                        )
                        current_text = current_text[-CHUNK_OVERLAP:]
                        current_start = item["start"]

                if current_text.strip():
                    documents.append(
                        Document(
                            page_content=current_text.strip(),
                            metadata={
                                "start": current_start,
                                "end": transcript_with_timestamps[-1]["end"]
                            }
                        )
                    )

                # Initialize FAISS index
                embeddings = get_embeddings_model()
                st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
                st.session_state.current_video_id = vid_id
                st.success(f"Video parsed successfully! Created {len(documents)} chunks.")

            except TranscriptsDisabled:
                st.error("Captions are disabled/unavailable for this video.")
            except Exception as e:
                st.error(f"Failed to process video: {e}")

# Query Section
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("Query the Video Content")
    
    question = st.text_input("Ask a question about the video:")
    
    if st.button("Ask Assistant"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing context..."):
                try:
                    # Switched to Qwen2.5-7B-Instruct for reliable serverless access
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

                    parallel_chain = RunnableParallel({
                        'context': retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                    })
                    
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt | chat_model | parser
                    
                    response = main_chain.invoke(question)
                    st.markdown("### Answer:")
                    st.write(response)

                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")