import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# -----------------------------
# Setup
# -----------------------------
load_dotenv()  # Load variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎬", layout="centered")
st.title("🎬 YouTube Video Summarizer")
st.write("Paste a YouTube video URL and get a concise summary of the content.")

# Do NOT stop the app here – just warn the user.
if not OPENAI_API_KEY:
    st.warning(
        "⚠️ OPENAI_API_KEY is not set. You can still see the UI, "
        "but summarization will fail until you add it to your `.env` file."
    )

# -----------------------------
# Helper Functions
# -----------------------------
def extract_video_id(youtube_url: str) -> str:
    """
    Extract YouTube video ID from different URL formats.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
    """
    parsed_url = urlparse(youtube_url)

    # Standard watch URL
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            query = parse_qs(parsed_url.query)
            return query.get("v", [None])[0]
        # Shorts URL format
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1].split("?")[0]

    # youtu.be short URL
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def get_video_transcript(video_id: str, languages=None) -> str:
    """
    Fetch the transcript for a YouTube video and return as a single string.
    """
    if languages is None:
        languages = ["en", "en-IN"]

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video in the requested languages.")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {str(e)}")

    # Join all text pieces into one large string
    full_text = " ".join([item["text"] for item in transcript_list])
    return full_text


def create_docs_from_transcript(transcript_text: str):
    """
    Split transcript into chunks and wrap them as LangChain Documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_text(transcript_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs


def build_summarization_chain(summary_style: str = "concise"):
    """
    Build a LangChain summarization chain (map-reduce) with custom prompts.
    summary_style: 'concise' | 'detailed' | 'bullet-points'
    """
    if summary_style == "detailed":
        style_instruction = (
            "Provide a detailed but clear summary that captures key points, arguments, and examples."
        )
    elif summary_style == "bullet-points":
        style_instruction = (
            "Provide a concise summary in clear bullet points, focusing on the key ideas only."
        )
    else:  # concise
        style_instruction = (
            "Provide a concise paragraph summary focusing on the main ideas, not minor details."
        )

    map_prompt = PromptTemplate(
        template=(
            "You are an assistant that summarizes parts of a YouTube video transcript.\n"
            f"{style_instruction}\n\n"
            "Here is a part of the transcript:\n"
            "-----------------\n"
            "{text}\n"
            "-----------------\n\n"
            "Summary of this part:"
        ),
        input_variables=["text"],
    )

    combine_prompt = PromptTemplate(
        template=(
            "You are an assistant that summarizes YouTube videos for busy people.\n"
            f"{style_instruction}\n\n"
            "You will be given partial summaries of different segments of the same video.\n"
            "Combine them into a single, coherent summary that a new viewer can quickly understand.\n\n"
            "Partial summaries:\n"
            "-----------------\n"
            "{text}\n"
            "-----------------\n\n"
            "Final summary:"
        ),
        input_variables=["text"],
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",  # or another model you prefer
        temperature=0.3,
        api_key=OPENAI_API_KEY,  # CHANGED: use api_key instead of openai_api_key
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False,
    )
    return chain


# -----------------------------
# Streamlit UI
# -----------------------------
youtube_url = st.text_input(
    "Enter YouTube video URL:",
    placeholder="https://www.youtube.com/watch?v=...",
)

summary_style = st.selectbox(
    "Summary style",
    ["concise", "detailed", "bullet-points"],
    index=0,
)

if st.button("Summarize Video", type="primary"):
    # Now we check the API key at the moment of action
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Please add it to your `.env` file and restart the app.")
        st.stop()

    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    with st.spinner("Processing video..."):
        # 1. Extract video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Could not extract video ID. Please check the URL.")
            st.stop()

        # 2. Get transcript
        try:
            transcript_text = get_video_transcript(video_id)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        if not transcript_text.strip():
            st.error("Transcript is empty or could not be processed.")
            st.stop()

        # 3. Create LangChain Documents from transcript
        docs = create_docs_from_transcript(transcript_text)

        # 4. Build summarization chain
        chain = build_summarization_chain(summary_style=summary_style)

        # 5. Run summarization
        try:
            summary = chain.run(docs)
        except Exception as e:
            st.error(f"Failed to summarize video: {str(e)}")
            st.stop()

    # 6. Display result
    st.subheader("📝 Video Summary")
    if summary_style == "bullet-points":
        st.markdown(summary)
    else:
        st.write(summary)

    # Optional: show raw transcript toggle
    with st.expander("Show raw transcript (for reference)"):
        st.text_area("Transcript", value=transcript_text, height=300)
