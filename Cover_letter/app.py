import os
from io import StringIO

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import PyPDF2


# -----------------------------
# Utility: Extract text from uploaded resume
# -----------------------------
def extract_text_from_file(uploaded_file) -> str:
    """
    Extracts text from an uploaded file.
    Supports PDF and plain text. You can extend this for DOCX if needed.
    """
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    # Handle PDF files
    if filename.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text).strip()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    # Handle plain text files
    if filename.endswith(".txt"):
        try:
            stringio = StringIO(
                uploaded_file.getvalue().decode("utf-8", errors="ignore")
            )
            return stringio.read().strip()
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return ""

    st.warning("Unsupported file type. Please upload a PDF or TXT resume.")
    return ""


# -----------------------------
# LangChain: LLM + Prompt
# -----------------------------
def get_llm(openai_api_key: str) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance.
    Adjust model name if needed (e.g., 'gpt-4o' or another OpenAI-compatible model).
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        openai_api_key=openai_api_key,
    )


def get_cover_letter_prompt() -> ChatPromptTemplate:
    """
    Returns a LangChain ChatPromptTemplate to generate a tailored cover letter.
    """
    template = """
You are an assistant that writes tailored, concise, professional cover letters.

You will be given:
- The candidate's resume
- The target job role
- The company name
- Optional extra details or job description

Write a one-page cover letter (around 200–400 words) that:
- Highlights the most relevant experience and skills from the resume
- Clearly aligns the candidate with the job role and the company's context
- Uses a confident, positive, and professional tone
- Does NOT fabricate experience or skills that are not present in the resume
- Does NOT copy large sections of the resume verbatim
- Starts with a clear greeting and ends with a polite closing.

Return only the cover letter text. Do not include explanations.

Candidate Resume:
\"\"\" 
{resume_text}
\"\"\"

Job Role: {job_role}
Company Name: {company_name}
Additional Details / Job Description (if any):
\"\"\" 
{extra_details}
\"\"\"
"""
    return ChatPromptTemplate.from_template(template)


def generate_cover_letter(
    llm: ChatOpenAI,
    resume_text: str,
    job_role: str,
    company_name: str,
    extra_details: str,
) -> str:
    """
    Uses LangChain (LLM + prompt) to generate the cover letter.
    """
    prompt = get_cover_letter_prompt()
    messages = prompt.format_messages(
        resume_text=resume_text,
        job_role=job_role,
        company_name=company_name,
        extra_details=extra_details or "Not provided",
    )

    response = llm.invoke(messages)
    return response.content.strip()


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Cover Letter Generator",
        page_icon="📄",
        layout="centered",
    )

    st.title("📄 AI-Powered Cover Letter Generator")
    st.write(
        "Upload your resume and provide job details. "
        "This app will generate a tailored cover letter using LangChain and a chat-based LLM."
    )

    # ----------------------------------------
    # Load API Key securely from .env
    # ----------------------------------------
    load_dotenv()  # Loads variables from .env into environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        st.error(
            "OpenAI API key not found. Please create a `.env` file in the project "
            "root and add a line:\n\n`OPENAI_API_KEY=your_api_key_here`"
        )
        st.stop()

    # ----------------------------------------
    # Main UI (no API key shown anywhere)
    # ----------------------------------------
    st.markdown("### 1️⃣ Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or TXT)",
        type=["pdf", "txt"],
        help="Upload a PDF or plain text version of your resume.",
    )

    st.markdown("### 2️⃣ Enter Job Details")
    col1, col2 = st.columns(2)
    with col1:
        job_role = st.text_input(
            "Target Job Role", placeholder="e.g., Frontend Developer"
        )
    with col2:
        company_name = st.text_input("Company Name", placeholder="e.g., Google")

    extra_details = st.text_area(
        "Additional Details / Job Description (optional)",
        placeholder="Paste the job description or add any notes you want the cover letter to consider.",
        height=150,
    )

    generate_button = st.button("✨ Generate Cover Letter")

    if generate_button:
        if uploaded_file is None:
            st.error("Please upload your resume first.")
            return

        if not job_role or not company_name:
            st.error("Please fill in both the Job Role and Company Name.")
            return

        with st.spinner(
            "Reading your resume and generating a tailored cover letter..."
        ):
            resume_text = extract_text_from_file(uploaded_file)
            if not resume_text:
                st.error(
                    "Could not extract text from the uploaded resume. "
                    "Please check the file and try again."
                )
                return

            try:
                llm = get_llm(openai_api_key)
                cover_letter = generate_cover_letter(
                    llm=llm,
                    resume_text=resume_text,
                    job_role=job_role,
                    company_name=company_name,
                    extra_details=extra_details,
                )
            except Exception as e:
                st.error(f"Error while generating cover letter: {e}")
                return

        st.markdown("### 3️⃣ Generated Cover Letter")
        st.success(
            "Your cover letter has been generated below. You can copy and refine it as needed."
        )
        st.write(cover_letter)

        # Optional: download as text
        st.download_button(
            label="📥 Download Cover Letter as .txt",
            data=cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
