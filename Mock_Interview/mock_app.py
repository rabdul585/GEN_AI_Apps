import os
from io import StringIO

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import PyPDF2


# -----------------------------
# Utility: Extract text from uploaded file
# -----------------------------
def extract_text_from_file(uploaded_file) -> str:
    """
    Extracts text from an uploaded file.
    Supports PDF and plain text. Can be extended for DOCX if needed.
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

    st.warning("Unsupported file type. Please upload a PDF or TXT file.")
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
        temperature=0.3,
        openai_api_key=openai_api_key,
    )


def get_mock_interview_prompt() -> ChatPromptTemplate:
    """
    Returns a prompt template that generates most expected
    interview questions and strong sample answers.
    """
    template = """
You are an expert interview coach and hiring manager.

You will be given:
- The candidate's CV / resume
- The job description / role details
- The desired number of questions

Your job:
- Generate realistic, high-quality interview questions that are very likely to be asked for this role.
- Provide strong sample answers for each question based on the candidate's CV and the job description.

Guidelines:
- Ask Pure technical, and role-specific questions as appropriate.
- Use the candidate's real experience and skills from the CV; do NOT invent fake projects or companies.
- Answers should be concise but well-structured (3–6 sentences each).
- Use first-person voice in answers ("I ...").
- Return the output as a numbered list, where each item looks like:

Q: <question text>
A: <answer text>

If some information is missing from the CV, you may give a generic but realistic answer,
without fabricating specific companies, dates, or technologies.

Candidate CV:
\"\"\" 
{cv_text}
\"\"\"

Job Description / Role Details:
\"\"\" 
{jd_text}
\"\"\"

Number of questions to generate: {num_questions}
"""
    return ChatPromptTemplate.from_template(template)


def generate_mock_interview(
    llm: ChatOpenAI,
    cv_text: str,
    jd_text: str,
    num_questions: int,
) -> str:
    """
    Uses LangChain (LLM + prompt) to generate mock interview Q&A.
    """
    prompt = get_mock_interview_prompt()
    messages = prompt.format_messages(
        cv_text=cv_text,
        jd_text=jd_text,
        num_questions=num_questions,
    )
    response = llm.invoke(messages)
    return response.content.strip()


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Mock Interview Generator",
        page_icon="🎤",
        layout="centered",
    )

    st.title("🎤 AI Mock Interview – Questions & Model Answers")
    st.write(
        "Upload your CV and job description. "
        "This app will generate likely interview questions **with strong sample answers** "
        "based on your profile and the role."
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
    # Inputs: CV + Job Description
    # ----------------------------------------
    st.markdown("### 1️⃣ Upload Your CV / Resume")
    cv_file = st.file_uploader(
        "Upload your CV (PDF or TXT)",
        type=["pdf", "txt"],
        help="Upload a PDF or plain text version of your CV / resume.",
        key="cv_uploader",
    )

    st.markdown("### 2️⃣ Provide Job Description")
    jd_col1, jd_col2 = st.columns(2)
    with jd_col1:
        jd_file = st.file_uploader(
            "Upload Job Description (optional, PDF or TXT)",
            type=["pdf", "txt"],
            help="You can upload the JD as a file, or paste it in the text area below.",
            key="jd_uploader",
        )
    with jd_col2:
        num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="How many Q&A pairs to generate.",
        )

    jd_text_manual = st.text_area(
        "Or paste the Job Description / Role details here",
        placeholder="Paste the job description or key responsibilities/requirements...",
        height=160,
    )

    st.markdown("---")
    generate_button = st.button("✨ Generate Mock Interview Q&A")

    if generate_button:
        # Basic validation
        if cv_file is None:
            st.error("Please upload your CV / resume first.")
            return

        # Extract CV text
        with st.spinner("Reading your CV..."):
            cv_text = extract_text_from_file(cv_file)
        if not cv_text:
            st.error(
                "Could not extract text from the uploaded CV. "
                "Please check the file and try again."
            )
            return

        # Extract JD text from file (if any)
        jd_text_from_file = ""
        if jd_file is not None:
            with st.spinner("Reading the job description file..."):
                jd_text_from_file = extract_text_from_file(jd_file)

        # Combine file JD + manual JD
        jd_text_combined = "\n\n".join(
            [t for t in [jd_text_from_file, jd_text_manual] if t.strip()]
        )

        if not jd_text_combined:
            st.warning(
                "No job description provided. The app will generate more generic questions. "
                "For better results, upload or paste the JD."
            )
            jd_text_combined = "No specific job description provided."

        # Generate Q&A
        with st.spinner(
            f"Generating {num_questions} likely interview questions and answers..."
        ):
            try:
                llm = get_llm(openai_api_key)
                qa_output = generate_mock_interview(
                    llm=llm,
                    cv_text=cv_text,
                    jd_text=jd_text_combined,
                    num_questions=num_questions,
                )
            except Exception as e:
                st.error(f"Error while generating mock interview Q&A: {e}")
                return

        st.markdown("### 3️⃣ Mock Interview – Questions & Sample Answers")
        st.success(
            "Review the questions and answers below. Use them to practice and refine your own responses."
        )
        st.markdown(qa_output)

        # Download option
        st.download_button(
            label="📥 Download Q&A as .txt",
            data=qa_output,
            file_name="mock_interview_qa.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
