# 📄 AI Cover Letter Generator (LangChain + Streamlit)

This is a simple GEN-AI web application built with **LangChain** and **Streamlit**.

The app allows users to:

- Upload a **resume** (PDF or TXT)
- Enter:
  - **Target Job Role**
  - **Company Name**
  - **Additional details / Job description** (optional)
- Generate a **tailored cover letter** using a Large Language Model (LLM)

---

## 🚀 Features

- Uses **LangChain** with an OpenAI-compatible chat model
- Reads content from the uploaded resume (PDF or TXT)
- Generates a **professional, concise cover letter** (around 200–400 words)
- Aligns the cover letter with:
  - The candidate’s **experience** (from the resume)
  - The **job role**
  - The **company name**
- Avoids fabricating experience / skills not present in the resume
- Built with a clean and simple **Streamlit UI**
- Option to **download** the generated cover letter as a `.txt` file

---

## 🧩 Tech Stack

- **Python**
- **Streamlit** – Web UI
- **LangChain** – Orchestration framework
- **langchain-openai** – OpenAI-compatible LLM wrapper
- **OpenAI** – LLM provider (via API)
- **PyPDF2** – For reading text from PDF resumes

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
