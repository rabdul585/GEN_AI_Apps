# 🤖 Abdul's RAG Chatbot

A smart AI assistant that lets you chat with your PDF documents. Upload any PDF and ask questions in natural language - get accurate answers with source references!

## What is this project?

This is an AI-powered chatbot that can read and understand PDF documents. Instead of manually searching through pages, you can simply ask questions like "What is the main topic?" or "Summarize the key points" and get instant, accurate answers. The bot shows you exactly where in the document it found the information, so you can trust the answers.

## Try it out (Demo)

Want to see how it works? Here's what you do:

1. Start the app on your computer (see "How to Run" below)
2. Upload a PDF document using the sidebar
3. Click "Process PDF" to analyze the document
4. Go to the Chat page
5. Ask questions about your document content
6. See answers with source citations from the PDF

## Main Features

- **PDF Upload**: Works with any text-based PDF document
- **Smart Q&A**: Ask natural language questions about your document
- **Source Citations**: Every answer shows exactly where it came from in the PDF
- **Chat History**: Remembers your conversation during the session
- **Multi-page Support**: Handles long documents with many pages
- **Fast Processing**: Quick setup - just upload and start chatting
- **Beautiful UI**: Clean, modern interface with helpful navigation

## What We Used to Build It

- **User Interface**: Streamlit (makes web apps easy)
- **AI Brain**:
  - LangChain (helps coordinate the AI workflow)
  - OpenAI's GPT (the smart AI that answers questions)
  - HuggingFace (for understanding text meaning)
- **Document Processing**: PyPDF (reads PDF files)
- **Smart Search**: FAISS (finds relevant information quickly)
- **Settings**: python-dotenv (keeps your secret keys safe)
- **Python**: Version 3.8 or newer

## How It Works (Simple Flow)

```
Upload PDF ──▶ Break into Chunks ──▶ Create Smart Search ──▶ Answer Questions
     │                    │                    │
     ▼                    ▼                    ▼
Read Text ──▶ Split Smartly ──▶ Store in Memory ──▶ Show Sources!
```

**Step by step:**
1. You upload a PDF and the app reads all the text
2. The document is split into smaller chunks for better understanding
3. Each chunk gets converted to numbers (embeddings) so the AI can find relevant parts
4. When you ask a question, the AI finds the most relevant chunks
5. The AI uses those chunks to create an accurate answer
6. You see the answer plus the exact source text from the PDF

## Getting Started

### What You Need First

- Python 3.8 or higher (download from python.org)
- An OpenAI account with an API key (get one at platform.openai.com)

### Install the App

1. **Get the files**: Download or copy this project to your computer.

2. **Open a terminal/command prompt** and go to the project folder:
   ```bash
   cd path/to/Unified_RAG
   ```

3. **Install what you need**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your secret key**:
   - Make a file called `.env` in the same folder
   - Add this line to it:
     ```
     OPENAI_API_KEY=your_secret_key_here
     ```
   - Replace `your_secret_key_here` with your real OpenAI key

## How to Run the App

1. **Start it up**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to: `http://localhost:8501`

3. **Use the app**:
   - Use the sidebar to upload a PDF
   - Click "Process PDF" to analyze it
   - Go to the "Chat" page
   - Ask questions about your document
   - Check the sources to verify answers

## Project Files

```
Unified_RAG/
├── app.py              # The main chatbot application
├── requirements.txt    # List of things to install
└── .env               # Your secret settings (don't share!)
```

## Example: How to Use It

**Imagine you have a research paper PDF:**

1. **Upload PDF**: You upload "climate_change_study.pdf" (a 50-page research document)

2. **Process It**: Click "Process PDF" - the app splits it into 200 text chunks

3. **Ask Questions**:
   - "What are the main findings?"
   - "What methodology did they use?"
   - "What are the conclusions?"

**What you get:**
```
🤖 Assistant: The main findings show that global temperatures have risen 1.1°C since pre-industrial times, with the last decade being the warmest on record...

📚 Sources:
📄 Source 1 (Page 15): "Global surface temperatures have increased by approximately 1.1°C..."
📄 Source 2 (Page 32): "The 2010s were the warmest decade in the instrumental record..."
```

## Things to Know (Limitations)

- You need an OpenAI account and some credits to use the AI
- Only works with text-based PDFs (not scanned images)
- Very large PDFs might take longer to process
- Chat history only lasts during your current session
- Answers are based only on the uploaded document content

## What's Next? (Future Plans)

- Add support for Word documents and other file types
- Let you save and load chat conversations
- Add multiple document support (chat with several PDFs at once)
- Include voice input for questions
- Add export options for chat summaries
- Support for different languages

## Who Made This?

**Author:** Abdul AI

## Get in Touch

Have questions or ideas? Contact me:
- Email: your-email@example.com
- GitHub: your-github-username
- LinkedIn: your-linkedin-profile

---

*This tool helps you understand documents better, but always read the original source for complete information!*