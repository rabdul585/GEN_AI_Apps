# AI Mock Interview Generator

A Streamlit-based web application that generates realistic interview questions and strong sample answers based on your CV/resume and job description using AI.

## Project Overview and Purpose

The AI Mock Interview Generator helps job seekers prepare for interviews by creating personalized mock interviews. By analyzing your CV and the target job description, the app generates likely interview questions along with well-structured sample answers tailored to your experience and the role requirements. This tool bridges the gap between generic interview preparation and role-specific practice, helping candidates feel more confident and prepared.

## Demo

To see the app in action:
1. Run the application locally (see Installation section below)
2. Upload your CV (PDF or TXT format)
3. Provide the job description (upload file or paste text)
4. Select the number of questions to generate
5. Click "Generate Mock Interview Q&A" to see AI-generated questions and answers

## Key Features

- **CV Upload Support**: Accepts PDF and plain text CV/resume files
- **Flexible Job Description Input**: Upload JD files or paste text directly
- **Customizable Question Count**: Generate 5-20 Q&A pairs
- **AI-Powered Generation**: Uses OpenAI's GPT models for intelligent question and answer creation
- **Download Option**: Export generated Q&A as a text file for offline review
- **Error Handling**: Graceful handling of file processing and API errors
- **Responsive UI**: Clean, centered layout optimized for desktop and mobile

## Tech Stack

- **Frontend/UI**: Streamlit
- **AI/ML**:
  - LangChain (for LLM orchestration and prompt management)
  - OpenAI GPT models (via langchain-openai)
- **File Processing**: PyPDF2 (for PDF text extraction)
- **Environment Management**: python-dotenv
- **Python Version**: 3.8+

## Architecture and Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Uploads  │───▶│  Streamlit App   │───▶│  File Processing│
│   CV & JD       │    │                  │    │  (PDF/TXT)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prompt        │───▶│   LLM Generation │───▶│   Q&A Output    │
│   Engineering   │    │   (OpenAI GPT)   │    │   Display       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Workflow Steps

1. **Input Collection**: User uploads CV and provides job description
2. **Text Extraction**: App extracts text content from uploaded files
3. **Prompt Construction**: Creates detailed prompts with CV and JD context
4. **AI Generation**: Uses LangChain to invoke OpenAI API with custom prompts
5. **Output Formatting**: Displays numbered Q&A pairs with download option

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (sign up at [OpenAI Platform](https://platform.openai.com/))

### Installation Steps

1. **Clone or download the project files** to your local machine.

2. **Navigate to the project directory**:
   ```bash
   cd path/to/Mock_Interview
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit langchain-openai python-dotenv PyPDF2
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Optionally configure the model (default is gpt-4o-mini):
     ```
     OPENAI_MODEL=gpt-4o-mini
     ```

## How to Run the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run mock_app.py
   ```

2. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`

3. **Use the app**:
   - Upload your CV (PDF or TXT)
   - Provide job description (upload file or paste text)
   - Select number of questions (5-20)
   - Click "Generate Mock Interview Q&A"
   - Review the generated questions and answers
   - Download the Q&A as a text file if desired

## Folder Structure

```
Mock_Interview/
├── mock_app.py          # Main Streamlit application
├── .env                 # Environment variables (API keys)
└── README.md           # This file
```

## Example Usage

### Scenario: Software Developer Position

1. **Upload CV**: A PDF resume highlighting Python development experience, React projects, and AWS certifications
2. **Job Description**: "Senior Python Developer - 5+ years experience, Django, REST APIs, PostgreSQL"
3. **Generate 10 Questions**

**Sample Output:**
```
Q: Can you describe your experience with Django and how you've used it in production applications?
A: I have 4 years of experience working with Django, having built and maintained several web applications. In my previous role at TechCorp, I developed a RESTful API using Django REST framework that handled over 10,000 daily requests...

Q: How do you approach database optimization in Python applications?
A: I focus on query optimization, proper indexing, and using database connection pooling. At my last position, I improved database query performance by 60% by implementing selective indexing and query refactoring...
```

## Limitations

- Requires active OpenAI API key and sufficient API credits
- Currently supports only PDF and TXT file formats for CV/JD
- Quality of generated questions depends on CV detail and JD specificity
- API rate limits may affect generation speed for large requests
- Generated answers are samples only - should be personalized for actual interviews

## Future Improvements

- Support for additional file formats (DOCX, DOC)
- Integration with LinkedIn profiles for CV data
- Voice-based mock interviews with speech recognition
- Industry-specific question templates
- Performance tracking and improvement suggestions
- Multi-language support
- Integration with job board APIs for automatic JD fetching

## Author

Abdul AI

## Contact

For questions, suggestions, or contributions, please reach out via:
- Email: [your-email@example.com]
- GitHub: [your-github-username]
- LinkedIn: [your-linkedin-profile]

---

*This project is for educational and preparation purposes. Always prepare genuine answers for actual interviews.*