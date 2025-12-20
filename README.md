# YouTube Video Summarizer

A Streamlit-based web application that summarizes YouTube videos by extracting their transcripts and using AI to generate concise, detailed, or bullet-point summaries.

## Features

- **Easy URL Input**: Paste any YouTube video URL (supports standard, shorts, and youtu.be formats)
- **Multiple Summary Styles**: Choose from concise paragraph, detailed summary, or bullet-point format
- **AI-Powered Summarization**: Uses OpenAI's GPT models via LangChain for intelligent summarization
- **Transcript Display**: Optional view of the raw transcript for reference
- **Error Handling**: Graceful handling of videos without transcripts or API failures

## Tech Stack

- **Frontend/UI**: Streamlit
- **Backend Logic**: Python
- **AI/ML**:
  - LangChain (for text processing and summarization chains)
  - OpenAI GPT models (via langchain-openai)
- **Transcript Extraction**: youtube-transcript-api
- **Environment Management**: python-dotenv
- **Text Processing**: tiktoken (for tokenization)

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Streamlit App   │───▶│  Transcript API │
│ (YouTube URL)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Text Splitter   │───▶│ Summarization    │───▶│   AI Summary    │
│ (LangChain)     │    │ Chain (Map-Reduce)│    │   Output       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Workflow Steps

1. **URL Processing**: Extract video ID from various YouTube URL formats
2. **Transcript Fetching**: Retrieve video transcript using YouTube Transcript API
3. **Text Chunking**: Split long transcripts into manageable chunks using RecursiveCharacterTextSplitter
4. **AI Summarization**: Apply map-reduce chain with custom prompts based on selected style
5. **Result Display**: Show formatted summary to user with optional transcript view

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for summarization)
- Internet connection (for YouTube API and OpenAI API)

## Installation

1. **Clone or download the project files** to your local machine.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy the `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Optionally configure the model:
     ```
     OPENAI_MODEL=gpt-4o-mini  # or your preferred model
     ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the app**:
   - Open your browser and go to `http://localhost:8501`

3. **Summarize a video**:
   - Paste a YouTube video URL in the input field
   - Select your preferred summary style (concise, detailed, or bullet-points)
   - Click "Summarize Video"
   - View the AI-generated summary

## Configuration

### Summary Styles

- **Concise**: A short paragraph focusing on main ideas
- **Detailed**: Comprehensive summary with key points and examples
- **Bullet-points**: Structured list of key ideas

### Model Configuration

You can modify the model used in `app.py`:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Change this to your preferred model
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)
```

## Testing

The `test.py` file contains a simple test to verify transcript fetching:

```bash
python test.py
```

This will fetch and display information about a sample video's transcript.

## Limitations

- Only works with videos that have available transcripts
- Requires OpenAI API access and credits
- Summarization quality depends on transcript accuracy
- Large videos may take longer to process due to API rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source. Please check the license file for details.

## Support

If you encounter issues:
- Ensure your OpenAI API key is valid and has sufficient credits
- Check that the YouTube video has transcripts enabled
- Verify all dependencies are installed correctly
- Check the console for error messages
