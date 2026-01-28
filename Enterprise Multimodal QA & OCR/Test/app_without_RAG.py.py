import streamlit as st
from dotenv import load_dotenv
import os
import requests
from pathlib import Path
import base64
from PIL import Image
import io
import json
from datetime import datetime
import time

st.set_page_config(
    page_title="Enterprise Multimodal QA & OCR - OpenRouter",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()

# Initialize session state
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

st.sidebar.header("🔑 API Key Status")

env_file_path = Path(".env")
if env_file_path.exists():
    st.sidebar.success(".env file found")
else:
    st.sidebar.warning(".env file not found in current directory")
    st.sidebar.write(f"Looking in: {os.getcwd()}")

api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    st.sidebar.success("OPENROUTER_API_KEY loaded")
    if len(api_key) > 12:
        masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    else:
        masked_key = f"{api_key[:4]}...{api_key[-2:]}"
    st.sidebar.write(f"Key Preview: {masked_key}")
    st.sidebar.write(f"Key Length: {len(api_key)} characters")
else:
    st.sidebar.error("OPENROUTER_API_KEY NOT loaded")
    st.sidebar.write("Troubleshooting:")
    st.sidebar.write("1. Check if .env file exists")
    st.sidebar.write("2. Verify the key name is: OPENROUTER_API_KEY=sk-or-...")
    st.sidebar.write("3. No quotes or spaces around the key")
    st.stop()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# UPDATED MODEL LIST WITH GROK AND BETTER OPTIONS
AVAILABLE_MODELS = {
    "🚀 Grok 4.1 Fast (FREE) - RECOMMENDED": {
        "id": "x-ai/grok-4.1-fast",
        "context": "128K tokens",
        "description": "Fastest, large context, excellent for documents",
        "ocr_quality": "⭐⭐⭐⭐⭐ Excellent"
    },
    "🌟 Google Gemini 2.0 Flash Exp (FREE)": {
        "id": "google/gemini-2.0-flash-exp:free",
        "context": "1.05M tokens",
        "description": "Best OCR accuracy, fastest, largest context",
        "ocr_quality": "⭐⭐⭐⭐⭐ Excellent"
    },
    "🦙 Meta Llama 4 Maverick (FREE)": {
        "id": "meta-llama/llama-4-maverick:free",
        "context": "256K tokens",
        "description": "400B MoE, powerful multimodal reasoning",
        "ocr_quality": "⭐⭐⭐⭐ Very Good"
    },
    "💎 Google Gemma 3 27B (FREE)": {
        "id": "google/gemma-3-27b-it:free",
        "context": "131K tokens",
        "description": "140+ languages OCR, structured outputs",
        "ocr_quality": "⭐⭐⭐⭐ Very Good"
    },
    "🔷 Qwen 2.5 VL 3B (FREE)": {
        "id": "qwen/qwen2.5-vl-3b-instruct:free",
        "context": "32K tokens",
        "description": "Compact, efficient for simple OCR",
        "ocr_quality": "⭐⭐⭐ Good"
    },
    "🔥 DeepSeek R1 (FREE)": {
        "id": "deepseek/deepseek-r1:free",
        "context": "64K tokens",
        "description": "Good reasoning, alternative option",
        "ocr_quality": "⭐⭐⭐⭐ Very Good"
    }
}

st.sidebar.header("🤖 Model Selection")
st.sidebar.info("All models support OCR & Vision!")

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(AVAILABLE_MODELS.keys()),
    index=0,
    help="Select model based on your OCR needs. Grok 4.1 Fast recommended for large documents!"
)

selected_model_info = AVAILABLE_MODELS[selected_model_name]
selected_model = selected_model_info["id"]

st.sidebar.success(f"**Model ID:** {selected_model}")
st.sidebar.write(f"**Context:** {selected_model_info['context']}")
st.sidebar.write(f"**OCR Quality:** {selected_model_info['ocr_quality']}")
st.sidebar.write(f"**Info:** {selected_model_info['description']}")
st.sidebar.write("**Cost:** $0/M tokens 🎉")

# Processing Mode Selection
st.sidebar.header("⚙️ Processing Mode")
processing_mode = st.sidebar.radio(
    "Select mode:",
    [
        "🔍 OCR + Analysis",
        "📝 Pure OCR (Text Only)",
        "📊 Document Analysis",
        "🔬 Advanced OCR",
        "💬 General Conversation"
    ],
    index=0,
    help="Choose how to process your documents"
)

# Advanced OCR Options
if processing_mode in ["🔬 Advanced OCR", "📊 Document Analysis"]:
    st.sidebar.subheader("🎛️ Advanced Options")
    
    preserve_formatting = st.sidebar.checkbox(
        "Preserve formatting",
        value=True,
        help="Maintain document structure and layout"
    )
    
    extract_tables = st.sidebar.checkbox(
        "Extract tables",
        value=False,
        help="Identify and extract table data"
    )
    
    detect_language = st.sidebar.checkbox(
        "Detect language",
        value=False,
        help="Identify document language"
    )
    
    extract_metadata = st.sidebar.checkbox(
        "Extract metadata",
        value=False,
        help="Get document type, layout info"
    )

# Rate Limiting Settings
st.sidebar.header("⚙️ Rate Limit Settings")
retry_attempts = st.sidebar.slider(
    "Retry attempts on rate limit:",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of retries if rate limited"
)

retry_delay = st.sidebar.slider(
    "Delay between retries (seconds):",
    min_value=1,
    max_value=10,
    value=3,
    help="Wait time before retrying"
)

# Export Options
st.sidebar.header("💾 Export Options")
export_format = st.sidebar.selectbox(
    "Export format:",
    ["Text (.txt)", "JSON (.json)", "Markdown (.md)"],
    help="Choose format for downloaded text"
)

def image_to_base64(image_file):
    """Convert uploaded image to base64 string with OCR optimization"""
    try:
        image = Image.open(image_file)
        
        # Higher resolution for better OCR accuracy
        max_size = (2048, 2048)  # Increased for OCR
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed (for better OCR)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=95)  # High quality for OCR
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def get_ocr_prompt(mode, custom_question=None):
    """Generate optimized prompts based on processing mode"""
    
    if mode == "📝 Pure OCR (Text Only)":
        return """Extract ALL text from this image with maximum accuracy.

INSTRUCTIONS:
- Extract every word, number, symbol, and character
- Preserve exact line breaks and paragraph structure
- Maintain original text order (top to bottom, left to right)
- Include headers, footers, page numbers, captions
- Do NOT add explanations or commentary
- Do NOT translate or interpret
- If no text found, respond: "No text detected in image"

OUTPUT: Only the extracted text, exactly as it appears."""

    elif mode == "🔍 OCR + Analysis":
        base_prompt = """Perform OCR extraction and analysis.

TASK 1 - TEXT EXTRACTION:
Extract all text from the image accurately.

TASK 2 - ANALYSIS:
"""
        if custom_question:
            base_prompt += f"{custom_question}\n\n"
        else:
            base_prompt += """Answer these questions:
- What type of document is this?
- What is the main topic/purpose?
- Who is the intended audience?
- Are there any key insights or important information?

"""
        base_prompt += """FORMAT YOUR RESPONSE AS:

=== EXTRACTED TEXT ===
[All text from image]

=== ANALYSIS ===
[Your analysis here]"""
        return base_prompt

    elif mode == "📊 Document Analysis":
        prompt = """Perform comprehensive document analysis with OCR.

ANALYZE:
1. **Document Type**: (e.g., invoice, receipt, form, report, letter)
2. **Document Structure**: Headers, sections, layout
3. **Key Information**: Important dates, numbers, names, amounts
4. **Text Content**: Full text extraction
"""
        if st.session_state.get('extract_tables', False):
            prompt += "5. **Tables**: Extract any tables with rows and columns\n"
        if st.session_state.get('detect_language', False):
            prompt += "6. **Language**: Identify the language(s) used\n"
        if st.session_state.get('extract_metadata', False):
            prompt += "7. **Metadata**: Page layout, text density, quality assessment\n"
        
        prompt += "\nProvide organized, detailed analysis."
        return prompt

    elif mode == "🔬 Advanced OCR":
        prompt = """Perform ADVANCED OCR with detailed extraction.

EXTRACT WITH PRECISION:
1. **All Text Content**: Every character, exactly as shown
2. **Document Structure**: 
   - Headings and subheadings
   - Paragraphs and sections
   - Lists (numbered, bulleted)
   - Footnotes and annotations
"""
        if st.session_state.get('preserve_formatting', True):
            prompt += """3. **Formatting Preservation**:
   - Bold, italic, underlined text (indicate with markup)
   - Font sizes (indicate relative sizes)
   - Text alignment
   - Spacing and indentation
"""
        if st.session_state.get('extract_tables', False):
            prompt += """4. **Table Extraction**:
   - Identify all tables
   - Extract as structured data (rows × columns)
   - Include headers and data cells
"""
        if st.session_state.get('detect_language', False):
            prompt += """5. **Language Detection**:
   - Primary language
   - Any secondary languages
   - Mixed language content
"""
        
        prompt += "\n\nProvide highly accurate, structured output."
        return prompt

    else:  # General Conversation
        return custom_question if custom_question else "Describe this image in detail."

def get_openrouter_response(question, model, images=None, max_retries=3, retry_delay_sec=3):
    """
    Send request to OpenRouter API with retry logic for rate limiting
    
    Args:
        question: Text prompt
        model: Model ID
        images: List of base64 encoded images
        max_retries: Number of retry attempts
        retry_delay_sec: Seconds to wait between retries
    """
    
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Enterprise OCR QA App"
            }
            
            # Build message content
            if images and len(images) > 0:
                content = []
                
                # Add text first (recommended by OpenRouter)
                if question:
                    content.append({
                        "type": "text",
                        "text": question
                    })
                
                # Add images
                for img_base64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    })
            else:
                content = question
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=120  # Longer timeout for OCR processing
            )
            
            # Success case
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            
            # Rate limit case - retry with exponential backoff
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay_sec * (2 ** attempt)  # Exponential backoff
                    st.warning(f"⏳ Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                    raise Exception(f"Rate Limit Error (429): {error_msg}\n\n💡 Suggestions:\n" +
                                  "1. Try a different model (Grok 4.1 Fast recommended)\n" +
                                  "2. Wait a few minutes and try again\n" +
                                  "3. Check OpenRouter dashboard for rate limits")
            
            # Other errors
            else:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API Error ({response.status_code}): {error_msg}")
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"⏳ Request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay_sec)
                continue
            else:
                raise Exception("Request timed out after multiple attempts. Try a smaller image or different model.")
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"⏳ Network error. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay_sec)
                continue
            else:
                raise Exception(f"Network Error: {str(e)}")
        
        except Exception as e:
            # Don't retry on other exceptions
            raise Exception(f"Error: {str(e)}")
    
    raise Exception("Failed after maximum retry attempts")

def extract_text_from_response(response, mode):
    """Extract just the text portion from OCR response"""
    if mode == "📝 Pure OCR (Text Only)":
        return response
    
    # Try to extract text section from formatted responses
    if "=== EXTRACTED TEXT ===" in response:
        parts = response.split("=== EXTRACTED TEXT ===")
        if len(parts) > 1:
            text_section = parts[1].split("=== ANALYSIS ===")[0] if "=== ANALYSIS ===" in parts[1] else parts[1]
            return text_section.strip()
    
    return response

def create_download_content(text, format_type):
    """Create downloadable content in specified format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == "Text (.txt)":
        return text, f"ocr_extract_{timestamp}.txt", "text/plain"
    
    elif format_type == "JSON (.json)":
        json_data = {
            "timestamp": timestamp,
            "model": selected_model,
            "processing_mode": processing_mode,
            "extracted_text": text,
            "character_count": len(text),
            "word_count": len(text.split())
        }
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        return json_str, f"ocr_extract_{timestamp}.json", "application/json"
    
    elif format_type == "Markdown (.md)":
        md_content = f"""# OCR Extraction Results

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** {selected_model}  
**Processing Mode:** {processing_mode}  

---

## Extracted Text

{text}

---

**Statistics:**
- Characters: {len(text)}
- Words: {len(text.split())}
- Lines: {len(text.splitlines())}
"""
        return md_content, f"ocr_extract_{timestamp}.md", "text/markdown"

# Connection Test
st.sidebar.header("🔌 Connection Status")
if st.sidebar.button("Test API Connection"):
    with st.sidebar:
        with st.spinner("Testing connection..."):
            try:
                test_response = get_openrouter_response(
                    "Say 'Hello!' in one sentence.", 
                    selected_model,
                    max_retries=retry_attempts,
                    retry_delay_sec=retry_delay
                )
                st.success("✅ API connection successful!")
                st.write(f"Response: {test_response[:100]}...")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

# Main Title
st.title("🔍 Enterprise Multimodal QA & OCR")
st.subheader("Powered by Free OpenRouter Vision Models")
st.write(f"**Current Model:** {selected_model_name} | **Mode:** {processing_mode}")

# Rate Limit Notice
st.info("💡 **Tip:** If you encounter rate limits (Error 429), try Grok 4.1 Fast or wait a few minutes before retrying.")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    
    # Custom question/prompt
    if processing_mode == "💬 General Conversation":
        placeholder = "e.g., What's in this image? Describe what you see."
    elif processing_mode == "📝 Pure OCR (Text Only)":
        placeholder = "Leave empty for automatic OCR extraction"
    else:
        placeholder = "Optional: Ask specific questions about the document"
    
    input_question = st.text_area(
        "Enter your question or leave empty for default OCR:",
        placeholder=placeholder,
        height=120,
        help="Custom prompts work with all modes except Pure OCR"
    )
    
    # Image uploader
    uploaded_files = st.file_uploader(
        "📸 Upload document images:",
        type=["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Upload scanned documents, screenshots, photos of text"
    )
    
    # Display uploaded images
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")
        
        # Image preview
        preview_cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            with preview_cols[idx % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, width='stretch')
                
                # Show image info
                file_size_kb = uploaded_file.size / 1024
                st.caption(f"Size: {file_size_kb:.1f} KB | {image.size[0]}×{image.size[1]}px")
    
    # Process button
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submit_button = st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        if st.session_state.extracted_texts:
            clear_button = st.button(
                "🗑️ Clear History",
                use_container_width=True
            )
            if clear_button:
                st.session_state.extracted_texts = []
                st.session_state.processing_history = []
                st.rerun()

with col2:
    st.subheader("💬 Results")
    
    if submit_button:
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one image to process.")
        else:
            # Process each image
            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"#### 📄 Processing: {uploaded_file.name}")
                
                with st.spinner(f"🔄 Processing image {file_idx + 1}/{len(uploaded_files)}..."):
                    try:
                        # Convert image to base64
                        uploaded_file.seek(0)
                        img_b64 = image_to_base64(uploaded_file)
                        
                        if not img_b64:
                            st.error(f"Failed to process {uploaded_file.name}")
                            continue
                        
                        # Generate appropriate prompt
                        if processing_mode == "📝 Pure OCR (Text Only)":
                            prompt = get_ocr_prompt(processing_mode)
                        else:
                            prompt = get_ocr_prompt(processing_mode, input_question)
                        
                        # Get OCR response with retry logic
                        response = get_openrouter_response(
                            prompt,
                            selected_model,
                            [img_b64],
                            max_retries=retry_attempts,
                            retry_delay_sec=retry_delay
                        )
                        
                        # Display response
                        with st.container():
                            st.markdown("---")
                            st.markdown(response)
                            st.markdown("---")
                        
                        # Extract and store text
                        extracted_text = extract_text_from_response(response, processing_mode)
                        
                        # Save to session state
                        extraction_record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": uploaded_file.name,
                            "mode": processing_mode,
                            "text": extracted_text,
                            "full_response": response,
                            "model": selected_model
                        }
                        st.session_state.extracted_texts.append(extraction_record)
                        st.session_state.processing_history.append(extraction_record)
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        
                        # Statistics
                        char_count = len(extracted_text)
                        word_count = len(extracted_text.split())
                        line_count = len(extracted_text.splitlines())
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Characters", f"{char_count:,}")
                        with col_stat2:
                            st.metric("Words", f"{word_count:,}")
                        with col_stat3:
                            st.metric("Lines", line_count)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}:")
                        st.error(str(e))
                        
                        # Provide helpful suggestions
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            st.info("""
                            **Rate Limit Solutions:**
                            1. 🚀 Try **Grok 4.1 Fast** - often has better rate limits
                            2. ⏰ Wait 2-5 minutes before retrying
                            3. 📊 Check [OpenRouter Dashboard](https://openrouter.ai/activity) for usage
                            4. 🔄 Try a different free model from the dropdown
                            """)

# Export Section
if st.session_state.extracted_texts:
    st.markdown("---")
    st.subheader("💾 Export Extracted Text")
    
    # Show extraction history
    with st.expander(f"📚 Extraction History ({len(st.session_state.extracted_texts)} items)"):
        for idx, record in enumerate(reversed(st.session_state.extracted_texts)):
            st.markdown(f"""
            **{idx + 1}. {record['filename']}**  
            📅 {record['timestamp']} | 🤖 Mode: {record['mode']}  
            📊 {len(record['text'])} chars, {len(record['text'].split())} words
            """)
            
            if st.button(f"View #{idx + 1}", key=f"view_{idx}"):
                st.text_area(
                    f"Extracted text from {record['filename']}:",
                    value=record['text'],
                    height=200,
                    key=f"text_{idx}"
                )
    
    # Combined export
    st.markdown("#### Export All Extracted Text")
    
    # Combine all extracted texts
    combined_text = "\n\n" + "="*80 + "\n\n".join([
        f"FILE: {record['filename']}\nDATE: {record['timestamp']}\nMODE: {record['mode']}\n\n{record['text']}"
        for record in st.session_state.extracted_texts
    ])
    
    # Create download button
    download_content, filename, mime_type = create_download_content(
        combined_text,
        export_format
    )
    
    st.download_button(
        label=f"⬇️ Download All ({export_format})",
        data=download_content,
        file_name=filename,
        mime=mime_type,
        use_container_width=True
    )
    
    # Individual file export
    if len(st.session_state.extracted_texts) > 1:
        st.markdown("#### Export Individual Files")
        for idx, record in enumerate(st.session_state.extracted_texts):
            download_content_single, filename_single, mime_type_single = create_download_content(
                record['text'],
                export_format
            )
            
            st.download_button(
                label=f"⬇️ {record['filename']}",
                data=download_content_single,
                file_name=f"{Path(record['filename']).stem}_{filename_single}",
                mime=mime_type_single,
                key=f"download_{idx}",
                use_container_width=True
            )

# Information Sections
st.markdown("---")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    with st.expander("📖 About OCR Modes"):
        st.markdown("""
        **🔍 OCR + Analysis**
        Extract text and get intelligent analysis
        
        **📝 Pure OCR**
        Extract only text, no analysis
        
        **📊 Document Analysis**
        Comprehensive document understanding
        
        **🔬 Advanced OCR**
        Maximum accuracy with formatting
        
        **💬 General Conversation**
        Natural chat about images
        """)

with col_info2:
    with st.expander("🎯 Best Practices"):
        st.markdown("""
        **For Best OCR Results:**
        - Use high-resolution images (300+ DPI)
        - Ensure good lighting
        - Avoid skewed/rotated images
        - Use Grok 4.1 Fast for large documents
        - Try Gemini 2.0 for best accuracy
        - Crop to relevant areas
        
        **Supported:**
        - Printed documents
        - Handwriting (varies by model)
        - 140+ languages (Gemma 3)
        - Tables and forms
        - Multi-column layouts
        """)

with col_info3:
    with st.expander("⚠️ Troubleshooting Rate Limits"):
        st.markdown("""
        **If you get Error 429:**
        
        1. **Switch Models:**
           - Try Grok 4.1 Fast first
           - Then DeepSeek R1
           - Or Llama 4 Maverick
        
        2. **Wait & Retry:**
           - Free models have hourly limits
           - Wait 2-5 minutes
           - Use retry settings in sidebar
        
        3. **Check Usage:**
           - Visit OpenRouter dashboard
           - View your rate limits
           - Check model availability
        
        4. **Optimize:**
           - Process one image at a time
           - Use smaller images if possible
           - Avoid rapid consecutive requests
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>Enterprise Multimodal QA & OCR Platform v2.0</strong></p>
        <p>Powered by <a href='https://openrouter.ai' target='_blank'>OpenRouter</a> | 
        Free Open Source Models | Built with Streamlit</p>
        <p>🔒 Secure | 🆓 Free | 🚀 Fast | 📊 Enterprise-Ready</p>
        <p>✨ Now with Grok 4.1 Fast & Advanced Rate Limit Handling</p>
    </div>
    """,
    unsafe_allow_html=True
)