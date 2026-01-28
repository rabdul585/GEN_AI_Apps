"""
Retrieval Module
Handles RAG (Retrieval Augmented Generation) queries and response generation
"""

from typing import List, Dict, Optional, Tuple
import requests
import time

from config import (
    OPENROUTER_API_URL,
    OPENROUTER_API_KEY,
    RAG_SYSTEM_PROMPT,
    RAG_QUERY_TEMPLATE,
    DEFAULT_TOP_K,
    MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_RETRIES,
    RETRY_DELAY,
    EXPONENTIAL_BACKOFF
)
from rag.vector_store import ChromaVectorStore


class RAGRetriever:
    """Handles retrieval-augmented generation queries"""
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        model_id: str = "x-ai/grok-4.1-fast",
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: ChromaVectorStore instance
            model_id: LLM model ID for generation
            temperature: Temperature for generation
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.api_key = OPENROUTER_API_KEY
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        print(f"✅ RAG Retriever initialized")
        print(f"   Model: {self.model_id}")
        print(f"   Top-K: {self.top_k}")
        print(f"   Temperature: {self.temperature}")
    
    def retrieve_context(
        self,
        query: str,
        k: int = None,
        filter_metadata: Dict = None,
        score_threshold: float = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter_metadata: Metadata filter for retrieval
            score_threshold: Minimum similarity score
        
        Returns:
            Tuple of (context_texts, metadata_list)
        """
        k = k or self.top_k
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
            score_threshold=score_threshold
        )
        
        if not results:
            return [], []
        
        # Extract texts and metadata
        context_texts = [doc for doc, score, meta in results]
        metadatas = [meta for doc, score, meta in results]
        
        return context_texts, metadatas
    
    def format_context(
        self,
        context_texts: List[str],
        metadatas: List[Dict] = None
    ) -> str:
        """
        Format retrieved context for LLM prompt
        
        Args:
            context_texts: List of context document texts
            metadatas: List of metadata dicts
        
        Returns:
            Formatted context string
        """
        if not context_texts:
            return "No relevant context found."
        
        formatted_parts = []
        
        for i, text in enumerate(context_texts):
            # Add source information if available
            source_info = ""
            if metadatas and i < len(metadatas):
                meta = metadatas[i]
                source_file = meta.get("source_file", "Unknown")
                page = meta.get("page_number")
                
                source_info = f"[Source: {source_file}"
                if page:
                    source_info += f", Page {page}"
                source_info += "]"
            
            formatted_parts.append(f"{source_info}\n{text}" if source_info else text)
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: str = None,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY
    ) -> str:
        """
        Generate response using LLM with retrieved context
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Custom system prompt
            max_retries: Number of retries on failure
            retry_delay: Delay between retries
        
        Returns:
            Generated response text
        """
        system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        
        # Format the prompt
        user_prompt = RAG_QUERY_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG OCR App"
        }
        
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": MAX_TOKENS
        }
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                # Success
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                
                # Rate limit - retry with backoff
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt) if EXPONENTIAL_BACKOFF else retry_delay
                        print(f"⏳ Rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Rate limit exceeded after retries")
                
                # Other errors
                else:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"API Error ({response.status_code}): {error_msg}")
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ Request timed out. Retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception("Request timed out after retries")
            
            except Exception as e:
                if attempt < max_retries - 1 and "rate limit" in str(e).lower():
                    wait_time = retry_delay * (2 ** attempt) if EXPONENTIAL_BACKOFF else retry_delay
                    time.sleep(wait_time)
                    continue
                raise
        
        raise Exception("Failed to generate response")
    
    def query(
        self,
        question: str,
        k: int = None,
        filter_metadata: Dict = None,
        include_context: bool = False,
        include_sources: bool = True
    ) -> Dict:
        """
        Complete RAG query: retrieve context and generate answer
        
        Args:
            question: User question
            k: Number of documents to retrieve
            filter_metadata: Metadata filter
            include_context: Whether to include retrieved context in response
            include_sources: Whether to include source information
        
        Returns:
            Dict with answer, context, sources, and metadata
        """
        # Retrieve context
        context_texts, metadatas = self.retrieve_context(
            query=question,
            k=k,
            filter_metadata=filter_metadata
        )
        
        if not context_texts:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "context": [],
                "sources": [],
                "num_sources": 0
            }
        
        # Format context
        formatted_context = self.format_context(context_texts, metadatas)
        
        # Generate response
        answer = self.generate_response(
            query=question,
            context=formatted_context
        )
        
        # Extract source files
        sources = []
        if include_sources and metadatas:
            seen_sources = set()
            for meta in metadatas:
                source_file = meta.get("source_file", "Unknown")
                if source_file not in seen_sources:
                    sources.append({
                        "file": source_file,
                        "page": meta.get("page_number"),
                        "type": meta.get("document_type")
                    })
                    seen_sources.add(source_file)
        
        result = {
            "answer": answer,
            "sources": sources,
            "num_sources": len(context_texts)
        }
        
        if include_context:
            result["context"] = context_texts
            result["context_metadata"] = metadatas
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        k: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            k: Number of documents to retrieve per question
            filter_metadata: Metadata filter
        
        Returns:
            List of result dicts
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}...")
            try:
                result = self.query(
                    question=question,
                    k=k,
                    filter_metadata=filter_metadata
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "num_sources": 0,
                    "error": str(e)
                })
        
        return results


class ConversationalRAG:
    """RAG with conversation history support"""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        max_history: int = 5
    ):
        """
        Initialize conversational RAG
        
        Args:
            retriever: RAGRetriever instance
            max_history: Maximum conversation history to maintain
        """
        self.retriever = retriever
        self.max_history = max_history
        self.history = []
    
    def query(
        self,
        question: str,
        k: int = None,
        filter_metadata: Dict = None,
        include_history: bool = True
    ) -> Dict:
        """
        Query with conversation context
        
        Args:
            question: User question
            k: Number of documents to retrieve
            filter_metadata: Metadata filter
            include_history: Whether to include conversation history
        
        Returns:
            Dict with answer and metadata
        """
        # Augment question with history if requested
        if include_history and self.history:
            context_question = self._build_contextual_question(question)
        else:
            context_question = question
        
        # Query
        result = self.retriever.query(
            question=context_question,
            k=k,
            filter_metadata=filter_metadata
        )
        
        # Update history
        self.history.append({
            "question": question,
            "answer": result["answer"]
        })
        
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return result
    
    def _build_contextual_question(self, question: str) -> str:
        """Build question with conversation context"""
        if not self.history:
            return question
        
        context_parts = ["Previous conversation:"]
        for i, turn in enumerate(self.history[-3:]):  # Last 3 turns
            context_parts.append(f"Q{i+1}: {turn['question']}")
            context_parts.append(f"A{i+1}: {turn['answer'][:200]}...")  # Truncate long answers
        
        context_parts.append(f"\nCurrent question: {question}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.history.copy()


# Example usage
if __name__ == "__main__":
    from rag.vector_store import ChromaVectorStore
    
    # Initialize
    vector_store = ChromaVectorStore()
    retriever = RAGRetriever(vector_store)
    
    # Sample query
    question = "What is artificial intelligence?"
    
    print(f"\nQuestion: {question}")
    result = retriever.query(question, include_context=True)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {result['num_sources']}")
    for source in result['sources']:
        print(f"  - {source['file']}")
