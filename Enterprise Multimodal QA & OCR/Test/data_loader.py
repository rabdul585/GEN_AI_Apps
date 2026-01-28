"""
Data Loader Module
Handles loading, processing, and managing OCR-extracted documents
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

from config import (
    PROCESSED_DIR,
    MIN_TEXT_LENGTH,
    METADATA_FIELDS
)


class Document:
    """Represents a processed document with text and metadata"""
    
    def __init__(
        self,
        text: str,
        metadata: Dict = None,
        doc_id: str = None
    ):
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
        self.created_at = datetime.now().isoformat()
        
        # Add essential metadata
        if "doc_id" not in self.metadata:
            self.metadata["doc_id"] = self.doc_id
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = self.created_at
    
    def _generate_id(self) -> str:
        """Generate unique document ID based on content hash"""
        content = f"{self.text}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create document from dictionary"""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id")
        )
    
    def __repr__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Document(id={self.doc_id}, text='{text_preview}')"


class DataLoader:
    """Manages loading and processing of OCR documents"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.processed_dir = PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def create_document(
        self,
        text: str,
        source_file: str = None,
        page_number: int = None,
        ocr_model: str = None,
        document_type: str = None,
        **kwargs
    ) -> Optional[Document]:
        """
        Create a document from OCR text with metadata
        
        Args:
            text: Extracted text content
            source_file: Original filename
            page_number: Page number (if multi-page)
            ocr_model: Model used for OCR
            document_type: Type of document (invoice, report, etc.)
            **kwargs: Additional metadata
        
        Returns:
            Document object or None if text is too short
        """
        # Validate text length
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return None
        
        # Build metadata
        metadata = {
            "source_file": source_file or "unknown",
            "page_number": page_number,
            "ocr_model": ocr_model,
            "document_type": document_type,
            "text_length": len(text),
            "word_count": len(text.split()),
            **kwargs
        }
        
        # Create document
        doc = Document(text=text, metadata=metadata)
        self.documents.append(doc)
        
        return doc
    
    def load_documents_from_ocr_results(
        self,
        ocr_results: List[Dict]
    ) -> List[Document]:
        """
        Load multiple documents from OCR processing results
        
        Args:
            ocr_results: List of dicts with 'text', 'filename', and other metadata
        
        Returns:
            List of created Document objects
        """
        created_docs = []
        
        for result in ocr_results:
            doc = self.create_document(
                text=result.get("text", ""),
                source_file=result.get("filename"),
                page_number=result.get("page_number"),
                ocr_model=result.get("model"),
                document_type=result.get("document_type"),
                processing_mode=result.get("mode"),
                timestamp=result.get("timestamp")
            )
            
            if doc:
                created_docs.append(doc)
        
        return created_docs
    
    def save_document(self, document: Document, filepath: Path = None) -> Path:
        """
        Save document to disk as JSON
        
        Args:
            document: Document to save
            filepath: Optional custom filepath
        
        Returns:
            Path where document was saved
        """
        if filepath is None:
            filename = f"{document.doc_id}.json"
            filepath = self.processed_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_document(self, filepath: Path) -> Document:
        """Load document from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Document.from_dict(data)
    
    def save_all_documents(self) -> List[Path]:
        """Save all loaded documents to disk"""
        saved_paths = []
        for doc in self.documents:
            path = self.save_document(doc)
            saved_paths.append(path)
        return saved_paths
    
    def load_all_documents(self) -> List[Document]:
        """Load all documents from processed directory"""
        self.documents = []
        
        for filepath in self.processed_dir.glob("*.json"):
            try:
                doc = self.load_document(filepath)
                self.documents.append(doc)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return self.documents
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
    
    def get_documents_by_source(self, source_file: str) -> List[Document]:
        """Get all documents from a specific source file"""
        return [
            doc for doc in self.documents
            if doc.metadata.get("source_file") == source_file
        ]
    
    def filter_documents(self, **filters) -> List[Document]:
        """
        Filter documents by metadata criteria
        
        Example:
            loader.filter_documents(document_type="invoice", ocr_model="grok-4.1-fast")
        """
        filtered = []
        
        for doc in self.documents:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents"""
        if not self.documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_length": 0
            }
        
        total_chars = sum(len(doc.text) for doc in self.documents)
        total_words = sum(doc.metadata.get("word_count", 0) for doc in self.documents)
        
        # Group by source
        sources = {}
        for doc in self.documents:
            source = doc.metadata.get("source_file", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_length": total_chars // len(self.documents),
            "unique_sources": len(sources),
            "sources": sources
        }
    
    def clear_documents(self):
        """Clear all loaded documents from memory"""
        self.documents = []
    
    def delete_all_processed_files(self):
        """Delete all processed JSON files from disk"""
        for filepath in self.processed_dir.glob("*.json"):
            filepath.unlink()
        self.documents = []
    
    def __len__(self):
        return len(self.documents)
    
    def __repr__(self):
        return f"DataLoader(documents={len(self.documents)})"


def merge_documents(documents: List[Document], separator: str = "\n\n") -> Document:
    """
    Merge multiple documents into a single document
    
    Args:
        documents: List of documents to merge
        separator: Text to place between merged documents
    
    Returns:
        Single merged Document
    """
    if not documents:
        raise ValueError("No documents to merge")
    
    # Merge text
    merged_text = separator.join(doc.text for doc in documents)
    
    # Merge metadata
    merged_metadata = {
        "merged_from": [doc.doc_id for doc in documents],
        "source_files": [doc.metadata.get("source_file") for doc in documents],
        "is_merged": True,
        "num_documents": len(documents)
    }
    
    return Document(text=merged_text, metadata=merged_metadata)


def split_document(
    document: Document,
    max_length: int = 5000
) -> List[Document]:
    """
    Split a large document into smaller documents
    
    Args:
        document: Document to split
        max_length: Maximum length per split document
    
    Returns:
        List of split Documents
    """
    if len(document.text) <= max_length:
        return [document]
    
    # Split text
    text = document.text
    splits = []
    
    for i in range(0, len(text), max_length):
        chunk_text = text[i:i + max_length]
        
        # Create metadata for split
        split_metadata = document.metadata.copy()
        split_metadata.update({
            "split_from": document.doc_id,
            "split_index": len(splits),
            "is_split": True
        })
        
        splits.append(Document(text=chunk_text, metadata=split_metadata))
    
    return splits


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = DataLoader()
    
    # Create sample documents
    doc1 = loader.create_document(
        text="This is a sample invoice document with important financial information.",
        source_file="invoice_001.png",
        document_type="invoice",
        ocr_model="grok-4.1-fast"
    )
    
    doc2 = loader.create_document(
        text="This is a report about Q4 sales performance and future projections.",
        source_file="report_q4.pdf",
        document_type="report",
        ocr_model="gemini-2.0-flash"
    )
    
    # Save documents
    loader.save_all_documents()
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"Statistics: {stats}")
    
    # Filter documents
    invoices = loader.filter_documents(document_type="invoice")
    print(f"Found {len(invoices)} invoices")
