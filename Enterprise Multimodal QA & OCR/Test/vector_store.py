"""
Vector Store Module
Handles ChromaDB operations for storing and retrieving embeddings
"""

from typing import List, Dict, Optional, Tuple
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ chromadb not installed. Run: pip install chromadb")

from config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD
)
from rag.embeddings import EmbeddingManager


class ChromaVectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_manager: EmbeddingManager = None,
        persist_directory: str = None
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_manager: EmbeddingManager instance for generating embeddings
            persist_directory: Directory to persist the database
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or str(CHROMA_DB_DIR)
        
        # Initialize embedding manager
        if embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
        else:
            self.embedding_manager = embedding_manager
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        print(f"✅ ChromaDB initialized: {self.collection_name}")
        print(f"   Persist directory: {self.persist_directory}")
        print(f"   Embedding dimension: {self.embedding_manager.embedding_dimension}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We handle embeddings ourselves
            )
            print(f"📚 Loaded existing collection: {self.collection_name}")
            print(f"   Documents in collection: {collection.count()}")
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
                embedding_function=None
            )
            print(f"🆕 Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
        embeddings: List[List[float]] = None
    ) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text contents
            metadatas: List of metadata dicts (one per text)
            ids: Optional list of IDs (will be generated if not provided)
            embeddings: Pre-computed embeddings (will be generated if not provided)
        
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate embeddings if not provided
        if embeddings is None:
            print(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_manager.embed_documents(texts)
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure all metadata values are strings, ints, or floats (ChromaDB requirement)
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(texts)} documents to {self.collection_name}")
        
        return ids
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata values to ChromaDB-compatible types"""
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = "None"
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized
    
    def query(
        self,
        query_text: str = None,
        query_embedding: List[float] = None,
        n_results: int = DEFAULT_TOP_K,
        where: Dict = None,
        where_document: Dict = None
    ) -> Dict:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: Query text (will generate embedding)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Metadata filter (e.g., {"source_file": "invoice.pdf"})
            where_document: Document content filter
        
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        # Generate query embedding if not provided
        if query_embedding is None:
            if query_text is None:
                raise ValueError("Either query_text or query_embedding must be provided")
            query_embedding = self.embedding_manager.embed_query(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        # Flatten results (ChromaDB returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter_metadata: Dict = None,
        score_threshold: float = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform similarity search and return results with scores
        
        Args:
            query: Query text
            k: Number of results
            filter_metadata: Metadata filter
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of tuples: (document_text, similarity_score, metadata)
        """
        results = self.query(
            query_text=query,
            n_results=k,
            where=filter_metadata
        )
        
        # Convert distances to similarity scores (1 - distance for cosine)
        similarities = [1 - dist for dist in results["distances"]]
        
        # Filter by threshold if provided
        threshold = score_threshold or SIMILARITY_THRESHOLD
        
        filtered_results = []
        for doc, sim, meta in zip(
            results["documents"],
            similarities,
            results["metadatas"]
        ):
            if sim >= threshold:
                filtered_results.append((doc, sim, meta))
        
        return filtered_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Dict with document, metadata, and embedding
        """
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not results["ids"]:
            return None
        
        return {
            "id": results["ids"][0],
            "document": results["documents"][0],
            "metadata": results["metadatas"][0],
            "embedding": results["embeddings"][0] if results["embeddings"] else None
        }
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)
        print(f"🗑️ Deleted {len(ids)} documents")
    
    def delete_by_metadata(self, where: Dict) -> None:
        """Delete documents matching metadata filter"""
        self.collection.delete(where=where)
        print(f"🗑️ Deleted documents matching filter: {where}")
    
    def update_document(
        self,
        doc_id: str,
        text: str = None,
        metadata: Dict = None
    ) -> None:
        """
        Update a document's text and/or metadata
        
        Args:
            doc_id: Document ID to update
            text: New text (will regenerate embedding)
            metadata: New metadata
        """
        update_kwargs = {"ids": [doc_id]}
        
        if text is not None:
            embedding = self.embedding_manager.embed_query(text)
            update_kwargs["documents"] = [text]
            update_kwargs["embeddings"] = [embedding]
        
        if metadata is not None:
            update_kwargs["metadatas"] = [self._sanitize_metadata(metadata)]
        
        self.collection.update(**update_kwargs)
        print(f"✏️ Updated document: {doc_id}")
    
    def count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()
    
    def get_all_documents(
        self,
        limit: int = None,
        offset: int = 0
    ) -> Dict:
        """
        Get all documents from the collection
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        
        Returns:
            Dict with ids, documents, metadatas
        """
        results = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )
        
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }
    
    def clear_collection(self) -> None:
        """Delete all documents from the collection"""
        # Get all IDs
        all_data = self.get_all_documents()
        if all_data["ids"]:
            self.delete_documents(all_data["ids"])
        print(f"🧹 Cleared collection: {self.collection_name}")
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        print(f"🔄 Reset collection: {self.collection_name}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the vector store"""
        total_docs = self.count()
        
        stats = {
            "collection_name": self.collection_name,
            "total_documents": total_docs,
            "embedding_dimension": self.embedding_manager.embedding_dimension,
            "embedding_provider": self.embedding_manager.provider_name,
            "persist_directory": self.persist_directory
        }
        
        if total_docs > 0:
            # Get sample metadata
            sample = self.get_all_documents(limit=1)
            if sample["metadatas"]:
                stats["sample_metadata"] = sample["metadatas"][0]
        
        return stats
    
    def export_to_json(self, filepath: str = None) -> str:
        """Export all documents to JSON file"""
        import json
        from pathlib import Path
        
        if filepath is None:
            filepath = CHROMA_DB_DIR / f"{self.collection_name}_export.json"
        
        all_data = self.get_all_documents()
        
        export_data = {
            "collection_name": self.collection_name,
            "total_documents": len(all_data["ids"]),
            "documents": [
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta
                }
                for doc_id, doc, meta in zip(
                    all_data["ids"],
                    all_data["documents"],
                    all_data["metadatas"]
                )
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exported to: {filepath}")
        return str(filepath)


# Example usage
if __name__ == "__main__":
    print("Initializing ChromaDB Vector Store...")
    
    # Create vector store
    vector_store = ChromaVectorStore()
    
    # Sample documents
    documents = [
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning algorithms require large datasets for training.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information."
    ]
    
    metadatas = [
        {"topic": "AI", "category": "general"},
        {"topic": "ML", "category": "technical"},
        {"topic": "DL", "category": "technical"},
        {"topic": "NLP", "category": "application"},
        {"topic": "CV", "category": "application"}
    ]
    
    # Add documents
    doc_ids = vector_store.add_documents(documents, metadatas)
    print(f"\nAdded {len(doc_ids)} documents")
    
    # Query
    query = "How does AI work with data?"
    print(f"\nQuerying: '{query}'")
    results = vector_store.similarity_search(query, k=3)
    
    print("\nTop Results:")
    for doc, score, meta in results:
        print(f"  Score: {score:.3f} | Topic: {meta.get('topic')}")
        print(f"  Text: {doc[:80]}...")
        print()
    
    # Statistics
    stats = vector_store.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
