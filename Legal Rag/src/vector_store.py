"""
Vector Store module using ChromaDB with Arabic embeddings
"""

import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

from .config import get_config


logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    """HF token for authenticated downloads (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


class ArabicEmbeddingFunction:
    """Custom embedding function for Arabic text using sentence transformers"""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize Arabic embedding function.

        Args:
            model_name: Name of the sentence transformer model (or HF model; SentenceTransformer wraps with mean pooling).
            device: Device to run on (auto, cuda, or cpu). auto => cuda if available else cpu.
        """
        # Resolve device: auto -> cuda if available else cpu
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        token = _get_hf_token()
        logger.info(
            f"Loading embedding model: {model_name} on {self.device}. "
            "SentenceTransformer uses mean pooling; 'UNEXPECTED keys' when loading non-sentence-transformers models (e.g. AraBERT) is expected."
        )
        try:
            self.model = SentenceTransformer(model_name, device=self.device, token=token)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            fallback_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            logger.info(f"Falling back to: {fallback_model}")
            self.model = SentenceTransformer(fallback_model, device=self.device, token=token)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        # Convert texts to list if needed
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return embeddings.tolist()


class VectorStore:
    """Vector store for labor law documents using ChromaDB"""

    def __init__(self, config=None):
        """
        Initialize vector store

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.embedding_function = None
        self.client = None
        self.collection = None

        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        # Create embedding function
        self.embedding_function = ArabicEmbeddingFunction(
            model_name=self.config.embeddings.model_name,
            device=self.config.get_device()
        )

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {self.config.vector_store.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=self.config.vector_store.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.config.vector_store.collection_name,
                metadata={"hnsw:space": self.config.vector_store.distance_metric}
            )
            logger.info(f"Collection '{self.config.vector_store.collection_name}' ready. Documents: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Add documents to the vector store

        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs

        Returns:
            True if successful
        """
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_function(texts)

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(texts)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Metadata filter (e.g., {"article": "1"})
            where_document: Document content filter

        Returns:
            Query results dictionary
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query_text])[0]

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )

            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def query_with_scores(
        self,
        query_text: str,
        n_results: int = 5,
        min_score: float = 0.0,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query and return results with similarity scores

        Args:
            query_text: Query string
            n_results: Number of results to return
            min_score: Minimum similarity score threshold
            where: Metadata filter

        Returns:
            List of results with scores
        """
        results = self.query(query_text, n_results, where)

        formatted_results = []
        if results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                # Convert distance to similarity score (for cosine distance: similarity = 1 - distance)
                distance = results["distances"][0][i]
                similarity = 1 - distance

                if similarity >= min_score:
                    formatted_results.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "score": similarity,
                        "distance": distance
                    })

        return formatted_results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document dictionary or None
        """
        try:
            results = self.collection.get(ids=[doc_id])

            if results["documents"]:
                return {
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "id": results["ids"][0]
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.config.vector_store.collection_name)
            logger.info(f"Deleted collection: {self.config.vector_store.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "name": self.config.vector_store.collection_name,
                "document_count": count,
                "persist_directory": self.config.vector_store.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


if __name__ == "__main__":
    # Test vector store
    import logging
    logging.basicConfig(level=logging.INFO)

    # Initialize vector store
    vs = VectorStore()

    # Get stats
    stats = vs.get_collection_stats()
    print(f"Collection stats: {stats}")

    # Test query if collection has documents
    if stats["document_count"] > 0:
        results = vs.query_with_scores(
            "ما هي حقوق العامل في الإجازات؟",
            n_results=3
        )
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Article {result['metadata'].get('article', 'N/A')} (Score: {result['score']:.3f})")
            print(f"   {result['text'][:200]}...")
