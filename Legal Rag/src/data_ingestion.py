"""
Data Ingestion module for loading labor law documents into ChromaDB
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .config import get_config
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class DataIngestion:
    """Load labor law data from JSONL files into ChromaDB"""

    def __init__(self, config=None):
        """
        Initialize data ingestion

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.vector_store = VectorStore(self.config)

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of document dictionaries
        """
        documents = []
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return documents

        logger.info(f"Loading data from: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                        continue

            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return []

    def validate_document(self, doc: Dict[str, Any]) -> bool:
        """
        Validate that a document has all required fields

        Args:
            doc: Document dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'law', 'article', 'chunk_index', 'text', 'source']

        for field in required_fields:
            if field not in doc:
                logger.warning(f"Document missing required field: {field}")
                return False

            if not doc[field] and doc[field] != 0:  # Allow 0 as valid value
                logger.warning(f"Document has empty required field: {field}")
                return False

        return True

    def prepare_documents_for_ingestion(
        self,
        documents: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Prepare documents for ChromaDB ingestion

        Args:
            documents: List of document dictionaries from JSONL

        Returns:
            Tuple of (texts, metadatas, ids)
        """
        texts = []
        metadatas = []
        ids = []

        for doc in documents:
            if not self.validate_document(doc):
                continue

            # Use normalized_text if available, otherwise use text
            text = doc.get('normalized_text', doc.get('text', ''))

            # Prepare metadata (ChromaDB requires metadata values to be strings, ints, or floats)
            metadata = {
                'law': str(doc.get('law', '')),
                'article': str(doc.get('article', '')),
                'chunk_index': int(doc.get('chunk_index', 0)),
                'source': str(doc.get('source', ''))
            }

            texts.append(text)
            metadatas.append(metadata)
            ids.append(doc['id'])

        return texts, metadatas, ids

    def ingest_from_file(self, file_path: str, batch_size: int = 50) -> bool:
        """
        Ingest documents from a JSONL file into ChromaDB

        Args:
            file_path: Path to JSONL file
            batch_size: Number of documents to process at a time

        Returns:
            True if successful
        """
        # Load documents
        documents = self.load_jsonl(file_path)

        if not documents:
            logger.error("No documents loaded")
            return False

        # Prepare for ingestion
        texts, metadatas, ids = self.prepare_documents_for_ingestion(documents)

        if not texts:
            logger.error("No valid documents to ingest")
            return False

        logger.info(f"Ingesting {len(texts)} documents in batches of {batch_size}...")

        # Ingest in batches with progress bar
        success = True
        for i in tqdm(range(0, len(texts), batch_size), desc="Ingesting batches"):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            if not self.vector_store.add_documents(batch_texts, batch_metadatas, batch_ids):
                logger.error(f"Failed to ingest batch {i // batch_size + 1}")
                success = False

        if success:
            logger.info("Data ingestion completed successfully")
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Collection now contains {stats['document_count']} documents")

        return success

    def ingest_labor_law_data(self) -> bool:
        """
        Ingest the default labor law data file

        Returns:
            True if successful
        """
        file_path = self.config.data.labor_law_file
        logger.info(f"Ingesting labor law data from: {file_path}")
        return self.ingest_from_file(file_path)

    def reingest(self, file_path: str = None) -> bool:
        """
        Delete existing collection and re-ingest data

        Args:
            file_path: Path to JSONL file (uses default if None)

        Returns:
            True if successful
        """
        logger.warning("Deleting existing collection and re-ingesting data...")

        # Delete existing collection
        self.vector_store.delete_collection()

        # Reinitialize vector store
        self.vector_store = VectorStore(self.config)

        # Ingest data
        if file_path:
            return self.ingest_from_file(file_path)
        else:
            return self.ingest_labor_law_data()


def main():
    """Main function for standalone execution"""
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Ingest labor law data into ChromaDB")
    parser.add_argument(
        '--file',
        type=str,
        help='Path to JSONL file (uses default from config if not specified)'
    )
    parser.add_argument(
        '--reingest',
        action='store_true',
        help='Delete existing data and re-ingest'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for ingestion (default: 50)'
    )

    args = parser.parse_args()

    # Initialize ingestion
    ingestion = DataIngestion()

    # Perform ingestion
    if args.reingest:
        success = ingestion.reingest(args.file)
    elif args.file:
        success = ingestion.ingest_from_file(args.file, args.batch_size)
    else:
        success = ingestion.ingest_labor_law_data()

    if success:
        print("\n✓ Data ingestion completed successfully!")

        # Show stats
        stats = ingestion.vector_store.get_collection_stats()
        print(f"\nCollection Statistics:")
        print(f"  Name: {stats['name']}")
        print(f"  Documents: {stats['document_count']}")
        print(f"  Location: {stats['persist_directory']}")

        # Test query
        print("\nTesting retrieval with sample query...")
        results = ingestion.vector_store.query_with_scores(
            "حقوق العامل",
            n_results=3
        )
        print(f"Found {len(results)} relevant documents")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Article {result['metadata'].get('article')} (Score: {result['score']:.3f})")
            print(f"   {result['text'][:150]}...")

    else:
        print("\n✗ Data ingestion failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
