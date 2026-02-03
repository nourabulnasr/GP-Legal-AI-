"""
RAG (Retrieval-Augmented Generation) Engine for contract analysis
Implements two-stage retrieval: section-level and clause-level
"""

import logging
from typing import List, Dict, Any, Optional

from .config import get_config
from .vector_store import VectorStore
from .contract_parser import ParsedContract, ContractSection


logger = logging.getLogger(__name__)


class RetrievedContext:
    """Represents retrieved legal context for a contract clause"""

    def __init__(
        self,
        query_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        retrieval_stage: str = "unknown"
    ):
        """
        Initialize retrieved context

        Args:
            query_text: Original query text
            retrieved_chunks: List of retrieved law chunks with scores
            retrieval_stage: Stage of retrieval (section or clause)
        """
        self.query_text = query_text
        self.retrieved_chunks = retrieved_chunks
        self.retrieval_stage = retrieval_stage

    def get_context_text(self, max_chunks: Optional[int] = None) -> str:
        """
        Get formatted context text for LLM prompt

        Args:
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string
        """
        chunks = self.retrieved_chunks[:max_chunks] if max_chunks else self.retrieved_chunks

        if not chunks:
            return "لم يتم العثور على قوانين ذات صلة."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            article = metadata.get('article', 'N/A')
            law = metadata.get('law', 'N/A')
            text = chunk['text']
            score = chunk.get('score', 0)

            context_parts.append(
                f"المادة {article} - {law} (الصلة: {score:.2f}):\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_top_articles(self, n: int = 3) -> List[str]:
        """
        Get list of top N article numbers

        Args:
            n: Number of articles to return

        Returns:
            List of article numbers
        """
        articles = []
        for chunk in self.retrieved_chunks[:n]:
            article = chunk['metadata'].get('article')
            if article and article not in articles:
                articles.append(article)

        return articles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query_text": self.query_text,
            "retrieval_stage": self.retrieval_stage,
            "num_chunks": len(self.retrieved_chunks),
            "chunks": [
                {
                    "article": chunk['metadata'].get('article'),
                    "law": chunk['metadata'].get('law'),
                    "score": chunk.get('score'),
                    "text": chunk['text'][:200] + "..."  # Truncate for summary
                }
                for chunk in self.retrieved_chunks
            ]
        }


class RAGEngine:
    """RAG engine for retrieving relevant laws for contract analysis"""

    def __init__(self, config=None):
        """
        Initialize RAG engine

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.vector_store = VectorStore(self.config)

    def retrieve_for_text(
        self,
        text: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        stage: str = "single"
    ) -> RetrievedContext:
        """
        Retrieve relevant laws for a given text

        Args:
            text: Text to query
            top_k: Number of results (uses config default if None)
            min_score: Minimum similarity score (uses config default if None)
            stage: Retrieval stage identifier

        Returns:
            RetrievedContext object
        """
        top_k = top_k or self.config.rag.clause_top_k
        min_score = min_score or self.config.rag.min_similarity_score

        logger.debug(f"Retrieving for text (top_k={top_k}, min_score={min_score}): {text[:100]}...")

        # Query vector store
        results = self.vector_store.query_with_scores(
            query_text=text,
            n_results=top_k,
            min_score=min_score
        )

        logger.info(f"Retrieved {len(results)} relevant law chunks")

        return RetrievedContext(
            query_text=text,
            retrieved_chunks=results,
            retrieval_stage=stage
        )

    def retrieve_for_section(self, section: ContractSection) -> RetrievedContext:
        """
        Retrieve relevant laws for a contract section (first pass)

        Args:
            section: ContractSection object

        Returns:
            RetrievedContext object
        """
        return self.retrieve_for_text(
            text=section.text,
            top_k=self.config.rag.section_top_k,
            stage="section"
        )

    def retrieve_for_clause(self, clause_text: str) -> RetrievedContext:
        """
        Retrieve relevant laws for a specific clause (second pass)

        Args:
            clause_text: Clause text

        Returns:
            RetrievedContext object
        """
        return self.retrieve_for_text(
            text=clause_text,
            top_k=self.config.rag.clause_top_k,
            stage="clause"
        )

    def retrieve_for_contract(
        self,
        contract: ParsedContract,
        use_two_stage: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant laws for entire contract

        Args:
            contract: ParsedContract object
            use_two_stage: Use two-stage retrieval if True

        Returns:
            List of dictionaries with section and retrieved context
        """
        results = []

        logger.info(f"Retrieving laws for contract with {len(contract.sections)} sections")

        if use_two_stage and len(contract.sections) > 0:
            # Stage 1: Section-level retrieval
            logger.info("Stage 1: Section-level retrieval")
            for i, section in enumerate(contract.sections, 1):
                logger.debug(f"Processing section {i}/{len(contract.sections)}")

                context = self.retrieve_for_section(section)

                results.append({
                    "section": section,
                    "context": context,
                    "stage": "section"
                })

        else:
            # Single-stage: Retrieve for full contract
            logger.info("Single-stage retrieval for full contract")

            # Use first 1000 words to avoid overwhelming the retrieval
            full_text = contract.get_full_text()
            truncated_text = " ".join(full_text.split()[:1000])

            context = self.retrieve_for_text(
                text=truncated_text,
                top_k=self.config.rag.section_top_k,
                stage="contract"
            )

            results.append({
                "section": None,
                "context": context,
                "stage": "contract"
            })

        logger.info(f"Retrieved context for {len(results)} sections/contract")
        return results

    def get_relevant_articles_summary(
        self,
        contract: ParsedContract
    ) -> Dict[str, Any]:
        """
        Get a summary of all relevant articles for the contract

        Args:
            contract: ParsedContract object

        Returns:
            Dictionary with article statistics
        """
        retrieval_results = self.retrieve_for_contract(contract)

        all_articles = []
        article_scores = {}

        for result in retrieval_results:
            context = result['context']
            for chunk in context.retrieved_chunks:
                article = chunk['metadata'].get('article')
                score = chunk.get('score', 0)

                if article:
                    all_articles.append(article)

                    if article not in article_scores:
                        article_scores[article] = []
                    article_scores[article].append(score)

        # Calculate statistics
        unique_articles = set(all_articles)
        article_avg_scores = {
            article: sum(scores) / len(scores)
            for article, scores in article_scores.items()
        }

        # Sort by average score
        sorted_articles = sorted(
            article_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "total_articles_referenced": len(all_articles),
            "unique_articles": len(unique_articles),
            "top_articles": [
                {"article": article, "avg_score": score}
                for article, score in sorted_articles[:10]
            ],
            "all_articles": list(unique_articles)
        }


if __name__ == "__main__":
    # Test RAG engine
    import sys
    logging.basicConfig(level=logging.INFO)

    # Initialize engine
    engine = RAGEngine()

    # Test with sample query
    test_queries = [
        "الأجر الأساسي للعامل وساعات العمل",
        "الإجازات السنوية والمرضية",
        "إنهاء عقد العمل والفصل"
    ]

    print("Testing RAG Engine with sample queries:\n")

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        context = engine.retrieve_for_text(query, top_k=3)

        print(f"Retrieved {len(context.retrieved_chunks)} chunks:")
        for i, chunk in enumerate(context.retrieved_chunks, 1):
            article = chunk['metadata'].get('article')
            score = chunk.get('score')
            text_preview = chunk['text'][:150]

            print(f"\n{i}. Article {article} (Score: {score:.3f})")
            print(f"   {text_preview}...")

    # If contract file provided, test with contract
    if len(sys.argv) > 1:
        from .contract_parser import ContractParser

        print("\n" + "=" * 80)
        print("Testing with contract file")
        print("=" * 80)

        parser = ContractParser()
        contract = parser.parse_file(sys.argv[1])

        if contract:
            summary = engine.get_relevant_articles_summary(contract)

            print(f"\nRelevant Articles Summary:")
            print(f"  Total references: {summary['total_articles_referenced']}")
            print(f"  Unique articles: {summary['unique_articles']}")
            print(f"\n  Top 5 articles:")
            for item in summary['top_articles'][:5]:
                print(f"    - Article {item['article']} (avg score: {item['avg_score']:.3f})")
