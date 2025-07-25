from indexer import Indexer
import logging
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MergerRetriever
from pydantic import Field
import sys
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever
from collections import defaultdict
from itertools import chain
from langchain.retrievers.ensemble import unique_by_key
from prompts import RAG_PROMPT, MULTI_QUERY_PROMPT
from structured_output.multiquery import MultiQuery
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)


def get_llm(task: str = "multi_query", model: str = None, **kwargs):
    """
    create LLMs on demand.
    
    Args:
        task: The task type (e.g., 'multi_query', 'hyde', 'decomposition')
        model: Model name to use (defaults based on task)
        **kwargs: Additional parameters for the LLM
        
    Returns:
        Configured LLM instance
    """
    default_models = {
        "multi_query": "gpt-3.5-turbo",
    }
    
    model_name = model or default_models.get(task, "gpt-3.5-turbo")
    
    
    default_params = {
        "multi_query": {"temperature": 0, "max_tokens": 500},
        "hyde": {"temperature": 0.3, "max_tokens": 1000}
    }
    
    # Merge default params with provided kwargs
    params = {**default_params.get(task, {"temperature": 0}), **kwargs}
    
    return ChatOpenAI(model=model_name, **params)

class Retriever(BaseRetriever):
    """
    A hybrid retriever that uses sparse and dense retrieval to provide relevant answers that satisfy the user's query.

    It supports multiple retrieval methods:
        - Sparse retrieval: Full-text search with keyword matching and BM25 ranking
        - Dense retrieval: Vector search for semantic similarity
        - Reciprocal rank fusion (RRF)
        - Interleaved or Round-robin merging of results from multiple retrievers
        - Re-ranking with a Cross-encoder
        - Multi-query generation: Generate multiple queries to retrieve more relevant documents
        - Query Decomposition: Decompose the query into multiple sub-queries
        - HyDE (Hypothetical Document Embeddings): Generate hypothetical documents to retrieve more relevant documents
    """

    # Pydantic fields
    indexer: Indexer = Field(default=None)
    merger_method: str = Field(default="rrf")
    sparse_weight: float = Field(default=0.5) # weight for sparse retrieval (1 means only sparse retrieval, 0 means only dense retrieval)
    k: int = Field(default=10) # number of documents to retrieve

    def __init__(self, indexer: Indexer, merger_method: str = "rrf", sparse_weight: float = 0.5, k: int = 10, **kwargs):
        # Call parent constructor with the fields
        super().__init__(indexer=indexer, merger_method=merger_method, sparse_weight=sparse_weight, k=k, **kwargs)
        

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """
        Get the relevant documents for a given query.
        """
        return self.search(query)

    def search(self, query: str) -> list[Document]:
        """
        Search for relevant documents using the configured retrieval method.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Starting search with method: {self.merger_method}")
        
        # Adjust the k value for the sparse retriever
        self.indexer.sparse_index.k = self.k
        sparse_retriever: BaseRetriever = self.indexer.sparse_index

        dense_index = self.indexer.dense_index # cannot be invoked directly, so we need to use the as_retriever method
        dense_retriever: VectorStoreRetriever = dense_index.as_retriever(search_kwargs={"k": self.k})
    
        
        if self.merger_method == "rrf":
            # this implements RRF with weights
            logger.info(f"Using RRF with weights {self.sparse_weight} and {1 - self.sparse_weight}")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[self.sparse_weight, 1 - self.sparse_weight]
            )
            return ensemble_retriever.invoke(query)
            
        elif self.merger_method == "interleaved": 
            logger.info(f"Using interleaved merging")
            merger = MergerRetriever(retrievers=[sparse_retriever, dense_retriever])
            return merger.invoke(query)
            
        elif self.merger_method == "sparse_only" or self.sparse_weight == 1:
            logger.info(f"Using sparse retrieval only")
            return sparse_retriever.invoke(query)
            
        elif self.merger_method == "dense_only" or self.sparse_weight == 0:
            logger.info(f"Using dense retrieval only")
            return dense_retriever.invoke(query)
            
        elif self.merger_method == "cross_encoder":
            logger.info(f"Using cross-encoder (not implemented yet, falling back to RRF)")
            # TODO: Implement cross-encoder re-ranking
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[self.sparse_weight, 1 - self.sparse_weight]
            )
            return ensemble_retriever.invoke(query)
            
        elif self.merger_method == "multi_query":
            number_of_queries = 3
            logger.info(f"Using multi-query with {number_of_queries} queries")
            
            generated_queries = self._generate_multi_query(query, number_of_queries)
            
            all_result_lists = []
            
            for query_text in generated_queries:
                try:
                    sparse_docs = sparse_retriever.invoke(query_text)
                    all_result_lists.append(sparse_docs)
                    
                    dense_docs = dense_retriever.invoke(query_text)
                    all_result_lists.append(dense_docs)
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve for query '{query_text}': {e}")
                    
            
            # Apply RRF on multiple sorted document lists
            logger.info(f"Applying RRF fusion across {len(all_result_lists)} result lists")
            return self._apply_reciprocal_rank_fusion(all_result_lists)
                
        elif self.merger_method == "query_decomposition":
            logger.info(f"Using query decomposition (not implemented yet, falling back to RRF)")
            # TODO: Implement query decomposition
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[self.sparse_weight, 1 - self.sparse_weight]
            )
            return ensemble_retriever.invoke(query)
            
        elif self.merger_method == "hyde":
            logger.info(f"Using HyDE (not implemented yet, falling back to RRF)")
            # TODO: Implement HyDE (Hypothetical Document Embeddings)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[self.sparse_weight, 1 - self.sparse_weight]
            )
            return ensemble_retriever.invoke(query)
            
        else:
            logger.warning(f"Unknown merger method: {self.merger_method}, falling back to RRF")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[self.sparse_weight, 1 - self.sparse_weight]
            )
            return ensemble_retriever.invoke(query)


    def _generate_multi_query(self, query: str, number_of_queries: int = 3) -> MultiQuery:
        """
        Generate multiple queries using LLM
        """
        llm = get_llm("multi_query")
        structured_llm = llm.with_structured_output(MultiQuery)
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT).partial(number_of_queries=number_of_queries)
        new_queries_chain = prompt | structured_llm
        
        generated_queries = new_queries_chain.invoke({"query": query})
        logger.info(f"Generated queries: {generated_queries}")
        
        generated_queries.queries.append(query)
        logger.info(f"Total queries for multi-query search: {len(generated_queries)}")

        return generated_queries

    def _apply_reciprocal_rank_fusion(
        self, query_results: list[list[Document]], k: int = 60
    ) -> list[Document]:
        """
        Apply Reciprocal Rank Fusion to merge results from multiple queries.
        
        Args:
            query_results: List of document lists from different queries
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked list of unique documents
        """
        rrf_scores = {}
        
        for doc_list in query_results:
            for rank, doc in enumerate(doc_list):
                # Use page_content + metadata for unique identification
                doc_key = self._get_document_key(doc)
                
                if doc_key not in rrf_scores:
                    rrf_scores[doc_key] = {
                        'document': doc,
                        'score': 0.0
                    }
                
                # RRF formula: score += 1 / (k + rank)
                rrf_scores[doc_key]['score'] += 1.0 / (k + rank)
        
        # Sort by RRF score (descending) and return top k documents
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        result_docs = [item['document'] for item in sorted_docs[:self.k]]
        logger.info(f"RRF fusion: {len(rrf_scores)} unique docs, returning top {len(result_docs)}")
        
        return result_docs
    
    def _get_document_key(self, doc: Document) -> str:
        """
        Generate a unique key for a document to handle duplicates.
        Uses a combination of content hash and metadata for uniqueness.
        """
        # Use first 100 chars of content + source metadata for key
        content_preview = doc.page_content[:100].strip()
        source = doc.metadata.get('source', '')
        page = doc.metadata.get('page', '')
        
        # Create a composite key
        return f"{hash(content_preview)}_{source}_{page}"

    # copied from langchain.retrievers.ensemble.py
    def weighted_reciprocal_rank(
        self, doc_lists: list[list[Document]]
    ) -> list[Document]:
        """

        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    (
                        doc.page_content
                        if self.id_key is None
                        else doc.metadata[self.id_key]
                    )
                ] += weight / (rank + self.c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
        )
        return sorted_docs
        