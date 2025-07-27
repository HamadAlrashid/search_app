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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)


class Retriever(BaseRetriever):
    """
    A hybrid retriever that combines sparse and dense retrieval methods.

    It supports hybrid retrieval methods:
        - Sparse retrieval: Full-text search with keyword matching and BM25 ranking
        - Dense retrieval: Vector search for semantic similarity
        - Hybrid retrieval:
            - Reciprocal rank fusion (RRF): Combines sparse and dense results
            - Interleaved merging: Round-robin merging of results from multiple retrievers
    """

    # Pydantic fields
    indexer: Indexer = Field(default=None)
    merger_method: str = Field(default="rrf")
    sparse_weight: float = Field(default=0.5) # weight for sparse retrieval (1 means only sparse retrieval, 0 means only dense retrieval)
    k: int = Field(default=10) # number of documents to retrieve

    def __init__(self, indexer: Indexer, merger_method: str = "rrf", sparse_weight: float = 0.5, k: int = 10, **kwargs):
        # Call parent constructor with the fields
        super().__init__(indexer=indexer, merger_method=merger_method, sparse_weight=sparse_weight, k=k, **kwargs)
        self.indexer.sparse_index.k = k
        
        

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
        
            
        else:
            logger.warning(f"Using Hybrid retrieval with 0.5 weights")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], 
                weights=[0.5, 0.5]
            )
            return ensemble_retriever.invoke(query)

    def multi_search(self, query: List[str]) -> List[List[Document]]:
        """
        Hybrid retrieval with multiple queries with no rank fusion or merging
        """
        sparse_retriever: BaseRetriever = self.indexer.sparse_index

        dense_index = self.indexer.dense_index # cannot be invoked directly, so we need to use the as_retriever method
        dense_retriever: VectorStoreRetriever = dense_index.as_retriever(search_kwargs={"k": self.k})
        results = []
        for query_text in query:
            results.append(sparse_retriever.invoke(query_text))
            results.append(dense_retriever.invoke(query_text))
        return results
    
    def apply_reciprocal_rank_fusion(
        self, query_results: List[List[Document]], k: int = 60
    ) -> List[Document]:
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
        logger.info(f"RRF fusion: {len(rrf_scores)} unique docs out of {sum(len(doc_list) for doc_list in query_results)} docs, returning top {len(result_docs)}")
        
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