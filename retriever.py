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
from prompts import RAG_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)

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
        pass

    def search(self, query: str) -> list[Document]:
        
        #adjust the k value for the sparse retriever
        self.indexer.sparse_index.k = self.k
        sparse_retriever : BaseRetriever = self.indexer.sparse_index

        dense_index = self.indexer.dense_index # cannot be invoked directly, so we need to use the as_retriever method
        dense_retriever : VectorStoreRetriever = dense_index.as_retriever(search_kwargs={"k": self.k})

        
        if self.merger_method == "rrf":
            # this implements RRF with weights
            logger.info(f"Using RRF with weights {self.sparse_weight} and {1 - self.sparse_weight}")
            ensemble_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], weights=[self.sparse_weight, 1 - self.sparse_weight])
            return ensemble_retriever.invoke(query)
        elif self.merger_method == "interleaved": 
            logger.info(f"Using interleaved merging")
            merger = MergerRetriever(retrievers=[sparse_retriever, dense_retriever])
            return merger.invoke(query)
        elif self.merger_method == "cross_encoder":
            logger.info(f"Using cross-encoder")
            pass
        elif self.merger_method == "multi_query":
            logger.info(f"Using multi-query")
            pass
        elif self.merger_method == "query_decomposition":
            logger.info(f"Using query decomposition")
            pass
        elif self.merger_method == "hyde":
            logger.info(f"Using HyDE")
            pass



    
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
        