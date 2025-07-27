import os
import faiss
import sqlite3
import json
import re
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing import Literal, List, Union, Tuple, Dict, Any
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from pydantic import Field
import logging 
import sys

try:
    from local_models import LocalEmbeddings, get_local_embeddings
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)


class FTS5Retriever(BaseRetriever):
    """
    A sparse retriever that uses SQLite FTS5 for full-text search and keyword matching to retrieve documents which are ranked by BM25.

    - How it works:
        Full-text search utilizes an inverted index which is a data structure that maps each token to a posting list. 
        A posting list is a list of documents where that token appears.
        Instead of using a hash table, the FTS5 engine uses an optimized B-tree to store the posting lists.

    - Where is the TF-IDF / BM25 math done?
        After the matching documents are retrieved, the TF-IDF scores are calculated to rank the documents.
    """
    
    # Declare Pydantic fields
    db_path: str = Field(default="local_fts_db.sqlite")
    k: int = Field(default=10)
    conn: sqlite3.Connection = Field(default=None, exclude=True)  # Exclude from serialization
    
    def __init__(self, db_path: str = "local_fts_db.sqlite", documents: List[Document] = None, k: int = 10, **kwargs):
        """
        Initialize the FTS5 retriever.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Call parent constructor with the fields
        super().__init__(db_path=db_path, k=k, **kwargs)
        
        # Set connection to None initially (will be set in _setup_database)
        self.conn = None
        self._setup_database()

        if documents:
            self.add_documents(documents)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Get the relevant documents for a given query."""
        return self.search(query, limit=self.k)
    
    def _setup_database(self):
        """Set up the SQLite database with FTS5 virtual table."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create an FTS5 virtual table 
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id UNINDEXED,
                content,
                metadata,
                tokenize='porter unicode61' 
            );
        """) 
        # porter: porter stemming algorithm
        # unicode61: Unicode 6.1 normalization
        
        
        
        self.conn.commit()
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the FTS5 index.
        
        Args:
            documents: List of Document objects
        """
        cursor = self.conn.cursor()
        
        for doc in documents:
            doc_id = doc.id
            content = doc.page_content
            metadata = json.dumps(doc.metadata)
            
            # Insert into FTS5 table
            cursor.execute("""
                INSERT OR REPLACE INTO documents_fts (doc_id, content, metadata)
                VALUES (?, ?, ?)
            """, (doc_id, content, metadata))
            
            
        
        self.conn.commit()
    
    def search(self, query: str, limit: int = 10) -> List[Document]:
        """
        Search for documents using FTS5 full-text search.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with scores

        The BM25 parameters are:
            k1 and b which control term frequency normalization and document length normalization respectively.
            The parameters are set to 1.2 and 0.75 by default, and cannot be changed without recompiling the FTS5 engine.
        """
        cursor = self.conn.cursor()
        
        # Use FTS5 MATCH for full-text search with BM25 ranking
        sql_query = """
            SELECT 
                fts.doc_id,
                fts.content,
                fts.metadata,
                bm25(documents_fts) as score
            FROM documents_fts fts
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts) DESC
            LIMIT ?
        """
        #query = f'"{query}"'
        
        # Sanitize query by removing special characters that might interfere with FTS5 search
        sanitized_query = re.sub(r'[?!\-@#$%^&*()[\]{}|\\:;"\'<>,.\/=+~`]', ' ', query)
        # Clean up multiple spaces and strip whitespace
        sanitized_query = ' '.join(sanitized_query.split())
        
        cursor.execute(sql_query, (sanitized_query, limit))
        results = cursor.fetchall()
        
        documents = []
        for row in results:
            d = Document(
                id=row[0],
                page_content=row[1],
                metadata=json.loads(row[2]),
            )
            documents.append(d)
        
        return documents
    
    def search_with_operators(self, query: str, limit: int = 10) -> List[Document]:
        """
        Advanced search with FTS5 operators.
        
        Supports:
        - AND: "word1 AND word2"
        - OR: "word1 OR word2"
        - NOT: "word1 NOT word2"
        - Phrase: '"exact phrase"'
        - Prefix: "word*"
        - Column search: "content: word"
        
        Args:
            query: FTS5 query with operators
            limit: Maximum number of results
            
        Returns:
            List of matching documents with scores
        """
        # TODO: 
        
        return self.search(query, limit)
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the index."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents_fts")
        return cursor.fetchone()[0]
    
    def delete_document(self, doc_id: str):
        """Delete a document from the index."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
    
    def clear_index(self):
        """Clear all documents from the index."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents_fts")
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the index."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents_fts")

        documents = []
        raw_docs : List[Tuple] = cursor.fetchall()
        for doc in raw_docs:
            if doc is not None:
                doc_id = doc[0]
                content = doc[1]
                metadata_str = doc[2]
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (ValueError, SyntaxError):
                    metadata = {} 
                    
                documents.append(Document(page_content=content, metadata=metadata, id=doc_id))
        return documents


SUPPORTED_EMBEDDING_MODELS = {
    "models/embedding-001": GoogleGenerativeAIEmbeddings,
    "text-embedding-3-large": OpenAIEmbeddings,
    # Local embedding models (use prefix 'local:' to indicate local model)
    "local:all-MiniLM-L6-v2": "LocalEmbeddings",
    "local:all-MiniLM-L12-v2": "LocalEmbeddings", 
    "local:all-mpnet-base-v2": "LocalEmbeddings",
    "local:BAAI/bge-small-en-v1.5": "LocalEmbeddings",
    "local:BAAI/bge-base-en-v1.5": "LocalEmbeddings",
    "local:paraphrase-multilingual-MiniLM-L12-v2": "LocalEmbeddings",
    "local:Qwen/Qwen3-Embedding-4B": "LocalEmbeddings",
    "local:Qwen/Qwen3-Embedding-0.6B": "LocalEmbeddings",
}

SUPPORTED_DENSE_INDEXES = {
    "faiss": FAISS,
}

SUPPORTED_SPARSE_INDEXES = {
    "bm25": FTS5Retriever,
}
class Indexer:
    """
    The Indexer class is responsible for creating and managing the dense and sparse indexes.
    """

    def __init__(self,
        index_dir: str = "indexes",
        sparse_index: str = "bm25",
        dense_index_type: str = "flat_l2",
        dense_index_params: Dict[str, Any] = None,
        embedding_model: str = "text-embedding-3-large",
    ):
            """
            Initialize the Indexer class.

            Args:
                index_dir: The directory to store the indexes.
                sparse_index: The name of the sparse index.
                dense_index_type: The type of dense FAISS index.
                dense_index_params: Parameters for the dense index.
                embedding_model: The name of the embedding model.
            """
            logger.info(f"Initializing Indexer with index_dir={index_dir}, sparse_index={sparse_index}, dense_index_type={dense_index_type}, embedding_model={embedding_model}")
            
            self.index_dir = index_dir
            os.makedirs(self.index_dir, exist_ok=True) 
            logger.info(f"Created index directory: {self.index_dir}")

            # Index names
            self.sparse_index_name = os.path.join(self.index_dir, f"{sparse_index}.sqlite")
            self.dense_index_name = "index"
            # Initialize sparse index
            if sparse_index in SUPPORTED_SPARSE_INDEXES:
                logger.info(f"Initializing sparse index: {sparse_index}")
                self.sparse_index = SUPPORTED_SPARSE_INDEXES[sparse_index](self.sparse_index_name)
                logger.info(f"Successfully initialized sparse index at: {self.sparse_index_name}")
            else:
                logger.error(f"Unsupported sparse index: {sparse_index}")
                raise ValueError(f"Unsupported sparse index: {sparse_index}")

            # Initialize embedding model
            if embedding_model in SUPPORTED_EMBEDDING_MODELS:
                logger.info(f"Initializing embedding model: {embedding_model}")
                
                # Handle local embedding models
                if embedding_model.startswith("local:"):
                    if not LOCAL_MODELS_AVAILABLE:
                        raise ImportError("local_models module not available. Please ensure the local_models.py file is present.")
                    
                    # Extract the actual model name (remove 'local:' prefix)
                    actual_model_name = embedding_model[6:]  # Remove 'local:' prefix
                    self.embedding_model: Embeddings = get_local_embeddings(model_name=actual_model_name)
                else:
                    # Handle cloud-based embedding models
                    embedding_class = SUPPORTED_EMBEDDING_MODELS[embedding_model]
                    self.embedding_model: Embeddings = embedding_class(model=embedding_model)
                
                self.embedding_dim = len(self.embedding_model.embed_query("test"))
                logger.info(f"Successfully initialized embedding model with dimension: {self.embedding_dim}")

            else:
                logger.error(f"Unsupported embedding model: {embedding_model}")
                raise ValueError(f"Unsupported embedding model: {embedding_model}")
            
            # Initialize dense index 
            self.dense_index_type = dense_index_type
            self.dense_index_params = dense_index_params or {}
            logger.info(f"Creating dense index with type: {dense_index_type}, params: {self.dense_index_params}")
            self.dense_index = self._create_vector_store()
            logger.info("Indexer initialization completed successfully")
            
            
    def _create_vector_store(self) -> FAISS:
        """ Initialize the vector store based on the index type """

        logger.info(f"Creating vector store with index type: {self.dense_index_type}")

        # Brute force with L2 (euclidean distance)
        if self.dense_index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Created IndexFlatL2 (Euclidean distance)")
        
        # Brute force with Inner-Product (cosine similarity)
        elif self.dense_index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Created IndexFlatIP (Inner Product/Cosine similarity)")
        
        # HNSW (Hierarchical-Navigable-Small-World-Graph), the standard for dense retrieval
        elif self.dense_index_type == "hnsw":
            M = self.dense_index_params.get("M", 32)  # Number of connections
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            logger.info(f"Created IndexHNSWFlat with M={M}")
            

            if "efConstruction" in self.dense_index_params:
                self.index.hnsw.efConstruction = self.dense_index_params["efConstruction"]
                logger.info(f"Set efConstruction to {self.dense_index_params['efConstruction']}")
            if "efSearch" in self.dense_index_params:
                self.index.hnsw.efSearch = self.dense_index_params["efSearch"]
                logger.info(f"Set efSearch to {self.dense_index_params['efSearch']}")
            
        # Inverted-File (IVF) + Flat (k-nearest neighbors)
        elif self.dense_index_type == "ivf_flat":
            nlist = self.dense_index_params.get("nlist", 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            logger.info(f"Created IndexIVFFlat with nlist={nlist}")
        
        # Inverted-File (IVF) + Product-Quantization (PQ)
        elif self.dense_index_type == "ivf_pq":
            nlist = self.dense_index_params.get("nlist", 100)
            m = self.dense_index_params.get("m", 8)  # Number of subquantizers
            nbits = self.dense_index_params.get("nbits", 8)  # Bits per subquantizer
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
            logger.info(f"Created IndexIVFPQ with nlist={nlist}, m={m}, nbits={nbits}")
        
        # Product-Quantization (PQ)
        elif self.dense_index_type == "pq":
            m = self.dense_index_params.get("m", 8)
            nbits = self.dense_index_params.get("nbits", 8)
            self.index = faiss.IndexPQ(self.embedding_dim, m, nbits)
            logger.info(f"Created IndexPQ with m={m}, nbits={nbits}")
            
        # Locality-Sensitive Hashing (LSH)
        elif self.dense_index_type == "lsh":
            nbits = self.dense_index_params.get("nbits", self.embedding_dim * 4)
            self.index = faiss.IndexLSH(self.embedding_dim, nbits)
            logger.info(f"Created IndexLSH with nbits={nbits}")

        vector_store = FAISS(embedding_function=self.embedding_model, index=self.index, docstore=InMemoryDocstore(), index_to_docstore_id={})
        logger.info("Successfully created FAISS vector store")
        return vector_store

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the index."""
        logger.info(f"Adding {len(documents)} documents to index")
        try:
            self.add_documents_to_sparse_index(documents)
            self.add_documents_to_dense_index(documents)
            logger.info(f"Successfully added {len(documents)} documents to index")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return False

    def add_documents_to_sparse_index(self, documents: List[Document]) -> bool:
        """Add documents to the sparse (FTS5) index."""
        logger.info(f"Adding {len(documents)} documents to sparse index")
        try:
            self.sparse_index.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to sparse index")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to sparse index: {e}")
            return False
    
    def add_documents_to_dense_index(self, documents: List[Document]) -> bool:
        """Add documents to the dense (FAISS) index."""
        logger.info(f"Adding {len(documents)} documents to dense index")
        try:
            self.dense_index.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to dense index")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to dense index: {e}")
            return False

    
    def search_sparse(self, query: str, limit: int = 10) -> List[Document]:
        """Search using the sparse (FTS5) index."""
        logger.info(f"Performing sparse search with query: '{query}', limit: {limit}")
        try:
            results = self.sparse_index.search(query, limit)
            logger.info(f"Sparse search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def search_dense(self, query: str, embedding : list[float] = None, k: int = 10) -> List[Document]:
        """Search using the dense (FAISS) index.
        
        Args:
            query: Search query (if embedding is not provided)
            embedding: Embedding of the query (Will be used if provided)
            k: Number of results to return

        Returns:
            List of documents
        """
        if embedding is None:
            logger.info(f"Performing dense search with query: '{query}', k: {k}")
            try:
                #embedding = self.embedding_model.embed_query(query)
                results = self.dense_index.similarity_search(query, k=k)
                logger.info(f"Dense search returned {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"Dense search failed: {e}")
                return []

        # make sure embedding size is correct
        if len(embedding) != self.embedding_dim:
            error_msg = f"Embedding size mismatch. Expected {self.embedding_dim}, got {len(embedding)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Performing dense search with provided embedding, k: {k}")
        try:
            results = self.dense_index.similarity_search_by_vector(embedding, k=k)
            logger.info(f"Dense search by vector returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Dense search by vector failed: {e}")
            return []
    
    def save_sparse_index(self):
        """Save the sparse index to disk."""
        logger.info("The sqlite index is on disk, which gets intialized and updated on the fly")
        pass
    
    def save_dense_index(self):
        """Save the dense index to disk."""
        logger.info(f"Saving dense index to {self.index_dir}")
        try:
            if self.dense_index:
                self.dense_index.save_local(folder_path=self.index_dir, index_name=self.dense_index_name)
                logger.info(f"Successfully saved dense index to {self.index_dir}")
            else:
                logger.warning("No dense index to save")
        except Exception as e:
            logger.error(f"Failed to save dense index: {e}")
    
    def load_dense_index(self):
        """Load the dense index from disk."""
        logger.info(f"Loading dense index from {self.index_dir}")
        try:
            if os.path.exists(self.index_dir + "/" + self.dense_index_name + ".faiss"):
                self.dense_index = FAISS.load_local(folder_path=self.index_dir, embeddings=self.embedding_model, index_name=self.dense_index_name, allow_dangerous_deserialization=True)
                logger.info(f"Dense index loaded from {self.index_dir}")
            else:
                logger.error(f"No dense index found at {self.index_dir}")
        except Exception as e:
            logger.error(f"Error loading dense index: {e}")

    def save_index(self):
        """Save the index to disk."""
        logger.info(f"Saving index to {self.index_dir}")
        try:
            self.save_dense_index()
            logger.info(f"Successfully saved index to {self.index_dir}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")


    def load_index(self):
        """Load the index from disk."""
        logger.info(f"Loading index from {self.index_dir}")
        try:
            self.load_dense_index()
            logger.info(f"Successfully loaded index from {self.index_dir}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")

    def clean_up(self):
        """Clean up the index and remove the index files."""
        logger.info("Starting cleanup process")
        try:
            # Clean up sparse index
            logger.info("Cleaning up sparse index")
            self.sparse_index.clear_index()
            self.sparse_index.close()
            if os.path.exists(self.sparse_index.db_path):
                os.remove(self.sparse_index.db_path)
                logger.info(f"Cleaned up sparse index {self.sparse_index.db_path}")
            
            # Clean up dense index
            dense_index_path = self.index_dir + "/" + self.dense_index_name 
            if os.path.exists(dense_index_path + ".faiss"):
                logger.info("Cleaning up dense index")
                os.remove(dense_index_path + ".faiss")
                os.remove(dense_index_path + ".pkl")
                logger.info(f"Cleaned up dense index {dense_index_path}")
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        



'''
class DenseFAISSIndex:
    """
    (UNTESTED) A FAISS index wrapper with more control over the langchain faiss interface
    """
    
    SUPPORTED_INDEX_TYPES = {
        "flat_l2": "IndexFlatL2",           # Exact L2 search (Euclidean distance)
        "flat_ip": "IndexFlatIP",           # Exact inner product search (Cosine similarity)  
        "hnsw": "IndexHNSWFlat",            # HNSW graph-based 
        "ivf_flat": "IndexIVFFlat",         # Inverted file + exact (k-nearest neighbors)
        "ivf_pq": "IndexIVFPQ",             # Inverted file + product quantization
        "pq": "IndexPQ",                    # Product quantization only
        "lsh": "IndexLSH"                   # Locality sensitive hashing
    }
    
    def __init__(self, 
                 embedding_dim: int,
                 index_type: str = "flat_l2",
                 index_params: Dict[str, Any] = None):
        """
        Initialize the dense FAISS index.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index to use
            index_params: Additional parameters for specific index types
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_params = index_params or {}
        self.index = None
        self.docstore = {}
        self.index_to_docstore_id = {}
        self._next_doc_id = 0
        
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(f"Unsupported index type: {index_type}. "
                           f"Supported types: {list(self.SUPPORTED_INDEX_TYPES.keys())}")
        
        self._create_index()
    
    def _create_index(self):
        """Create the FAISS index based on the specified type."""
        import faiss
        
        if self.index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        elif self.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
        elif self.index_type == "hnsw":
            M = self.index_params.get("M", 32)  # Number of connections
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            
            # configure hnsw parameters
            if "efConstruction" in self.index_params:
                self.index.hnsw.efConstruction = self.index_params["efConstruction"]
            if "efSearch" in self.index_params:
                self.index.hnsw.efSearch = self.index_params["efSearch"]
                
        elif self.index_type == "ivf_flat":
            nlist = self.index_params.get("nlist", 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
        elif self.index_type == "ivf_pq":
            nlist = self.index_params.get("nlist", 100)
            m = self.index_params.get("m", 8)  # Number of subquantizers
            nbits = self.index_params.get("nbits", 8)  # Bits per subquantizer
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
            
        elif self.index_type == "pq":
            m = self.index_params.get("m", 8)
            nbits = self.index_params.get("nbits", 8)
            self.index = faiss.IndexPQ(self.embedding_dim, m, nbits)
            
        elif self.index_type == "lsh":
            nbits = self.index_params.get("nbits", self.embedding_dim * 4)
            self.index = faiss.IndexLSH(self.embedding_dim, nbits)
    
    def train(self, embeddings: np.ndarray):
        """Train the index if required (for IVF and PQ indexes)."""
        if hasattr(self.index, 'train'):
            print(f"Training {self.index_type} index with {len(embeddings)} vectors...")
            self.index.train(embeddings.astype(np.float32))
            print("Training completed.")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Add documents and their embeddings to the index.
        
        Args:
            embeddings: Array of embeddings (n_docs, embedding_dim)
            documents: List of document dictionaries
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Train index if necessary
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train(embeddings)
        
        # Add embeddings to index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Store documents in docstore with proper mapping
        for i, doc in enumerate(documents):
            doc_id = f"doc_{self._next_doc_id}"
            self.docstore[doc_id] = doc
            self.index_to_docstore_id[start_id + i] = doc_id
            self._next_doc_id += 1
        
        print(f"Added {len(documents)} documents to {self.index_type} index. "
              f"Total documents: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               nprobe: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding (1, embedding_dim) or (embedding_dim,)
            k: Number of results to return
            nprobe: Number of clusters to search (for IVF indexes)
            
        Returns:
            List of search results with documents and scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Set nprobe for IVF indexes
        if nprobe and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve documents
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                doc_id = self.index_to_docstore_id.get(idx)
                if doc_id:
                    doc = self.docstore[doc_id].copy()
                    doc['score'] = float(distance)
                    doc['index_id'] = int(idx)
                    results.append(doc)
        
        return results
    
    def save_to_disk(self, save_path: str):
        """
        Save the index and metadata to disk.
        
        Args:
            save_path: Directory path to save the index
        """
        import pickle
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(save_path, "faiss.index")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'index_params': self.index_params,
            'docstore': self.docstore,
            'index_to_docstore_id': self.index_to_docstore_id,
            'next_doc_id': self._next_doc_id
        }
        
        metadata_file = os.path.join(save_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {save_path}")
    
    @classmethod
    def load_from_disk(cls, load_path: str):
        """
        Load the index and metadata from disk.
        
        Args:
            load_path: Directory path containing the saved index
            
        Returns:
            DenseFAISSIndex instance
        """
        import pickle
        
        # Load metadata
        metadata_file = os.path.join(load_path, "metadata.pkl")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_dim=metadata['embedding_dim'],
            index_type=metadata['index_type'],
            index_params=metadata['index_params']
        )
        
        # Load FAISS index
        index_file = os.path.join(load_path, "faiss.index")
        instance.index = faiss.read_index(index_file)
        
        # Restore metadata
        instance.docstore = metadata['docstore']
        instance.index_to_docstore_id = metadata['index_to_docstore_id']
        instance._next_doc_id = metadata['next_doc_id']
        
        print(f"Index loaded from {load_path}")
        return instance
    
    def get_index_info(self):
        """Get information about the current index."""
        info = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'total_documents': self.index.ntotal if self.index else 0,
            'is_trained': getattr(self.index, 'is_trained', True),
            'index_params': self.index_params
        }
        
        if hasattr(self.index, 'nprobe'):
            info['nprobe'] = self.index.nprobe
            
        return info
'''
