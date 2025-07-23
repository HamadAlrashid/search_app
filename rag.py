import os
import logging
import sys
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from preprocessor import Preprocessor
from indexer import Indexer
from retriever import Retriever
from prompts import RAG_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)

SUPPORTED_LLM_MODELS = {
    "gemini-1.5-pro": ChatGoogleGenerativeAI,
    "gemini-1.5-flash": ChatGoogleGenerativeAI,
    "gpt-4o": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "gpt-3.5-turbo": ChatOpenAI,
}

class RAG:
    """
    A comprehensive RAG (Retrieval-Augmented Generation) system that connects
    document preprocessing, indexing, retrieval, and language model generation.
    
    This class provides end-to-end functionality for:
    - Document ingestion from URLs and PDFs
    - Index creation and management (both sparse and dense)
    - Hybrid retrieval using BM25 and vector search
    - Answer generation using various LLM providers
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        index_dir: str = "indexes",
        sparse_index: str = "bm25",
        dense_index_type: str = "flat_l2",
        dense_index_params: Dict[str, Any] = None,
        retrieval_method: str = "rrf",
        sparse_weight: float = 0.5,
        k: int = 10,
        preprocessor_options: Dict[str, Any] = None,
        llm_params: Dict[str, Any] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            llm_model: Name of the LLM model to use
            embedding_model: Name of the embedding model for dense retrieval
            index_dir: Directory to store indexes
            sparse_index: Type of sparse index (default: "bm25")
            dense_index_type: Type of dense FAISS index (flat index is recommended unless the corpus size > 1M documents)
            dense_index_params: Parameters for dense index
            retrieval_method: Method for combining retrievers and reranking ("rrf" (reciprocal rank fusion), "interleaved", "cross-encoder", "multi-query", "hyde")
            sparse_weight: Weight for sparse retrieval in ensemble (0.0-1.0). If you want to use only dense retrieval, set to 0.0.
            k: Number of documents to retrieve
            preprocessor_options: Options for document preprocessing (see preprocessor.py)
            llm_params: Additional parameters for LLM initialization 
        """
        logger.info(f"Initializing RAG system with LLM: {llm_model}, Embedding: {embedding_model}")
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.k = k
        self.retrieval_method = retrieval_method
        self.sparse_weight = sparse_weight
        
        # Initialize components
        self._initialize_preprocessor(preprocessor_options or {})
        self._initialize_indexer(
            index_dir, sparse_index, dense_index_type, 
            dense_index_params or {}, embedding_model
        )
        self._initialize_retriever()
        self._initialize_llm(llm_params or {})
        self._initialize_prompt()
        
        logger.info("RAG system initialization completed successfully")
    
    def _initialize_preprocessor(self, options: Dict[str, Any]):
        """Initialize the document preprocessor."""
        logger.info("Initializing preprocessor")
        self.preprocessor = Preprocessor(options)
    
    def _initialize_indexer(
        self, 
        index_dir: str, 
        sparse_index: str, 
        dense_index_type: str,
        dense_index_params: Dict[str, Any],
        embedding_model: str
    ):
        """Initialize the indexer for sparse and dense retrieval."""
        logger.info("Initializing indexer")
        self.indexer = Indexer(
            index_dir=index_dir,
            sparse_index=sparse_index,
            dense_index_type=dense_index_type,
            dense_index_params=dense_index_params,
            embedding_model=embedding_model
        )
    
    def _initialize_retriever(self):
        """Initialize the hybrid retriever."""
        logger.info("Initializing retriever")
        self.retriever = Retriever(
            indexer=self.indexer,
            merger_method=self.retrieval_method,
            sparse_weight=self.sparse_weight,
            k=self.k
        )
    
    def _initialize_llm(self, llm_params: Dict[str, Any]):
        """Initialize the language model."""
        if self.llm_model not in SUPPORTED_LLM_MODELS:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}. "
                           f"Supported models: {list(SUPPORTED_LLM_MODELS.keys())}")
        
        logger.info(f"Initializing LLM: {self.llm_model}")
        
        # Set default parameters
        default_params = {
            "model": self.llm_model,
            "temperature": 0.1,
            "max_tokens": 2048
        }
        default_params.update(llm_params)
        
        llm_class = SUPPORTED_LLM_MODELS[self.llm_model]
        self.llm: BaseChatModel = llm_class(**default_params)
        
        logger.info(f"Successfully initialized LLM ({self.llm_model}) with parameters: {default_params}")
    
    def _initialize_prompt(self):
        """Initialize the RAG prompt template."""
        self.prompt_template = PromptTemplate(
            template=RAG_PROMPT,
            input_variables=["query", "context"]
        )
    
    def add_documents_from_urls(self, urls: List[str]) -> bool:
        """
        Add documents from a list of URLs to the index.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Processing {len(urls)} URLs")
        
        try:
            all_documents = []
            for url in urls:
                logger.info(f"Processing URL: {url}")
                documents = self.preprocessor.extract_text_from_url(url)
                """
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = url
                    doc.metadata["source_type"] = "url"
                """
                all_documents.extend(documents)
            
            # Add to indexes
            success = self.indexer.add_documents(all_documents)
            if success:
                logger.info(f"Successfully indexed {len(all_documents)} documents from URLs")
            else:
                logger.error("Failed to index documents from URLs")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing URLs: {e}")
            return False
    
    def add_documents_from_pdfs(self, pdf_paths: List[str], use_ocr: bool = False) -> bool:
        """
        Add documents from a list of PDF files to the index.
        
        Args:
            pdf_paths: List of PDF file paths to process
            use_ocr: Whether to use OCR for text extraction (false is recommended)
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Processing {len(pdf_paths)} PDF files (OCR: {use_ocr})")
        
        try:
            all_documents = []
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                
                logger.info(f"Processing PDF: {pdf_path}")
                
                
                if use_ocr:
                    documents = self.preprocessor.extract_text_from_unstructured_pdf(pdf_path)
                else:
                    documents = self.preprocessor.extract_text_from_pdf(pdf_path)
                
                
                all_documents.extend(documents)
                logger.info(f"Extracted {len(documents)} chunks from {pdf_path}")
            
            # Add to indexes
            success = self.indexer.add_documents(all_documents)
            if success:
                logger.info(f"Successfully indexed {len(all_documents)} documents from PDFs")
            else:
                logger.error("Failed to index documents from PDFs")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add pre-processed documents to the index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Adding {len(documents)} pre-processed documents to index")
        
        try:
            success = self.indexer.add_documents(documents)
            if success:
                logger.info(f"Successfully indexed {len(documents)} documents")
            else:
                logger.error("Failed to index documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        try:
            documents = self.retriever.search(query)
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM with the provided context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        logger.info("Generating answer with LLM")
        
        try:
            # Format the prompt
            formatted_prompt = self.prompt_template.format(
                query=query,
                context=context
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
    
    def query(self, query: str, return_context: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Main query method that performs end-to-end RAG.
        
        Args:
            query: User query
            return_context: Whether to return context along with answer
            
        Returns:
            Generated answer or dict with answer and context (for debugging)
        """
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Retrieve relevant documents
            documents = self.retrieve_documents(query)
            
            # Format context
            context = self.format_context(documents)
            
            # Generate answer
            answer = self.generate_answer(query, context)
            
            if return_context:
                return {
                    "answer": answer,
                    "context": context,
                    "documents": documents,
                    "num_documents": len(documents)
                }
            else:
                return answer
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = f"Sorry, I encountered an error while processing your query: {str(e)}"
            
            if return_context:
                return {
                    "answer": error_msg,
                    "context": "",
                    "documents": [],
                    "num_documents": 0
                }
            else:
                return error_msg
    
    def save_index(self):
        """Save the current index to disk."""
        logger.info("Saving index to disk")
        try:
            self.indexer.save_index()
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self):
        """Load index from disk."""
        logger.info("Loading index from disk")
        try:
            self.indexer.load_index()
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            sparse_doc_count = self.indexer.sparse_index.get_document_count()
            dense_doc_count = len(self.indexer.dense_index.index_to_docstore_id) if hasattr(self.indexer.dense_index, 'index_to_docstore_id') else 0
            
            return {
                "sparse_documents": sparse_doc_count,
                "dense_documents": dense_doc_count,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "retrieval_method": self.retrieval_method,
                "sparse_weight": self.sparse_weight,
                "k": self.k
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def cleanup(self):
        """Clean up and remove index files."""
        logger.info("Cleaning up RAG system")
        try:
            self.indexer.clean_up()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
