import os
import logging
import sys
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from preprocessor import Preprocessor
from indexer import Indexer
from retriever import Retriever
from prompts import RAG_PROMPT, MULTI_QUERY_PROMPT, QUERY_DECOMPOSITION_PROMPT, task_map
from structured_output.multiquery import MultiQuery
from local_models import LocalLLM, LocalEmbeddings, get_local_llm, get_local_embeddings

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
    "local:microsoft/DialoGPT-medium": LocalLLM,
    "local:distilgpt2": LocalLLM,
    "local:microsoft/phi-2": LocalLLM,
    "local:TinyLlama/TinyLlama-1.1B-Chat-v1.0": LocalLLM,
    "local:meta-llama/Llama-2-7b-chat-hf": LocalLLM,
    "local:mistralai/Mistral-7B-Instruct-v0.1": LocalLLM,
    "local:Qwen/Qwen1.5-7B-Chat": LocalLLM,
    "local:Qwen/Qwen2.5-1.5B-Instruct": LocalLLM,
    "local:Qwen/Qwen2.5-3B-Instruct": LocalLLM,
    "local:Qwen/Qwen2.5-7B-Instruct": LocalLLM,
    "local:Qwen/Qwen2.5-14B-Instruct": LocalLLM,
    "local:Qwen/Qwen2.5-32B-Instruct": LocalLLM,
    "local:Qwen/Qwen2.5-72B-Instruct": LocalLLM,
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
        
        llm_class = SUPPORTED_LLM_MODELS[self.llm_model]
        
        # Handle local models differently
        if self.llm_model.startswith("local:"):
            # Extract the actual model name (remove 'local:' prefix)
            actual_model_name = self.llm_model[6:]  # Remove 'local:' prefix
            
            # Set default parameters for local models
            default_params = {
                "model_name": actual_model_name,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            default_params.update(llm_params)
            
            # Use the helper function for better configuration
            self.llm: BaseChatModel = get_local_llm(**default_params)
            
        else:
            # Handle cloud-based models (OpenAI, Google)
            default_params = {
                "model": self.llm_model,
                "temperature": 0.1,
                "max_tokens": 2048
            }
            default_params.update(llm_params)
            
            self.llm: BaseChatModel = llm_class(**default_params)
        
        logger.info(f"Successfully initialized LLM ({self.llm_model}) with parameters: {default_params}")
    
    def _initialize_prompt(self):
        """Initialize the RAG prompt template."""
        self.prompt_template = PromptTemplate(
            template=RAG_PROMPT,
            input_variables=["query", "context"]
        )
    
    def _get_llm(self, task: str = "multi_query", model: str = None, **kwargs):
        """
        Create LLMs on demand for advanced retrieval tasks.
        
        Args:
            task: The task type (e.g., 'multi_query', 'hyde', 'decomposition')
            model: Model name to use (defaults based on task)
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Configured LLM instance
        """
        default_models = {
            "multi_query": "gpt-3.5-turbo",
            "query_decomposition": "gpt-3.5-turbo",
            "hyde": "gpt-3.5-turbo",
        }
        
        model_name = model or default_models.get(task, "gpt-3.5-turbo")
        
        default_params = {
            "multi_query": {"temperature": 0, "max_tokens": 500},
            "hyde": {"temperature": 0.3, "max_tokens": 1000}
        }
        
        # Merge default params with provided kwargs
        params = {**default_params.get(task, {"temperature": 0}), **kwargs}
        
        # Handle local models
        if model_name.startswith("local:"):
            actual_model_name = model_name[6:]  # Remove 'local:' prefix
            local_params = {
                "model_name": actual_model_name,
                **params
            }
            return get_local_llm(**local_params)
        else:
            return ChatOpenAI(model=model_name, **params)
    
    def _generate_queries(self, query: str, number_of_queries: int = 3, task: str = "multi_query") -> MultiQuery:
        """
        Generate multiple queries using LLM for multi-query retrieval or query decomposition.
        
        Args:
            query: Original user query
            number_of_queries: Number of alternative queries to generate
            task: Task type (multi_query or query_decomposition)
        Returns:
            MultiQuery object containing the generated queries
        """
        if task not in ["multi_query", "query_decomposition"]:
            logger.warning(f"Invalid task: {task}. Using default task: multi_query")
            task = "multi_query"
        
        logger.info(f"Generating {number_of_queries} queries for {task} retrieval")
        
        llm = self._get_llm(task)
        structured_llm = llm.with_structured_output(MultiQuery)
        prompt = ChatPromptTemplate.from_template(task_map[task]).partial(number_of_queries=number_of_queries)
        new_queries_chain = prompt | structured_llm
        
        generated_queries = new_queries_chain.invoke({"query": query})
        generated_queries.queries.append(query)  # Include original query
        
        logger.info(f"Generated queries: {generated_queries}")
        logger.info(f"Total queries for {task} search: {len(generated_queries)}")
        
        return generated_queries

    
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
        Retrieve relevant documents for a query using the configured retrieval method.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving documents for query: '{query}' using method: {self.retrieval_method}")
        
        try:
            if self.retrieval_method == "multi_query":
                logger.info("Using multi-query retrieval")
                generate_queries_partial = RunnableLambda(
                    lambda query: self._generate_queries(query, number_of_queries=3, task="multi_query")
                )
                
                chain = (generate_queries_partial | 
                         RunnableLambda(self.retriever.multi_search) | 
                         RunnableLambda(self.retriever.apply_reciprocal_rank_fusion))
                
                documents = chain.invoke(query)
                
                return documents

            else:
                # For basic methods: rrf, interleaved, sparse_only, dense_only
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
        
        return "\n\n".join(context_parts)+"\n\n"
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM with the provided context.
        If query decomposition is used, the context will be enriched with the Q&As of the sub-queries.
        
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
            
            # Enrich and Format context
            context = self.enrich_and_format_context(documents, query)
            
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

    def enrich_and_format_context(self, context: List[Document], query: str) -> str:
        """
        Format the context always and enrich it in case query decomposition is used as a retrieval method.
        The context will be enriched with the Q&As of the sub-queries.

        Args:
            context: Original context documents
            query: Original user query

        Returns:
            formatted string of the context to be used in the LLM prompt
        """
        formatted_context = self.format_context(context)
        
        if self.retrieval_method == "query_decomposition":
            logger.info("Enriching context with Q&As of the sub-queries")
            
            decomposed_queries = self._generate_queries(query, number_of_queries=3, task="query_decomposition")
            
            llm = self._get_llm("query_decomposition")
            
            # Create a chain for sub-query answering
            def search_and_preserve_query(sub_query: str):
                documents = self.retriever.search(sub_query)
                return {"query": sub_query, "documents": documents}
            
            def format_for_prompt(data: Dict[str, Any]):
                return {
                    "query": data["query"],
                    "context": self.format_context(data["documents"])
                }
            
            sub_query_chain = (
                RunnableLambda(search_and_preserve_query) |
                RunnableLambda(format_for_prompt) |
                self.prompt_template |
                llm
            )
            
            # Answer each sub-query and append to context
            for sub_query in decomposed_queries.queries:
                try:
                    response = sub_query_chain.invoke(sub_query)
                    sub_answer = response.content if hasattr(response, 'content') else str(response)
                    formatted_context += f"Sub-query: {sub_query}\nAnswer: {sub_answer}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to answer sub-query '{sub_query}': {e}")
            
        return formatted_context
    
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
