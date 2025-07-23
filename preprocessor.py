from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import TesseractBlobParser
import re, sys
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    A class that preprocesses input documents by:
        - Extracting text from URLs and PDFs via 
            - BeautifulSoup 
            - Text extraction with PyPDF2
            - OCR 
        - Splitting the text into chunks for indexing.
    """
    def __init__(self, options: dict = None):
        """
        Initialize the Preprocessor with optional configuration settings.
        
        Args:
            options (dict, optional): Configuration dictionary for preprocessing settings.
                                    Defaults to empty dict if no options are provided.
        """
        self.options = options or {}
        self.chunk_size = self.options.get("chunk_size", 1500)
        self.chunk_overlap = self.options.get("chunk_overlap", 200)
        self.clean_text = self.options.get("clean_text", True)
        logger.info(f"Preprocessor initialized with options: {self.options}")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")

    def extract_text_from_unstructured_pdf(self, pdf_path: str) -> List[Document]:
        """
        *requires more evals and parameter tuning
        Extract text from PDF using UnstructuredPDFLoader with chunk overlap."""
        
        try:
            loader = UnstructuredPDFLoader(
                pdf_path,
                mode="elements", 
                strategy="ocr_only", 
                languages=["ara", "eng"],
                max_characters=3000,  
                new_after_n_chars=3000,
                combine_text_under_n_chars=3000,
                infer_table_structure=False, 
                extract_images=False,
                include_orig_elements=False,
            )
            
            logger.info("UnstructuredPDFLoader configured with OCR for ara+eng")
            
            documents = loader.load()
            logger.info(f"Successfully extracted {len(documents)} initial document chunks from {pdf_path}")
            
            # Add chunk overlap using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,  
                chunk_overlap=self.chunk_overlap,  
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""]  
            )
            
            logger.info("Applying chunk overlap with RecursiveCharacterTextSplitter...")
            final_documents = text_splitter.split_documents(documents)
            logger.info(f"Successfully created {len(final_documents)} chunks with overlap from {pdf_path}")
            
            return final_documents
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            raise

    def extract_text_from_url(self, url: str) -> List[Document]:
        """
        Extract text content from web URLs using WebBaseLoader and split into chunks.
        
        Args:
            url (str): The URL to extract text content from
            
        Returns:
            List[Document]: List of Document objects containing chunked web content
        """
        logger.info(f"Starting URL extraction for: {url}")  
        logger.info(f"Chunking parameters - size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        
        try:
            # Load the document from URL
            loader = WebBaseLoader(
                web_paths=[url],
                requests_per_second=2,
                requests_kwargs={
                    'timeout': 30,
                    'verify': True,
                }
            )
            
            logger.info("WebBaseLoader configured, loading document...")
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} raw documents from URL")
            
            if self.clean_text:
                logger.info("Applying text cleaning transformations")
                for doc in docs:
                    original_length = len(doc.page_content)
                    doc.page_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', doc.page_content) 
                    doc.page_content = re.sub(r'[ \t]+', ' ', doc.page_content)  
                    doc.page_content = re.sub(r'\s+', ' ', doc.page_content)  
                    # Note: Removing this line as it removes all Arabic text
                    # doc.page_content = re.sub(r'[^a-zA-Z0-9\s]', '', doc.page_content)  
                    doc.page_content = doc.page_content.strip()  
                    cleaned_length = len(doc.page_content)
                    logger.debug(f"Text cleaned: {original_length} -> {cleaned_length} characters")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""]  
            )
            
            logger.info("Splitting documents into chunks...")
            chunked_documents = text_splitter.split_documents(docs)
            logger.info(f"Successfully created {len(chunked_documents)} chunks from URL content")
            
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Failed to extract text from URL {url}: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Extract text, images, and tables from PDF using PyMuPDFLoader with advanced processing and chunking.
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            List[Document]: List of Document objects containing extracted content with images and tables,
                           split into chunks for pages with substantial text content
        """
        logger.info(f"Starting PyMuPDF extraction for: {pdf_path}")
        logger.info(f"Chunking parameters - size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        
        try:
            loader = PyMuPDFLoader(
                pdf_path,
                extract_images=True,
                mode="page",
                images_inner_format="markdown-img",
                images_parser=TesseractBlobParser(),
                extract_tables="markdown",
            )
            
            logger.info("PyMuPDFLoader configured with image and table extraction enabled")
            
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} pages from PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""]  
            )
            
            logger.info("Splitting documents into chunks...")
            chunked_documents = text_splitter.split_documents(docs)
            logger.info(f"Successfully created {len(chunked_documents)} chunks from {len(docs)} pages")
            
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            raise


