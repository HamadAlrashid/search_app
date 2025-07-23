#!/usr/bin/env python3
"""
Example usage of the RAG system.

This script demonstrates how to use the RAG class to:
1. Initialize the system with different configurations
2. Add documents from URLs and PDFs
3. Perform queries and get answers
4. Save and load indexes
"""

import os
from rag import RAG
from langchain_core.documents import Document
from tests.utils import english_files

def example_basic_usage():
    """Basic example using default settings."""
    print("=" * 50)
    print("BASIC RAG USAGE EXAMPLE")
    print("=" * 50)
    
    # Initialize RAG with default settings (OpenAI GPT-4o-mini)
    rag = RAG(
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
        k=5
    )
    
    # Add some example URLs
    urls = [
        "https://python.langchain.com/docs/introduction/",
        "https://docs.python.org/3/tutorial/introduction.html"
    ]

    print("Adding documents from URLs...")
    success = rag.add_documents_from_urls(urls)
    print("Adding documents from test documents...")
    success2 = rag.add_documents_from_pdfs(english_files)
    if success and success2:
        print("✅ Documents added successfully")
    else:
        print("❌ Failed to add documents")
        return
    
    # Save the index
    rag.save_index()
    
    # Perform some queries
    queries = [
        "What is LangChain?", # url1
        "How do I use Python for beginners?", # url2 
        "How was SBERT trained?",# doc1 in english_files
        "What is Mask Language Models?",# doc2 in english_files
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        answer = rag.query(query, return_context=True)
        print(f"💬 Answer: {answer['answer']}")
        print(f"📚 Context used: {answer['context'][:200]}...")
        print(f"📄 Number of documents retrieved: {answer['num_documents']}")
    
    # Get index statistics
    stats = rag.get_index_stats()
    print(f"\n📊 Index Stats: {stats}")
    
    # Cleanup
    rag.cleanup()
    print("\n🧹 Cleanup completed")

def example_advanced_configuration():
    """Advanced example with custom configurations."""
    print("\n" + "=" * 50)
    print("ADVANCED RAG CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    # Initialize RAG with advanced settings
    rag = RAG(
        llm_model="gpt-4o",  # Better model
        embedding_model="text-embedding-3-large",
        index_dir="advanced_indexes",
        dense_index_type="hnsw",  # Better for large datasets
        dense_index_params={
            "M": 32,
            "efConstruction": 200,
            "efSearch": 100
        },
        retrieval_method="rrf",  # Reciprocal Rank Fusion
        sparse_weight=0.6,  # Favor sparse retrieval slightly
        k=8,
        preprocessor_options={
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "clean_text": True
        },
        llm_params={
            "temperature": 0.2,
            "max_tokens": 1500
        }
    )
    
    # Add custom documents
    custom_docs = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation.",
            metadata={"source": "custom", "topic": "RAG"}
        ),
        Document(
            page_content="Vector databases store high-dimensional vectors and enable efficient similarity search using methods like cosine similarity.",
            metadata={"source": "custom", "topic": "vector_db"}
        )
    ]
    
    print("Adding custom documents...")
    rag.add_documents(custom_docs)
    
    # Example query with context return
    query = "How does RAG work with vector databases?"
    print(f"\n🔍 Query: {query}")
    
    result = rag.query(query, return_context=True)
    print(f"💬 Answer: {result['answer']}")
    print(f"📚 Context used: {result['context'][:200]}...")
    print(f"📄 Number of documents retrieved: {result['num_documents']}")
    
    # Cleanup
    rag.cleanup()

def example_google_gemini():
    """Example using Google Gemini models."""
    print("\n" + "=" * 50)
    print("GOOGLE GEMINI RAG EXAMPLE")
    print("=" * 50)
    
    # Initialize with Gemini
    rag = RAG(
        llm_model="gemini-1.5-flash",
        embedding_model="models/embedding-001",
        index_dir="gemini_indexes",
        k=6
    )
    
    # Add some example documents
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            metadata={"source": "ml_guide", "category": "AI"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"source": "dl_guide", "category": "AI"}
        )
    ]
    
    print("Adding sample documents...")
    rag.add_documents(sample_docs)
    
    # Query
    query = "What is the difference between machine learning and deep learning?"
    print(f"\n🔍 Query: {query}")
    answer = rag.query(query)
    print(f"💬 Answer: {answer}")
    
    # Cleanup
    rag.cleanup()



def example_retrieval_methods():
    """Example comparing different retrieval methods."""
    print("\n" + "=" * 50)
    print("RETRIEVAL METHODS COMPARISON")
    print("=" * 50)
    
    # Sample documents
    docs = [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability.", metadata={"source": "python_guide"}),
        Document(page_content="JavaScript is a versatile programming language primarily used for web development.", metadata={"source": "js_guide"}),
        Document(page_content="Machine learning algorithms can be used to make predictions from data.", metadata={"source": "ml_guide"}),
    ]
    
    methods = ["rrf", "sparse_only", "dense_only", "interleaved", "cross-encoder", "multi-query", "hyde"]
    query = "What programming language is good for beginners?"
    rag = RAG(
            k=3,
            index_dir=f"test_indexes_{method}"
        )
        
    rag.add_documents(docs)

    for method in methods:
        print(f"\n🔄 Testing retrieval method: {method}")
        rag.retrieval_method = method
        
        answer = rag.query(query)
        print(f"💬 Answer: {answer[:150]}...")
        
        rag.cleanup()

def main():
    """Run all examples."""
    print("🚀 RAG System Examples")
    print("This script demonstrates various ways to use the RAG system.")
    
    try:
        # Run basic example
        example_basic_usage()
        
        # Run advanced configuration example
        example_advanced_configuration()
        
        # Run Google Gemini example (if API key available)
        if os.getenv("GOOGLE_API_KEY"):
            example_google_gemini()
        else:
            print("\n⚠️  Skipping Gemini example (GOOGLE_API_KEY not set)")
        
        
        # Run retrieval methods comparison
        #example_retrieval_methods()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
       

if __name__ == "__main__":
    main() 