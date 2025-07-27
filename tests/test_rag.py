from rag import RAG
from tests.utils import english_files, rag_answer_with_context
from local_models import get_local_embeddings, get_local_llm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage


def test_rag_initialization():
    rag = RAG()
    rag.add_documents_from_pdfs(english_files)
    assert rag.indexer.sparse_index.get_document_count() > 0
    rag.save_index()
    
    

def test_rag_multi_query():
    rag = RAG(retrieval_method="multi_query")
    rag.load_index()
    query = "How does LLMs process unknown tokens?"
    results = rag.query(query, return_context=True)
    assert results is not None
    assert len(results) > 0
    rag_answer_with_context(results)


def test_rag_query_decomposition():
    rag = RAG(retrieval_method="query_decomposition")
    rag.load_index()
    query = "How does LLMs process unknown tokens?"
    results = rag.query(query, return_context=True)
    assert results is not None
    assert len(results) > 0
    rag_answer_with_context(results)
    

def test_standalone_local_models():
    """Demonstrate using local models standalone (without RAG)."""
    
    embeddings = get_local_embeddings("Qwen/Qwen3-Embedding-0.6B")

    test_texts = ["Economic growth", "Artificial intelligence", "Machine learning", "Finance"]
    embeddings_result = embeddings.embed_documents(test_texts)
    
    # similarity matrix
    embeddings_array = np.array(embeddings_result)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)
    
    
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i <= j:  # Only print upper triangle to avoid duplicates
                score = similarity_matrix[i][j]
                print(f"{text1} <-> {text2}: {score:.3f}")
    
    

    llm = get_local_llm("microsoft/phi-2")
    
    response = llm.invoke("What is Linux?")
    print(response.content)
    assert "linux" in response.content.lower()


def test_rag_with_local_models():
    
    
    rag = RAG(
        llm_model="local:microsoft/phi-2",  
        embedding_model="local:Qwen/Qwen3-Embedding-0.6B",    
        k=5,
        index_dir="local_indexes",
        preprocessor_options={
            "chunk_size": 300,
            "chunk_overlap": 50
        }
    )
    
    sample_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    rag.add_documents_from_urls(sample_urls)

        
    query = "What is supervised learning?" #input("Enter a query: ")
    answer = rag.query(query, return_context=True)
    print(f"Answer: {answer['answer']}")
    print(f"Context: {answer['context']}")  
            
    assert "supervised learning" in answer['answer'].lower()

    stats = rag.get_index_stats()
    print(f"\n📊 Index Stats: {stats}")
        
      
      