from rag import RAG
from tests.utils import english_files, rag_answer_with_context


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
    

