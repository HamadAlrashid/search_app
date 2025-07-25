from retriever import Retriever
from indexer import Indexer
from tests.utils import documents, english_files, pretty_print_documents
from preprocessor import Preprocessor

    
def test_retriever_basic():
    preprocessor = Preprocessor()

    for file in english_files:
        documents.extend(preprocessor.extract_text_from_pdf(file))
        
    indexer = Indexer()
    indexer.add_documents(documents)

    retriever = Retriever(indexer)

    results = retriever.search("What is SBERT?")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    indexer.save_index()
    
# need to run the basic test first to save the index
def test_retriever_interleaved():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="interleaved")
    results = retriever.search("What is SBERT?")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    
    
    
def test_retriever_multi_query():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="multi_query")
    query = "What is SBERT?"
    queries = retriever._generate_multi_query(query, number_of_queries=3)
    assert len(queries) == 4
    assert query in queries.queries
    print("generated queries:")
    print(queries)
    
def test_retriever_multi_query_search():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="multi_query")
    query = "What is SBERT?"
    results = retriever.search(query)
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    