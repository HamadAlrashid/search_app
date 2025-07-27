from retriever import Retriever
from indexer import Indexer
from tests.utils import documents, english_files, pretty_print_documents
from preprocessor import Preprocessor


# run this first so that the index is saved for other tests
def test_retriever_init():
    preprocessor = Preprocessor()

    for file in english_files:
        documents.extend(preprocessor.extract_text_from_pdf(file))
        
    indexer = Indexer()
    indexer.add_documents(documents)
    indexer.save_index()
    assert indexer.sparse_index is not None
    assert indexer.dense_index is not None
    

def test_retriever_rrf():    
    indexer = Indexer()
    indexer.load_index()
    

    retriever = Retriever(indexer)

    results = retriever.search("What is SBERT?")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    indexer.save_index()
    
def test_retriever_interleaved():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="interleaved")
    results = retriever.search("What is SBERT?")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    
    
    
def test_retriever_sparse_only():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="sparse_only")
    results = retriever.search("SBERT")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    
def test_retriever_dense_only():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer, merger_method="dense_only")
    results = retriever.search("What is SBERT?")
    assert "SBERT" in results[0].page_content 
    pretty_print_documents(results)
    


def test_retriever_multi_search_with_rrf():
    indexer = Indexer()
    indexer.load_index()
    retriever = Retriever(indexer)
    query = ["What is temperature in LLMs?", "What are autoregressive models?"]
    results = retriever.multi_search(query)
    assert len(results) == 4
    assert "temperature" in results[0][0].page_content or "temperature" in results[1][0].page_content
    assert "autoregressive" in results[2][0].page_content or "autoregressive" in results[3][0].page_content
    print("Results:")
    

    rrf_results = retriever.apply_reciprocal_rank_fusion(results)
    assert sum([len(doc_list) for doc_list in results]) >= len(rrf_results)
    
    
    
    
    
    
    
    
    