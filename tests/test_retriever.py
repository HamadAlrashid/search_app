from retriever import Retriever
from indexer import Indexer
from utils import documents, english_files, pretty_print_documents
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
    
    
    
    