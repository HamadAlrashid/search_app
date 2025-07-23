import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexer import FTS5Retriever
from utils import documents



def clean_up(retriever: FTS5Retriever):
    retriever.clear_index()
    retriever.close()
    os.remove(retriever.db_path)

def test_fts5_retriever_init_without_documents():
   
   
    retriever = FTS5Retriever()
    assert os.path.exists(retriever.db_path)
    assert retriever.conn is not None
    assert retriever.get_document_count() == 0
    assert retriever.get_all_documents() == []
    
    # clean
    clean_up(retriever)




def test_fts5_retriever_init_with_documents():
    if os.path.exists("test_fts5_retriever_init_with_documents.sqlite"):
        os.remove("test_fts5_retriever_init_with_documents.sqlite")
    retriever = FTS5Retriever(db_path="test_fts5_retriever_init_with_documents.sqlite", documents=documents)
    assert os.path.exists(retriever.db_path)
    assert retriever.conn is not None
    assert retriever.get_document_count() == 3
    assert retriever.get_all_documents() == documents
    
    # clean
    clean_up(retriever)



    
def test_fts5_retriever_search():
    if os.path.exists("test_search.sqlite"):
        os.remove("test_search.sqlite")
    retriever = FTS5Retriever(db_path="test_search.sqlite", documents=documents)
    results = retriever.search("fox")
    assert len(results) == 1
    assert results[0].page_content == "The quick brown fox jumps over the lazy dog."
    assert results[0].metadata["source"] == "animal_story"
    assert results[0].id == '1'

    clean_up(retriever)
    
    


def test_fts5_retriever_search_many_candidates():
    if os.path.exists("test_search_many_candidates.sqlite"):
        os.remove("test_search_many_candidates.sqlite")
    retriever = FTS5Retriever(db_path="test_search_many_candidates.sqlite", documents=documents)
    results = retriever.search("is")
    assert len(results) == 2
    assert documents[1] in results
    assert documents[2] in results
    assert documents[0] not in results
    
    clean_up(retriever)


def test_fts5_retriever_delete_document():
    if os.path.exists("test_delete_document.sqlite"):
        os.remove("test_delete_document.sqlite")
    retriever = FTS5Retriever(db_path="test_delete_document.sqlite", documents=documents)
    results = retriever.search("fox")
    assert len(results) == 1

    retriever.delete_document(results[0].id)
    results = retriever.search("fox")
    assert len(results) == 0

    clean_up(retriever)

