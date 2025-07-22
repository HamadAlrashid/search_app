import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.documents import Document
from indexer import Indexer
from tests.test_fts5 import documents

# Add this logging setup for IDE testing
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,  # Override existing config
    stream=sys.stdout
)

def test_indexer_init():
    indexer = Indexer()
    assert indexer.sparse_index is not None
    assert indexer.embedding_model is not None
    assert indexer.embedding_dim is not None
    assert indexer.dense_index_type is not None
    assert indexer.dense_index is not None
    assert indexer.dense_index_params is not None

    indexer.clean_up()

def test_indexer_add_documents_to_sparse_index():
    indexer = Indexer()
    indexer.add_documents_to_sparse_index(documents)
    assert indexer.sparse_index.get_document_count() == len(documents)
    assert indexer.sparse_index.search("fox").pop() in documents
    indexer.clean_up()

def test_indexer_add_documents_to_dense_index():
    indexer = Indexer()
    indexer.add_documents_to_dense_index(documents)
    results = indexer.search_dense("Tallest point on Earth!!")
    assert len(results) == 3
    assert "Everest" in results[0].page_content
    indexer.clean_up()


def test_indexer_save_and_load_index():
    
    index_dir = "indexes_test"
    

    indexer = Indexer(index_dir=index_dir)
    indexer.add_documents(documents)
    indexer.save_index()
    assert os.path.exists(indexer.index_dir + "/" + indexer.dense_index_name + ".faiss")
    # close the sqlite connection
    indexer.sparse_index.close()

    indexer2 = Indexer(index_dir=index_dir)
    indexer2.load_index()

    assert indexer2.sparse_index.get_document_count() == len(documents)
    assert indexer2.dense_index.index_to_docstore_id == indexer.dense_index.index_to_docstore_id
    indexer2.clean_up()


def test_indexer_load_index_and_clean():
    index_dir = "indexes_test"
    indexer = Indexer(index_dir=index_dir)
    indexer.load_index()
    indexer.clean_up()
    assert not os.path.exists(index_dir + "/" + indexer.dense_index_name + ".faiss")
   