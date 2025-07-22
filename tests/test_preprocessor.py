from preprocessor import Preprocessor



def test_preprocessor_extract_text_from_url():
    print("test_preprocessor_extract_text_from_url")
    preprocessor = Preprocessor()
    docs = preprocessor.extract_text_from_url("https://waitbutwhy.com/2014/01/your-family-past-present-and-future.html")
    assert len(docs) > 0


def test_preprocessor_extract_text_from_pdf_arabic():
    print("test_preprocessor_extract_text_from_pdf_arabic")
    preprocessor = Preprocessor()
    docs = preprocessor.extract_text_from_pdf("tests/arabic_document.pdf")
    keyword = "ﻧﻈﺎم"
    assert (keyword in docs[0].page_content or 
    keyword in docs[1].page_content or
     keyword in docs[2].page_content or
     keyword in docs[3].page_content or
     keyword in docs[4].page_content or
     keyword in docs[5].page_content or
     keyword in docs[6].page_content or
     keyword in docs[7].page_content or
     keyword in docs[8].page_content or
     keyword in docs[9].page_content)
    for i, doc in enumerate(docs):
        print(f"-----------[{i}]------------------")
        print(doc.page_content)
        print("--------------------------------")


def test_preprocessor_extract_text_from_pdf_english():
    print("test_preprocessor_extract_text_from_pdf_english")
    preprocessor = Preprocessor()
    docs = preprocessor.extract_text_from_pdf("tests/llm_study_english.pdf")
    assert len(docs) > 0
    for i, doc in enumerate(docs):
        print(f"-----------[{i}]------------------")
        print(doc.page_content)
        print("--------------------------------")


def test_preprocessor_extract_text_from_unstructured_pdf():
    print("test_preprocessor_extract_text_from_unstructured_pdf")
    preprocessor = Preprocessor()
    docs = preprocessor.extract_text_from_unstructured_pdf("tests/embedding_models_notes_english.pdf")
    assert len(docs) > 0
    for i, doc in enumerate(docs):
        print(f"-----------[{i}]------------------")
        print(doc.page_content)
        print("--------------------------------")

