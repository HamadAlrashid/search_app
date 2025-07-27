from typing import Any, Dict
from langchain_core.documents import Document

documents = [
    Document(
        page_content="The quick brown fox jumps over the lazy dog.",
        metadata={"source": "animal_story"}
    ), 
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "tech_article"}
    ),
    Document(
        page_content="Mount Everest is the highest mountain in the world.",
        metadata={"source": "geography_fact"}
    )
]


english_files = ["tests/llm_study_english.pdf", "tests/embedding_models_notes_english.pdf"]


def pretty_print_documents(documents: list[Document]):
    for i, doc in enumerate(documents):
        print(f"-----------[{i}]----------")
        print(doc.page_content)
        print(doc.metadata)
        print("--------------------------------")


def pretty_save_results(results: list[Document], file_name: str):
    with open(file_name, "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            f.write(f"-----------[{i}]----------\n")
            f.write(result.page_content)
            f.write(f"Metadata: {result.metadata}\n")
            f.write("-------------------------------\n")


def rag_answer_with_context(answer : Dict[str, Any]):
    print(f"💬 Answer: {answer['answer']}\n\n")
    print(f"📚 Context used: {answer['context']}\n\n")
    print(f"📄 Number of documents retrieved: {answer['num_documents']}")
    