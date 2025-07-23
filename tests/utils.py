from langchain_core.documents import Document

documents = [
    Document(
        page_content="The quick brown fox jumps over the lazy dog.",
        metadata={"source": "animal_story"}, id=1
    ), 
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "tech_article"}, id=2
    ),
    Document(
        page_content="Mount Everest is the highest mountain in the world.",
        metadata={"source": "geography_fact"}, id=3
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