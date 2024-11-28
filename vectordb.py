import json
import os

from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def read_json_files(folder_path):
    """Reads all JSON files in a specified folder.

    Args:
        folder_path: The path to the folder containing the JSON files.

    Returns:
        A list of dictionaries, where each dictionary represents the data from a JSON file.
        Returns an empty list if the folder does not exist or if no JSON files are found.
    """

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    
    splitter = RecursiveJsonSplitter(max_chunk_size=2000)

    all_json_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            
                doc_name = json_data.pop("ชื่อเอกสาร")
                json_doc = splitter.create_documents(
                    texts=[json_data],
                    ensure_ascii=False,
                    metadatas=[{"doc_name": doc_name}]
                )

                all_json_docs += json_doc

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file '{filename}': {e}")
            except Exception as e:
                print(f"An error occurred while reading '{filename}': {e}")

    return all_json_docs

if __name__ == "__main__":
    # Example usage:
    folder_path = "json_files"  # Replace with the actual folder path
    all_json_docs = read_json_files(folder_path)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
    )

    db = Chroma.from_documents(
        collection_name="json",
        documents=all_json_docs,
        embedding=embeddings,
        persist_directory="vectordb/knowledge",
    )
