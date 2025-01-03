import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA


def get_source_docs():
    """Reads the documents from the knowledge base and returns a list of Document objects."""
    source_docs = []
    for doc_path in os.listdir("knowledge_base"):
        with open(f"knowledge_base/{doc_path}", "r") as f:
            source_docs.append(Document(page_content=f.read(), metadata={"source": doc_path}))
    return source_docs


def text_splitter(source_docs):
    """Splits documents into smaller chunks using a recursive text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = text_splitter.split_documents(source_docs)
    return docs_processed


def create_bm25_retriever(docs_processed):
    """Creates and returns a BM25Retriever instance."""
    retriever = BM25Retriever.from_documents(docs_processed)
    return retriever


def create_retriever_qa_chain(retriever):
    """Creates a RetrievalQA chain using the BM25 retriever and the Llama model."""
    # Initialize the updated Ollama Llama3.2:1b model
    llm = OllamaLLM(model="llama3.2:1b")

    # Create the RetrievalQA chain
    retriever_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever
    )
    return retriever_qa_chain


def main():
    """Main function to run the RAG pipeline."""
    # Step 1: Get and process the source documents
    source_docs = get_source_docs()
    docs_processed = text_splitter(source_docs)

    # Step 2: Create the BM25Retriever
    retriever = create_bm25_retriever(docs_processed)

    # Step 3: Create the RetrievalQA chain
    retriever_qa_chain = create_retriever_qa_chain(retriever)

    # Step 4: Query handling
    query = input("Enter your query: ")

    # Filter retrieved documents to avoid exceeding the token limit
    retrieved_docs = retriever.get_relevant_documents(query)
    combined_text = " ".join([doc.page_content for doc in retrieved_docs[:1024]])

    if len(combined_text.split()) > 1024:
        combined_text = " ".join(combined_text.split()[:1024])  # Truncate

    # Run the query
    result = retriever_qa_chain.invoke({"query": query})
    print("Answer:", result)


if __name__ == "__main__":
    main()
