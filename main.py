import os

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, LiteLLMModel, CodeAgent


def get_source_docs():
    source_docs = []
    for doc_path in os.listdir("knowledge_base"):
        with open(f"knowledge_base/{doc_path}", "r") as f:
            source_docs.append(Document(page_content=f.read(), metadata={"source": doc_path}))

    return source_docs


def text_splitter(source_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = text_splitter.split_documents(source_docs)

    return docs_processed


class RetrieverTool(Tool):
    name = "retriever"
    description = "This tool retrieves documents relevant to the prompt from a knowledge base using a BM25 retriever. Input the concept you want to search for as a string"
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


def main():
    source_docs = get_source_docs()

    docs_processed = text_splitter(source_docs)

    retriever_tool = RetrieverTool(docs_processed)

    model = LiteLLMModel(
        model_id = "ollama_chat/llama3.2:1b",
    )

    agent = CodeAgent(
        tools=[retriever_tool], model=model, max_iterations=4, verbose=True
    )

    agent_output = agent.run("¿Qué endpoints hay disponibles para la tabla de SORTING TIME?")

    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    main()

    print(0)