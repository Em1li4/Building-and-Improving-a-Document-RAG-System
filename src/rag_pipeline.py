import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: List[Any]) -> str:
    """Format retrieved document chunks into a single text block separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


DEFAULT_RAG_PROMPT = """You are a helpful and knowledgeable customer support assistant for Marlowe & Finch, a lightweight outdoor gear company.

Answer the customer's question using ONLY the retrieved context below. If the context does not contain enough information to answer the question, state clearly that you cannot find that information in the knowledge base and advise the customer to contact support directly at support@marloweandfinch.com.

Do not make up information or use knowledge outside the provided context. Maintain a friendly, helpful, and professional tone similar to our customer support team.

Context:
{context}

Question: {question}

Answer:"""


DEFAULT_REWRITE_PROMPT = """You are an AI assistant helping to improve customer support query retrieval for an outdoor gear company called Marlowe & Finch.

Rewrite the following short or vague customer query into a single, detailed, and specific search query that will be used to retrieve relevant information from a technical product and policy knowledge base.

Maintain the original customer intent. Do not generate multiple options or bullet points. Output EXACTLY ONE rewritten query text.

Short Customer Query: {short_query}

Rewritten Query:"""


class DocumentRAGPipeline:
    """
    A modular Document RAG System for Marlowe & Finch customer support.
    Supports document loading, chunking, vector storage, baseline RAG, and query rewriting.
    """

    def __init__(
        self,
        knowledge_base_path: str = "data/marlowe_knowledge_base.txt",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o",
        k_neighbors: int = 4,
        openai_api_key: Optional[str] = None,
    ):
        self.kb_path = knowledge_base_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.k_neighbors = k_neighbors
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(api_key=self.api_key)
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None

    def load_and_chunk(self):
        """Loads knowledge base text file and splits into chunks."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")

        loader = TextLoader(self.kb_path)
        self.documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = splitter.split_documents(self.documents)
        print(f"[RAGPipeline] Loaded {len(self.documents)} doc(s). Created {len(self.chunks)} chunks.")
        return self.chunks

    def build_vector_index(self):
        """Initializes OpenAI Embeddings and builds Chroma vector index."""
        if not self.chunks:
            self.load_and_chunk()

        embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.api_key
        )
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k_neighbors}
        )
        print(f"[RAGPipeline] Chroma vector index constructed with k={self.k_neighbors}.")

    def assemble_chain(self, prompt_template: str = DEFAULT_RAG_PROMPT):
        """Assembles the LangChain RAG pipeline chain."""
        if not self.retriever:
            self.build_vector_index()

        self.llm = ChatOpenAI(
            model=self.llm_model_name,
            openai_api_key=self.api_key
        )
        prompt = PromptTemplate.from_template(prompt_template)

        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("[RAGPipeline] RAG chain successfully assembled.")

    def rewrite_query(self, query: str, prompt_template: str = DEFAULT_REWRITE_PROMPT) -> str:
        """Expands/rewrites a short customer query for better semantic retrieval."""
        formatted_prompt = prompt_template.format(short_query=query)
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten

    def query(self, query_text: str, use_query_rewriting: bool = False) -> Dict[str, Any]:
        """
        Executes a customer query through the RAG pipeline.
        Returns dictionary with query, rewritten_query (if used), and generated answer.
        """
        if not self.rag_chain:
            self.assemble_chain()

        final_query = query_text
        rewritten_text = None

        if use_query_rewriting:
            rewritten_text = self.rewrite_query(query_text)
            final_query = rewritten_text

        answer = self.rag_chain.invoke(final_query)

        return {
            "original_query": query_text,
            "rewritten_query": rewritten_text,
            "used_query": final_query,
            "answer": answer
        }


if __name__ == "__main__":
    # Simple verification run
    print("Initializing Document RAG Pipeline...")
    pipeline = DocumentRAGPipeline()
    print("Pipeline ready.")
