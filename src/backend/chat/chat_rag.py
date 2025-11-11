from langchain_core.runnables import (
    RunnableSequence, RunnablePassthrough, RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI
from backend.retriever.retriever import Retriever
from backend.config import LLM_CONFIG, RETRIEVER_CONFIG


class ChatRAG:
    """
    The ChatRAG orchestrates the retrieval-augmented generation (RAG) flow
    using a retriever and a modern LangChain RunnableSequence pipeline.
    """
    def __init__(self, model_name: str = None, temperature: float = None, top_k: int = None, verbose: bool = True):
        self.verbose = verbose
        self.top_k = top_k or RETRIEVER_CONFIG["top_k"]
        self.model_name = model_name or LLM_CONFIG["model"]
        self.temperature = temperature or LLM_CONFIG["temperature"]
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=LLM_CONFIG["api_key"],
        )

        self.retriever = Retriever(top_k=self.top_k).load()
        self.chain = self._build_chain()

    def _combine_docs(self, docs: list[Document]) -> str:
        """Concatenate retrieved documents into a single context block."""
        if self.verbose:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata

                source_id = metadata.get("source_id", "N/A")
                title = metadata.get("Title", "N/A")
                release_year = metadata.get("Release Year", "N/A")
                wiki_page = metadata.get("Wiki Page", "N/A")
                origin = metadata.get("Origin/Ethnicity", "N/A")
                director = metadata.get("Director", "N/A")
                cast = metadata.get("Cast", "N/A")
                genre = metadata.get("Genre", "N/A")
               
                print(f"\nDoc {i}")
                print(f"  • Source ID: {source_id}")
                print(f"  • Title: {title}")
                print(f"  • Release Year: {release_year}")
                print(f"  • Wiki Page: {wiki_page}")
                print(f"  • Origin/Ethnicity: {origin}")
                print(f"  • Director: {director}")
                print(f"  • Cast: {cast}")
                print(f"  • Genre: {genre}")
                print(f"  • Content Preview: {doc.page_content[:300]}...\n")

        return "\n\n".join([doc.page_content for doc in docs])
    

    def _build_chain(self) -> RunnableSequence[str, str]:
        # Prompt template that includes context and question
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert AI assistant specialized in analyzing, summarizing, and reasoning about movie plot documents.
            You receive as input a set of retrieved passages (the CONTEXT) from a large corpus of movie plot summaries.
            Your role is to generate accurate, grounded, and well-structured answers using only the retrieved information.

            =====================
            ### 🔍 Core Principles
            1. **Grounded Responses:**  
            - Base every statement strictly on the retrieved context below.  
            - Never hallucinate or introduce facts not explicitly found in the provided context.  
            - If the evidence is missing, incomplete, or contradictory, explicitly acknowledge this.

            2. **Content Handling:**  
            - Treat each retrieved passage as potentially partial or overlapping.  
            - Synthesize the most coherent interpretation consistent with all retrieved evidence.  
            - Preserve consistency of character names, timeline, and events when summarizing or comparing.

            3. **Formatting & Style:**  
            - Write the response in **Markdown**, with sections like “## Summary”, “## Analysis”, and “## Evidence”.  
            - Use a clear, academic tone — concise, factual, and neutral.  
            - Use bullet points for lists and bold formatting for names, titles, or years.  
            - If quoting, use quotation marks (“ ”) and keep excerpts short.

            4. **Behavior in Edge Cases:**  
            - If the retrieved content is irrelevant or insufficient, respond:  
                “No sufficient information was found in the retrieved movie documents to answer this question.”  
            - When facing conflicting information, explain the discrepancies objectively.

            5. **Self-Verification Checklist (before finalizing the answer):**
            - [ ] Is every statement grounded in the context?  
            - [ ] Did I avoid assumptions or external knowledge?  
            - [ ] Is the response structured, clear, and traceable to evidence?  
            - [ ] Did I communicate uncertainty transparently?

            =====================
            ### ⚙️ Output Guidelines
            - Respond in **English** unless instructed otherwise.
            - Begin with a brief reasoning overview (1–2 sentences) if applicable.
            - End with a “Source Acknowledgment” section listing the titles or metadata mentioned in the context.

            =====================
            ### ⚠️ Forbidden Behaviors
            - Do **not** fabricate events, characters, or details.
            - Do **not** use general movie knowledge beyond what is present in the context.
            - Do **not** expose reasoning steps or internal deliberations.

            =====================
            ### Provided Context:
            {context}

            ### User Question:
            {question}
            """
        )

        return (
            {
                "context": self.retriever | RunnableLambda(self._combine_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Run the RAG pipeline for a given user question."""
        print(f"\nRunning RAG chat with model: {self.model_name}")
        print(f"Question: {question}")

        result = self.chain.invoke(question)

        print("Answer generated successfully.\n")
        return result

    def ask_llm_only(self, question: str) -> str:
        print(f"\nRunning LLM only (no RAG) with model: {self.model_name}")
        print(f"Question: {question}")
        
        response = self.llm.invoke([HumanMessage(content=question)])
        
        print("LLM-only answer generated.\n")
        return response.content