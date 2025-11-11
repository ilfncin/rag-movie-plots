from backend.chat.chat_rag import ChatRAG
# PYTHONPATH=src uv run src/tests/backend/chat_rag_tester.py
if __name__ == "__main__":
    chat = ChatRAG(verbose=True)

    question = "Who directed Titanic and what is the movie about?"
    
    answer_rag = chat.ask(question)
    print(f"\nAnswer WITH RAG:\n{answer_rag}")


    answer_llm = chat.ask_llm_only(question)
    print(f"\nAnswer WITHOUT RAG:\n{answer_llm}")