from backend.retriever.retriever import Retriever


class RetrieverTester:
    """
    Simple test runner for the Retriever class.

    Given a list of natural language questions, it retrieves and prints
    the top_k most relevant documents from the Chroma vector store.

    This is useful for debugging or qualitative evaluation before
    integrating with an LLM or chatbot pipeline.
    """

    def __init__(self, top_k=5):
        # Load retriever with the given top_k
        self.pipeline = Retriever(top_k=top_k)
        self.retriever = self.pipeline.load()
        self.top_k = top_k
        print("\nRetriever loaded successfully.\n")

    def test_questions(self, questions: list[str]):
        print(f"\nTesting {len(questions)} questions with top_k={self.top_k}...\n")

        for i, question in enumerate(questions, start=1):
            print(f"\nQuestion {i}: {question}")
            try:
                docs = self.retriever.invoke(question)

                for j, doc in enumerate(docs[:self.top_k], start=1):
                    print(f"\nDocument {j}:")
                    print(f"Text preview: {doc.page_content[:300]}...")  # Show up to 300 chars
                    print(f"Metadata: {doc.metadata}")
            except Exception as e:
                print(f"Error processing question '{question}': {e}")

            print("\n" + "-" * 100)


if __name__ == "__main__":
    tester = RetrieverTester(top_k=5)

    example_questions = [
        "What is the plot of the movie Titanic?",
        "Who directed Pulp Fiction?",
        "List some science fiction movies from the 1990s.",
        "Which movies were made in India?"
    ]

    tester.test_questions(example_questions)
