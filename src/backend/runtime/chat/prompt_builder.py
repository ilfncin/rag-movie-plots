from backend.runtime.prompts.rag_movie_v1 import RAG_MOVIE_PROMPT_V1

class PromptBuilder:
    """
    Builds the final prompt for RAG by injecting the question and
    retrieved context into a versioned prompt template.
    """
    def build(self, question: str, context: str) -> str:
        return RAG_MOVIE_PROMPT_V1.format(
            question=question,
            context=context
        )