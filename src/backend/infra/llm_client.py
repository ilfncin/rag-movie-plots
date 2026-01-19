import logging

from langchain_openai import ChatOpenAI
from backend.config.settings import LLM_CONFIG

logger = logging.getLogger("LLM_Client")

class LLMClient:
    """
    Wrapper around LangChain's ChatOpenAI client.

    Centralizes LLM configuration and invocation for observability. 
    This class provides a stateless interface for generating text from a 
    single prompt and intentionally avoids concerns such as prompt 
    construction, retrieval, conversation state, or retries.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"]
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Sends the prompt to the configured LLM, logs token usage statistics,
        and returns the generated text.
        """
        result = self.llm.invoke(prompt)

        usage = result.response_metadata.get("token_usage", {})

        logger.debug(
            "LLM token usage | prompt=%s | completion=%s | total=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

        return result.content