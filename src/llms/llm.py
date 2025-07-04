import logging
from pathlib import Path

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def get_llm_by_type() -> ChatOpenAI:
    """Get llm instance."""
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    )


if __name__ == "__main__":
    # Initialize LLMs for different purposes - now these will be cached
    basic_llm = get_llm_by_type()
    logger.info(basic_llm.invoke("Hello"))
