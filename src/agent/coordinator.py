import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from langgraph.types import Command
from langgraph.graph import END

from src.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.prompt import get_prompt
from src.agent.state import ResearchState

logger = logging.getLogger(__name__)


@tool
def handoff_to_lead():
    """Handoff to lead agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to lead agent
    return


def coordinator(
        state: ResearchState, config: RunnableConfig
) -> Command[Literal["lead_agent", "__end__"]]:
    """Coordinator """
    logger.info("coordinator running.")
    configurable = Configuration.from_runnable_config(config)
    messages = get_prompt("coordinator", state)

    llm = get_llm_by_type()

    response = llm.bind_tools([handoff_to_lead]).invoke(messages)
    logger.info(f"coordinator response: {response}")
    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            logger.info(f"tool_call: {tool_call}")
        return Command(goto="lead_agent")

    return Command(
        update={"messages": [AIMessage(content=response.content, name="coordinator")]},
        goto=END
    )
