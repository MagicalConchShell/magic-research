import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.state import ResearchState
from src.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.prompt import get_prompt

logger = logging.getLogger(__name__)


def reporter(
        state: ResearchState, config: RunnableConfig
) -> ResearchState:
    """Report the final answer"""
    logger.info("report running.")
    configurable = Configuration.from_runnable_config(config)
    messages = get_prompt("reporter", state)

    plan = state.get('research_plan')

    messages.append(
        HumanMessage(
            content=f"# Research Requirements\n\n## Task\n\n{plan.title}\n",
            name="observation",
        )
    )

    for result in state.get('subagent_result'):
        messages.append(
            HumanMessage(
                content=f"Below are some result for the research task:\n\n{result}",
                name="observation",
            )
        )

    logger.debug(f"messages: {messages}")

    response = get_llm_by_type().invoke(messages)
    response_content = response.content
    logger.debug(f"response_content: {response_content}")

    return {"final_report": response_content}
