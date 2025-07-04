import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agent.state import ResearchPlan, ResearchState
from src.llms.llm import get_llm_by_type
from src.prompts.prompt import get_prompt

logger = logging.getLogger(__name__)


def lead_agent(state: ResearchState, config: RunnableConfig) -> Command[Literal["human_feedback", "__end__"]]:
    """Lead agent generate"""
    logger.info("lead agent running.")
    # configurable = Configuration.from_runnable_config(config)
    messages = get_prompt("lead_agent", state)

    llm = get_llm_by_type().with_structured_output(ResearchPlan, method="json_mode")

    research_plan = llm.invoke(messages)
    # full_response = response.model_dump_json(indent=4, exclude_none=True)
    logger.info(f"Planner response: {research_plan} ")

    return Command(
        update={
            "messages": [AIMessage(content=research_plan.model_dump_json(indent=4, exclude_none=True), name="planner")],
            "research_plan": research_plan,
        },
        goto="human_feedback",
    )
