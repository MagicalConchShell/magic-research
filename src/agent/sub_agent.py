import logging
from typing import Annotated

from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

from src.agent.state import ResearchState, SubAgentState
from src.llms.llm import get_llm_by_type
from src.prompts.prompt import get_prompt
from src.tools.jina_client import crawl

logger = logging.getLogger(__name__)


def generate_multi_sub_agent(state: ResearchState, config: RunnableConfig):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """

    plan = state.get("research_plan")

    return [
        Send("sub_agent", {"id": int(idx), "subagent": agent})
        for idx, agent in enumerate(plan.subagents)
    ]


@tool
def complete_task():
    """Handoff to lead agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to lead agent
    return


@tool
def web_fetch(url: Annotated[str, "The url to crawl."]):
    """Use this to crawl a url and get a readable content in markdown format."""
    try:
        article = crawl(url)
        return {"url": url, "crawled_content": article.to_markdown()[:1000]}
    except BaseException as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def sub_agent(
        state: SubAgentState, config: RunnableConfig
) -> ResearchState:
    """Sub agent generate"""
    logger.info(f"sub agent [{state.get('id')}] running.")
    subagent = state.get("subagent")

    tools = [ArxivQueryRun(name="web_search"), complete_task]

    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Current Task\n\n## Title\n\n{subagent.title}\n\n## Description\n\n{subagent.description}\n\n"
            )
        ]
    }

    llm = get_llm_by_type()
    agent = create_react_agent(
        name="sub_agent",
        model=llm,
        tools=tools,
        prompt=lambda state: get_prompt("sub_agent", state),
    )

    logger.info(f"Agent input: {agent_input}")
    result = agent.invoke(input=agent_input)

    response_content = result["messages"][-1].content
    logger.debug(f" full response: {response_content}")

    return {"subagent_result": [response_content]}
