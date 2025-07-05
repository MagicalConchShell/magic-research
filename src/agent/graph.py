"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.coordinator import coordinator
from src.agent.human_feedback import human_feedback
from src.agent.lead_agent import lead_agent
from src.agent.reporter import reporter
from src.agent.state import ResearchState
from src.agent.sub_agent import generate_multi_sub_agent, sub_agent
from src.configuration import Configuration

load_dotenv()


def build_graph():
    """Build and return the agent workflow graph without memory."""
    builder = StateGraph(ResearchState, config_schema=Configuration)
    builder.add_node("coordinator", coordinator)
    builder.add_node("lead_agent", lead_agent)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("sub_agent", sub_agent)
    builder.add_node("reporter", reporter)
    builder.add_edge(START, "coordinator")
    # Add conditional edge to continue with sub agent
    builder.add_conditional_edges("human_feedback", generate_multi_sub_agent, ["sub_agent"])
    builder.add_edge("sub_agent", "reporter")
    builder.add_edge("reporter", END)
    return builder.compile()


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    memory = MemorySaver()

    builder = StateGraph(ResearchState, config_schema=Configuration)
    builder.add_node("lead", lead_agent)
    builder.add_edge(START, "lead")
    builder.add_edge("lead", END)

    return builder.compile(checkpointer=memory)


graph = build_graph()

if __name__ == "__main__":
    state = {
        # [{"role":"user","content":"中美货币政策对全球金融市场的影响"}]
        # "messages": [HumanMessage(content="中美货币政策对全球金融市场的影响")],
        "messages": [HumanMessage(content="how are u")],
    }
    print(graph.invoke(state))
