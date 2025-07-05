from __future__ import annotations

import operator
from enum import Enum
from typing import List, Optional, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class Subagent(BaseModel):
    title: str = Field(..., description="The title of the subagent.")
    description: str = Field(..., description="Specify exactly what data to collect")
    prompt: str = Field(..., description="Prompt to run")


class QueryTypeEnum(str, Enum):
    DEPTH_FIRST = "Depth-first"
    BREADTH_FIRST = "Breadth-first"
    Straightforward = "Straightforward"


class ResearchPlan(BaseModel):
    title: str
    query_type: QueryTypeEnum = Field(..., description="Query type of the query")
    subagents: List[Subagent] = Field(
        default_factory=list,
        description="Sub agents to get more context",
    )


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_plan: Optional[ResearchPlan]
    query_type: Optional[str]
    subagent_result: Annotated[list, operator.add]

    final_report: Optional[str]
    error: Optional[str]


class SubAgentState(TypedDict):
    id: str
    subagent: Subagent
