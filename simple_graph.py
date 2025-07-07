import random

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal, TypedDict

load_dotenv()


class GraphState(TypedDict):
    graph_state: str


def node_01(state: GraphState) -> GraphState:
    return {"graph_state": "Hello, My name is Khang and i'm"}


def node_02(state: GraphState) -> GraphState:
    return {"graph_state": state.get("graph_state") + " happy!"}


def node_03(state: GraphState) -> GraphState:
    return {"graph_state": state.get("graph_state") + " sad!"}


def decide_mood(state: GraphState) -> Literal["node_02", "node_03"]:
    if random.random() > 0.5:
        return "node_02"

    return "node_03"


builder = StateGraph(GraphState)
builder.add_node("node_01", node_01)
builder.add_node("node_02", node_02)
builder.add_node("node_03", node_03)

builder.add_edge(START, "node_01")
builder.add_conditional_edges("node_01", decide_mood)
builder.add_edge("node_02", END)
builder.add_edge("node_03", END)

graph = builder.compile()
