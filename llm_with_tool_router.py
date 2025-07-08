from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class AppState(MessagesState):
    pass


def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    Returns:
        int: The product of the two integers.
    """
    return a * b


llm = ChatOpenAI(model="gpt-4o")
llm_with_tool = llm.bind_tools([multiply])


def invoke_llm(state: AppState):
    return {"messages": [llm_with_tool.invoke(state.get("messages"))]}


builder = StateGraph(AppState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
