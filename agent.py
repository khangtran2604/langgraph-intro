from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class AppState(MessagesState):
    pass


def summary(a: int, b: int) -> int:
    """
    Sums two integers.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    """
    return a * b


def devide(a: int, b: int) -> int:
    """
    Divides two integers.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    """
    return a / b


llm = ChatOpenAI(model="gpt-4o")
llm_with_tool = llm.bind_tools([summary, multiply, devide])

sys_prompt = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: AppState):
    return {"messages": [llm_with_tool.invoke([sys_prompt] + state.get("messages"))]}


builder = StateGraph(AppState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([summary, multiply, devide]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()
