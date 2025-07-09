from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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


def divide(a: int, b: int) -> int:
    """
    Divides two integers.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    """
    return a / b


llm = ChatOpenAI(model="gpt-4o")
llm_with_tool = llm.bind_tools([summary, multiply, divide])


sys_prompt = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: AppState):
    return {"messages": [llm_with_tool.invoke([sys_prompt] + state.get("messages"))]}


builder = StateGraph(AppState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([summary, multiply, divide]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory_checkpointer = MemorySaver()
graph = builder.compile(checkpointer=memory_checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "khangtran_123"}}

init_messages = [HumanMessage(content="What is 3 multiply 4?")]

resp = graph.invoke({"messages": init_messages}, config)

for mess in resp.get("messages"):
    mess.pretty_print()


next_messages = [HumanMessage(content="Multiply that with 5")]

resp = graph.invoke({"messages": next_messages}, config)

for mess in resp.get("messages"):
    mess.pretty_print()
