from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


class MessagesState(MessagesState):
    pass


def add_two_number(a: int, b: int) -> int:
    """
    A simple function to add two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of the two numbers.
    """
    return a + b


llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools=[add_two_number])


def get_llm_response(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state.get("messages"))]}


def show_the_last_response(state: MessagesState):
    return {"messages": state.get("messages")}


builder = StateGraph(MessagesState)
builder.add_node("get_llm_response", get_llm_response)

builder.add_edge(START, "get_llm_response")
builder.add_edge("get_llm_response", END)

graph = builder.compile()
