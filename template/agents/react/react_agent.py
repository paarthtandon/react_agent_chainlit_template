from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.prebuilt import ToolExecutor, create_react_agent
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from template.tools import tools
from template.agents.react.prompts import SYSTEM_PROMPT


def build_react_agent() -> Runnable:
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
    model = ChatOpenAI(model='gpt-4o', temperature=0, rate_limiter=rate_limiter)

    system_messages = [
        SYSTEM_PROMPT
    ]

    state_modifier = RunnableLambda(
        lambda state: [SystemMessage(content=m) for m in system_messages]
        + state["messages"],
        name="StateModifier",
    )
    react_agent_executor = create_react_agent(
        model, ToolExecutor(tools), state_modifier=state_modifier
    )

    config = RunnableConfig(recursion_limit=50)

    def invoke_agent(state: MessagesState):
        response = react_agent_executor.invoke({"messages": state["messages"]}, config=config)
        return {"messages": response["messages"]}

    async def ainvoke_agent(state: MessagesState):
        response = await react_agent_executor.ainvoke({"messages": state["messages"]}, config=config)
        return {"messages": response["messages"]}

    workflow = StateGraph(MessagesState)
    workflow.add_node('agent', RunnableLambda(invoke_agent, ainvoke_agent))

    workflow.add_edge(START, 'agent')
    workflow.add_edge('agent', END)
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory).with_config(
        {"run_name": "ReAct Agent"}
    )
