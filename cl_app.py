import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from template.agents.react.react_agent import build_react_agent
from template.stream import ChainlitStreamDispatcher

agent = build_react_agent()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chain", agent)

@cl.on_message
async def on_message(input_message: cl.Message):
    chain: CompiledStateGraph = cl.user_session.get("chain")
    session_id = cl.user_session.get("id")

    message = HumanMessage(
        content=[
            {"type": "text", "text": input_message.content},
        ]
    )
    message.pretty_print()

    inputs = {
        "messages": [message]
    }
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    dispatcher = ChainlitStreamDispatcher(
        chain.astream_events(inputs, config=config, version="v2"), session_id
    )
    await dispatcher.run()

    if dispatcher.messages_sent == 0:
        await cl.Message(
            content="Something went wrong, the agent did not output a message!"
        ).send()
