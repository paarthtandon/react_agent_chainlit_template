from typing import Any, AsyncIterator

import chainlit as cl
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.schema import StreamEvent

class ChainlitStreamDispatcher:
    """
    Dispatches chainlit messages from a langgraph event stream.
    """

    _stream: AsyncIterator[StreamEvent]
    _current_output_content: str
    _run_names: dict[str, str]
    _current_message: cl.Message | None
    _messages_sent: int
    _session: str

    def __init__(self, stream: AsyncIterator[StreamEvent], session: str):
        self._stream = stream
        self._current_output_content = ""
        self._run_names = {}
        self._current_message = None
        self._messages_sent = 0
        self._session = session

    async def on_output_message_start(self):
        """A new output message is started."""
        self._current_message = cl.Message(content="")
        await self._current_message.send()

    async def on_output_chunk(self, content: str):
        """A chunk of displayable output was streamed."""
        assert self._current_message is not None
        await self._current_message.stream_token(content)

    async def on_output_message_complete(self):
        """The currently streamed output message is complete."""
        await self._current_message.update()
        self._current_message = None
        self._messages_sent += 1

    async def on_tool_call_complete(self, name: str, output: ToolMessage | Any):
        """Result of a tool call."""
        if isinstance(output, ToolMessage):
            output.pretty_print()
        else:
            print(f"{name} returned: {output}")

    async def on_chat_model_complete(self, name: str, output: AIMessage):
        """Result of a chat model call."""
        output.pretty_print()

    async def on_chain_call(self, name: str, input: dict):
        """A chain invocation."""

    async def run(self):
        """Consume the event stream, dispatching method hooks if triggered."""
        async for stream_event in self._stream:
            event = stream_event["event"]
            name = stream_event["name"]
            data = stream_event["data"]
            parents = stream_event["parent_ids"]

            self._run_names[stream_event["run_id"]] = name
            parent_names = [
                self._run_names[p] for p in parents[:-1] if p in self._run_names
            ]

            match event:
                case "on_chat_model_stream":
                    chunk = data["chunk"]
                    content = chunk.content

                    if 'tool_calls' in chunk.additional_kwargs:
                        continue

                    if not parent_names:
                        continue

                    if not content:
                        continue

                    if 'tools' not in parent_names:
                        if not self._current_output_content:
                            await self.on_output_message_start()

                        self._current_output_content += content
                        await self.on_output_chunk(content)

                case "on_chat_model_end":
                    if self._current_output_content:
                        await self.on_output_message_complete()

                    self._current_output_content = ""
                    await self.on_chat_model_complete(name, data["output"])

                case "on_tool_end":
                    await self.on_tool_call_complete(name, data["output"])

                case "on_chain_start":
                    await self.on_chain_call(name, data.get("input", {}))
                case _:
                    continue

    @property
    def messages_sent(self):
        return self._messages_sent