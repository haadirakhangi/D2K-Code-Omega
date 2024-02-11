from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema.runnable.config import RunnableConfig
import openai
import chainlit as cl
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

@cl.on_chat_start
async def on_chat_start():
    # model = ChatOpenAI(model='gpt-3.5-turbo-1106',streaming=True)
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You're a helpful csv assistant",
    #         ),
    #         ("human", "{input}"),
    #     ]
    # )
    # runnable = prompt | model | StrOutputParser()
    agent = create_csv_agent(ChatOpenAI(model='gpt-3.5-turbo-1106',temperature=0, streaming=True), 'd2k_data.csv', verbose=True, agent_type = AgentType.OPENAI_FUNCTIONS)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")

    print(agent.invoke({"input": message.content}))
    async for chunk in agent.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
