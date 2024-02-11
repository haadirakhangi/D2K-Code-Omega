from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema.runnable.config import RunnableConfig
import openai
import chainlit as cl
import pandas as pd
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

df = pd.read_csv("d2k_data.csv")

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(model='gpt-3.5-turbo-1106',streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a helpful csv assistant",
            ),
            ("human", "{input}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True, agent_type = AgentType.OPENAI_FUNCTIONS)
    cl.user_session.set("runnable", agent)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    print(runnable.invoke({"input": message.content}))
    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
