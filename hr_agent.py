import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load OPENAI_API_KEY from the .env file
load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = AssistantAgent(
    name="hr_assistant",
    model_client=model_client,
    system_message="You are an HR assistant.",
)


async def main():
    await Console(
        agent.run_stream(
            task="Write a job description for a ADAS Program Manager "
                 "in exactly 5 lines: a title line followed by 4 numbered "
                 "points. Output nothing else."
        )
    )
    await model_client.close()


asyncio.run(main())