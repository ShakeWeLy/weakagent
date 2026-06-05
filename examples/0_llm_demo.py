import asyncio
from weakagent.llm import LLM
from weakagent.memory import Message

llm = LLM(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-",
    max_tokens=8192,
    temperature=0.0,
    supports_images=True,
    use_max_completion_tokens=False,
    enable_think="default",
)

messages = [Message.user_message("Hello, which model are you using?")]

content = asyncio.run(llm.ask(messages))
print(content)