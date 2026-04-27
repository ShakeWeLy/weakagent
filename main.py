# import asyncio
# from weakagent.config.settings import config
# from weakagent.llm.factory import LLMFactory


# async def main():
#     llm = LLMFactory.create(config_name="default")
#     print(llm.model)

#     context = await llm.ask([
#         {"role": "user", "content": "Hello, how are you?"}
#     ])
#     # print(context)


# if __name__ == "__main__":
#     asyncio.run(main())