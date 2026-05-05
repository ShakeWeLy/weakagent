from weakagent.agent.brief_react import BriefReActAgent
from weakagent.tools.terminate import Terminate
from weakagent.tools.tool_collection import ToolCollection
from weakagent.utils.verbose import verbose_result
from weakagent.llm.llm import LLM

def test_obj_detection_tool():
    from weakagent.tools.images.read_images import ReadImagesTool
    from weakagent.tools.images.obj_detection import ObjDetectionTool
    agent = BriefReActAgent(
    max_steps=5,
    available_tools=ToolCollection(ReadImagesTool(), ObjDetectionTool(), Terminate()),
    )
    agent.llm = LLM(config_name="default")

    result = asyncio.run(agent.run("Read the image(path:./Snipaste_2026-04-12_21-25-08.jpeg) and tell me the postion of blue box in the image"))

    verbose_result(result, agent)


def test_file_tool():
    from weakagent.tools.files.grep_file import GrepTool
    from weakagent.tools.files.list_file import ListFilesTool
    agent = BriefReActAgent(
    max_steps=5,
    available_tools=ToolCollection(GrepTool(), ListFilesTool(), Terminate()),
    )
    agent.llm = LLM(config_name="default")

    result = asyncio.run(agent.run("help me to test ListFilesTool, and GrepTool in simple way"))

    verbose_result(result, agent)



if __name__ == "__main__":
    import asyncio

    # asyncio.run(main())
    # test_obj_detection_tool()
    test_file_tool()
