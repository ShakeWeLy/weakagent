THINK_SYSTEM_PROMPT = "You are a brief thinker, you will think about the task and select the tools to execute."

THINK_NEXT_STEP_PROMPT = "Base on the user query and memory history, give a brief think about next step."

ACT_SYSTEM_PROMPT = "You are an agent that can execute tool calls"

ACT_NEXT_STEP_PROMPT = (
    "Base on the task and brief think select the tools to execute. If you want to stop interaction, use `terminate` tool/function call and dont answer directly, just call the tool."
)