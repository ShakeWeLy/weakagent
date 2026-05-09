from weakagent.tools.base import BaseTool, ToolExecutionResult


class RunSubAgentTool(BaseTool):
    name: str = "run_sub_agent"
    description: str = (
        "Run a sub-agent by existing sub_agent_id, or by factory name "
        "(which will create the sub-agent first)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "sub_agent_id": {
                "type": "string",
                "description": "Existing managed sub-agent id to run.",
            },
            "sub_agent_name": {
                "type": "string",
                "description": "Sub-agent factory name to execute when sub_agent_id is not provided.",
            },
            "request": {
                "type": "string",
                "description": "request sent to the sub-agent, tell the sub-agent what to do.",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        sub_agent_id: str | None = None,
        sub_agent_name: str | None = None,
        request: str | None = None,
        caller_agent_id: str | None = None,
    ) -> ToolExecutionResult:
        # Delayed imports to avoid circular dependency
        from weakagent.agent.runtime import AgentRuntime

        runtime = AgentRuntime.get_instance()
        if runtime is None:
            return self.fail_response(
                "AgentRuntime singleton not initialized. "
                "Call `await AgentRuntime.instance()` first."
            )

        if sub_agent_id:
            try:
                runtime.get(sub_agent_id)
            except Exception as exc:
                return self.fail_response(f"Invalid sub_agent_id `{sub_agent_id}`: {exc}")
        else:
            if not sub_agent_name:
                return self.fail_response(
                    "Either sub_agent_id or sub_agent_name is required"
                )

            factory = runtime.factory
            if sub_agent_name not in factory.supported_types:
                return self.fail_response(
                    f"Unknown sub-agent name: {sub_agent_name}. "
                    f"Supported types: {', '.join(factory.supported_types)}"
                )

            # Create and register the sub-agent under parent (caller) via runtime
            try:
                sub_agent_id = runtime.create_agent(
                    sub_agent_name,
                    parent_id=caller_agent_id,
                )
            except Exception as exc:
                return self.fail_response(
                    f"Failed to create sub-agent `{sub_agent_name}`: {exc}"
                )

            # Update caller's active sub-agent pointer when available.
            if caller_agent_id:
                try:
                    caller = runtime.get(caller_agent_id)
                    if hasattr(caller, "current_sub_agent_id"):
                        setattr(caller, "current_sub_agent_id", sub_agent_id)
                except Exception:
                    pass

        # Run the sub-agent
        try:
            child_result = await runtime.run(sub_agent_id, request=request)
        except Exception as exc:
            return self.fail_response(
                f"sub-agent `{sub_agent_name}` execution failed: {exc}"
            )

        return self.success_response(
            {
                "sub_agent_name": sub_agent_name,
                "sub_agent_id": sub_agent_id,
                "request": request,
                "message": f"Sub-agent `{sub_agent_id}` executed successfully",
                "result": child_result,
            }
        )

