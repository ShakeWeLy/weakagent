from weakagent.tools.base import BaseTool, ToolExecutionResult


class RunSubAgentTool(BaseTool):
    name: str = "run_sub_agent"
    description: str = (
        "Run a sub-agent by factory name (e.g., 'chat', 'toolcall', 'brief_react') "
        "with an optional delegated request. The sub-agent will be dynamically created."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "sub_agent_name": {
                "type": "string",
                "description": "Sub-agent factory name to execute. Supported: chat, toolcall, brief_react, multi_react, reacttoolcall.",
            },
            "request": {
                "type": "string",
                "description": "request sent to the sub-agent, tell the sub-agent what to do.",
            },
        },
        "required": ["sub_agent_name","request"],
    }

    async def execute(
        self,
        sub_agent_name: str,
        request: str | None = None,
        caller_agent_id: str | None = None,
    ) -> ToolExecutionResult:
        # Delayed imports to avoid circular dependency
        from weakagent.agent.factory import AgentFactory
        from weakagent.agent.manager import AgentManager

        if not sub_agent_name:
            return self.fail_response("sub_agent_name is required")

        manager = AgentManager.get_instance()
        if manager is None:
            return self.fail_response(
                "AgentManager singleton not initialized. "
                "Call `await AgentManager.instance()` first."
            )

        # Validate sub-agent name via factory
        factory = manager.factory
        if sub_agent_name not in factory.supported_types:
            return self.fail_response(
                f"Unknown sub-agent name: {sub_agent_name}. "
                f"Supported types: {', '.join(factory.supported_types)}"
            )

        # Create and register the sub-agent under parent (caller) via manager
        try:
            sub_agent_id = manager.create_agent(
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
                caller = manager.get(caller_agent_id)
                if hasattr(caller, "current_sub_agent_id"):
                    setattr(caller, "current_sub_agent_id", sub_agent_id)
            except Exception:
                pass

        # Run the sub-agent
        try:
            child_result = await manager.run(sub_agent_id, request=request)
        except Exception as exc:
            return self.fail_response(
                f"sub-agent `{sub_agent_name}` execution failed: {exc}"
            )

        return self.success_response(
            {
                "sub_agent_name": sub_agent_name,
                "sub_agent_id": sub_agent_id,
                "request": request,
                "message": f"Sub-agent `{sub_agent_name}` executed successfully",
                "result": child_result,
            }
        )

