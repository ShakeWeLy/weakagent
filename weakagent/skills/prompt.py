"""System-prompt fragments for agent skills."""

SKILLS_USAGE_PROMPT = """
## Skills

You have access to a skills catalog in `<available_skills>`.
When a user task matches a skill:
1. Pick the best skill by name and description.
2. Use the `read` tool to load the skill file at `<location>` (usually SKILL.md).
3. Follow the skill instructions; use other tools as needed.

Do not guess skill contents without reading the file first.
"""
