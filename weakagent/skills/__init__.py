"""
Skills module for agent system.

Skills are markdown files with frontmatter that provide specialized instructions
for specific tasks. Loaded from project ``skills/`` and ``workspace/skills/``.
"""

from weakagent.skills.types import (
    Skill,
    SkillEntry,
    SkillMetadata,
    SkillInstallSpec,
    LoadSkillsResult,
    SkillSnapshot,
)
from weakagent.skills.loader import SkillLoader
from weakagent.skills.manager import SkillManager
from weakagent.skills.service import SkillService
from weakagent.skills.formatter import format_skills_for_prompt

__all__ = [
    "Skill",
    "SkillEntry",
    "SkillMetadata",
    "SkillInstallSpec",
    "LoadSkillsResult",
    "SkillSnapshot",
    "SkillLoader",
    "SkillManager",
    "SkillService",
    "format_skills_for_prompt",
]
