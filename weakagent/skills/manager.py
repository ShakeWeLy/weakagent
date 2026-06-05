"""
Skill manager for managing skill lifecycle and operations.
"""

import json
import os
from typing import Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from weakagent.config.settings import PROJECT_ROOT
from weakagent.utils.logger import get_logger
from weakagent.skills.types import SkillEntry, SkillSnapshot
from weakagent.skills.loader import SkillLoader
from weakagent.skills.formatter import format_skill_entries_for_prompt

logger = get_logger(__name__)

SKILLS_CONFIG_FILE = "skills_config.json"


def load_skills_settings() -> dict:
    """Load optional [skills] section from config.toml."""
    cfg_path = PROJECT_ROOT / "config.toml"
    if not cfg_path.exists():
        return {}
    try:
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    section = raw.get("skills") or {}
    return section if isinstance(section, dict) else {}


class SkillManager:
    """Manages skills for an agent."""

    def __init__(
        self,
        builtin_dir: Optional[str] = None,
        custom_dir: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize the skill manager.

        Args:
            builtin_dir: Built-in skills directory (project root ``skills/``)
            custom_dir: Custom skills directory (workspace ``skills/``)
            config: Optional runtime config (e.g. env keys for skill requirements)
        """
        settings = load_skills_settings()
        project_root = str(PROJECT_ROOT)

        self.builtin_dir = builtin_dir or settings.get("builtin_dir") or os.path.join(
            project_root, "skills"
        )
        if not os.path.isabs(self.builtin_dir):
            self.builtin_dir = os.path.join(project_root, self.builtin_dir)

        self.custom_dir = custom_dir or settings.get("custom_dir") or os.path.join(
            project_root, "workspace", "skills"
        )
        if not os.path.isabs(self.custom_dir):
            self.custom_dir = os.path.join(project_root, self.custom_dir)

        self.config = config or {}
        self.exclude_skills: List[str] = list(settings.get("exclude") or [])
        self._skills_config_path = os.path.join(self.custom_dir, SKILLS_CONFIG_FILE)

        self.skills_config: Dict[str, dict] = {}
        self.loader = SkillLoader()
        self.skills: Dict[str, SkillEntry] = {}

        self.refresh_skills()

    def refresh_skills(self):
        """Reload all skills from builtin and custom directories, then sync config."""
        self.skills = self.loader.load_all_skills(
            builtin_dir=self.builtin_dir,
            custom_dir=self.custom_dir,
        )
        self._sync_skills_config()
        logger.debug("SkillManager: loaded %s skills", len(self.skills))

    def _load_skills_config(self) -> Dict[str, dict]:
        """Load skills_config.json from custom_dir. Returns empty dict if not found."""
        if not os.path.exists(self._skills_config_path):
            return {}
        try:
            with open(self._skills_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning("[SkillManager] Failed to load %s: %s", SKILLS_CONFIG_FILE, e)
        return {}

    def _save_skills_config(self):
        """Persist skills_config to custom_dir/skills_config.json."""
        os.makedirs(self.custom_dir, exist_ok=True)
        try:
            with open(self._skills_config_path, "w", encoding="utf-8") as f:
                json.dump(self.skills_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error("[SkillManager] Failed to save %s: %s", SKILLS_CONFIG_FILE, e)

    def _sync_skills_config(self):
        """Merge scanned skills with persisted skills_config.json."""
        saved = self._load_skills_config()
        merged: Dict[str, dict] = {}

        for name, entry in self.skills.items():
            skill = entry.skill
            prev = saved.get(name, {})
            category = prev.get("category", "skill")

            if name in saved:
                enabled = prev.get("enabled", True)
            else:
                enabled = entry.metadata.default_enabled if entry.metadata else True

            entry_dict = {
                "name": name,
                "description": skill.description,
                "source": prev.get("source") or skill.source,
                "enabled": enabled,
                "category": category,
            }
            display_name = prev.get("display_name")
            if display_name:
                entry_dict["display_name"] = display_name
            merged[name] = entry_dict

        self.skills_config = merged
        self._save_skills_config()

    def is_skill_enabled(self, name: str) -> bool:
        """Return True if skill is enabled in skills_config (default True)."""
        entry = self.skills_config.get(name)
        if entry is None:
            return True
        return entry.get("enabled", True)

    def set_skill_enabled(self, name: str, enabled: bool):
        """Set enabled flag and persist to skills_config.json."""
        if name not in self.skills_config:
            raise ValueError(f"skill '{name}' not found in config")
        self.skills_config[name]["enabled"] = enabled
        self._save_skills_config()

    def get_skills_config(self) -> Dict[str, dict]:
        """Return a copy of skills_config for query APIs."""
        return dict(self.skills_config)

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        return self.skills.get(name)

    def list_skills(self) -> List[SkillEntry]:
        return list(self.skills.values())

    @staticmethod
    def _normalize_skill_filter(skill_filter: Optional[List[str]]) -> Optional[List[str]]:
        if skill_filter is None:
            return None
        normalized = []
        for item in skill_filter:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    normalized.append(name)
            elif isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, str):
                        name = subitem.strip()
                        if name:
                            normalized.append(name)
        return normalized or None

    def filter_skills(
        self,
        skill_filter: Optional[List[str]] = None,
        include_disabled: bool = False,
    ) -> List[SkillEntry]:
        """Filter skills that are eligible (enabled + requirements met)."""
        from weakagent.skills.config import should_include_skill

        entries = list(self.skills.values())
        entries = [e for e in entries if should_include_skill(e, self.config)]

        normalized = self._normalize_skill_filter(skill_filter)
        if normalized is not None:
            entries = [e for e in entries if e.skill.name in normalized]

        if not include_disabled:
            entries = [e for e in entries if self.is_skill_enabled(e.skill.name)]

        if self.exclude_skills:
            entries = [e for e in entries if e.skill.name not in self.exclude_skills]

        return entries

    def filter_unavailable_skills(
        self,
        skill_filter: Optional[List[str]] = None,
    ) -> tuple:
        """Return enabled skills whose requirements are not met."""
        from weakagent.skills.config import should_include_skill, get_missing_requirements

        entries = list(self.skills.values())
        entries = [e for e in entries if self.is_skill_enabled(e.skill.name)]

        normalized = self._normalize_skill_filter(skill_filter)
        if normalized is not None:
            entries = [e for e in entries if e.skill.name in normalized]

        unavailable = []
        missing_map: Dict[str, dict] = {}
        for e in entries:
            if not should_include_skill(e, self.config):
                missing = get_missing_requirements(e)
                if missing:
                    unavailable.append(e)
                    missing_map[e.skill.name] = missing

        return unavailable, missing_map

    def build_skills_prompt(
        self,
        skill_filter: Optional[List[str]] = None,
    ) -> str:
        """Build `<available_skills>` block for the system prompt."""
        from weakagent.skills.formatter import format_unavailable_skills_for_prompt

        eligible = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        logger.debug(
            "[SkillManager] eligible: %s / %s skills",
            len(eligible),
            len(self.skills),
        )

        result = format_skill_entries_for_prompt(eligible)

        unavailable, missing_map = self.filter_unavailable_skills(skill_filter=skill_filter)
        if unavailable:
            result += format_unavailable_skills_for_prompt(unavailable, missing_map)

        return result

    def build_skill_snapshot(
        self,
        skill_filter: Optional[List[str]] = None,
        version: Optional[int] = None,
    ) -> SkillSnapshot:
        entries = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        prompt = format_skill_entries_for_prompt(entries)

        skills_info = []
        resolved_skills = []
        for entry in entries:
            skills_info.append(
                {
                    "name": entry.skill.name,
                    "primary_env": entry.metadata.primary_env if entry.metadata else None,
                }
            )
            resolved_skills.append(entry.skill)

        return SkillSnapshot(
            prompt=prompt,
            skills=skills_info,
            resolved_skills=resolved_skills,
            version=version,
        )

    def sync_skills_to_workspace(self, target_workspace_dir: str):
        """Copy loaded skills into a sandbox workspace."""
        import shutil

        target_skills_dir = os.path.join(target_workspace_dir, "skills")
        if os.path.exists(target_skills_dir):
            shutil.rmtree(target_skills_dir)
        os.makedirs(target_skills_dir, exist_ok=True)

        for entry in self.skills.values():
            skill_name = entry.skill.name
            source_dir = entry.skill.base_dir
            target_dir = os.path.join(target_skills_dir, skill_name)
            try:
                shutil.copytree(source_dir, target_dir)
                logger.debug("Synced skill '%s' to %s", skill_name, target_dir)
            except Exception as e:
                logger.warning("Failed to sync skill '%s': %s", skill_name, e)

        logger.info("Synced %s skills to %s", len(self.skills), target_skills_dir)

    def get_skill_by_key(self, skill_key: str) -> Optional[SkillEntry]:
        for entry in self.skills.values():
            if entry.metadata and entry.metadata.skill_key == skill_key:
                return entry
            if entry.skill.name == skill_key:
                return entry
        return None
