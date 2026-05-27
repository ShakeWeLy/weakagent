"""Collection classes for managing multiple tools."""

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Set

from weakagent.utils.exceptions import ToolError
from weakagent.utils.logger import get_logger
from weakagent.tools.base import BaseTool, ToolExecutionResult

logger = get_logger(__name__)


# ── Built-in tool registry for dynamic discovery ──────────────────────────
_BUILTIN_TOOL_REGISTRY: Dict[str, type[BaseTool]] = {}


def _discover_builtin_tools() -> Dict[str, type[BaseTool]]:
    """Scan weakagent.tools sub-packages and collect all BaseTool subclasses.

    Returns a dict mapping ``{tool_name: ToolClass}`` for every concrete
    :class:`BaseTool` found in the ``weakagent.tools`` package tree.
    """
    discovered: Dict[str, type[BaseTool]] = {}

    def _walk(package_name: str, visited: Optional[Set[str]] = None) -> None:
        if visited is None:
            visited = set()
        if package_name in visited:
            return
        visited.add(package_name)

        try:
            pkg = importlib.import_module(package_name)
        except Exception:
            return

        # collect classes from this module
        for name in dir(pkg):
            obj = getattr(pkg, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
                and not getattr(obj, "__abstractmethods__", None)
                and hasattr(obj, "name")
                and obj.name
            ):
                discovered[obj.name] = obj

        # recurse into sub-packages
        if hasattr(pkg, "__path__"):
            for _, sub_name, is_pkg in pkgutil.walk_packages(
                pkg.__path__, f"{package_name}."
            ):
                if is_pkg:
                    _walk(sub_name, visited)
                else:
                    try:
                        mod = importlib.import_module(sub_name)
                        for name in dir(mod):
                            obj = getattr(mod, name)
                            if (
                                isinstance(obj, type)
                                and issubclass(obj, BaseTool)
                                and obj is not BaseTool
                                and not getattr(obj, "__abstractmethods__", None)
                                and hasattr(obj, "name")
                                and obj.name
                            ):
                                discovered[obj.name] = obj
                    except Exception:
                        continue

    _walk("weakagent.tools")
    logger.info("Discovered %d built-in tool classes", len(discovered))
    return discovered


def get_builtin_tool_registry(refresh: bool = False) -> Dict[str, type[BaseTool]]:
    """Return the (lazily-populated) global built-in tool registry."""
    global _BUILTIN_TOOL_REGISTRY
    if not _BUILTIN_TOOL_REGISTRY or refresh:
        _BUILTIN_TOOL_REGISTRY = _discover_builtin_tools()
    return _BUILTIN_TOOL_REGISTRY


# ── ToolCollection ────────────────────────────────────────────────────────

class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        valid: list[BaseTool] = []
        for tool in tools:
            try:
                tool.validate_schema()
                valid.append(tool)
            except ValueError as exc:
                logger.error("Skipping invalid tool %r at mount: %s", tool.name, exc)
        self.tools = tuple(valid)
        self.tool_map = {tool.name: tool for tool in self.tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for tool in self.tools:
            try:
                params.append(tool.to_params())
            except ValueError as exc:
                logger.error("Omitting tool %r from LLM request: %s", tool.name, exc)
        return params

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolExecutionResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolExecutionResult.fail(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolExecutionResult.fail(error=str(e))

    async def execute_all(self) -> List[ToolExecutionResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolExecutionResult.fail(error=str(e)))
        return results

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tool_map.get(name)

    def list_tool_catalog(self) -> Dict[str, List[Dict[str, Any]]]:
        """Summarize mounted tools and discoverable built-ins not yet mounted."""
        registry = get_builtin_tool_registry()
        mounted_names = set(self.tool_map)
        mounted = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "mounted": True,
            }
            for tool in self.tools
        ]
        available_to_add: List[Dict[str, Any]] = []
        for tool_name in sorted(registry):
            if tool_name in mounted_names:
                continue
            tool_cls = registry[tool_name]
            available_to_add.append(
                {
                    "name": tool_name,
                    "description": getattr(tool_cls, "description", "") or "",
                    "mounted": False,
                }
            )
        return {"mounted": mounted, "available_to_add": available_to_add}

    def add_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Instantiate a built-in tool by registry name and mount it.

        Returns the mounted tool, or the existing instance if already present.
        Returns None when the name is unknown or instantiation fails.
        """
        key = (name or "").strip()
        if not key:
            return None

        existing = self.tool_map.get(key)
        if existing is not None:
            return existing

        registry = get_builtin_tool_registry()
        tool_cls = registry.get(key)
        if tool_cls is None:
            logger.warning("Tool %s not found in built-in registry", key)
            return None

        try:
            tool = tool_cls()
            tool.validate_schema()
        except ValueError as exc:
            logger.warning("Failed schema validation for tool %s: %s", key, exc)
            return None
        except Exception as exc:
            logger.warning("Failed to instantiate tool %s: %s", key, exc)
            return None

        self.add_tool(tool)
        return tool

    def remount_tool_by_name(self, name: str, *, refresh_registry: bool = False) -> Optional[BaseTool]:
        """Re-instantiate a built-in tool by name and replace the mounted instance."""
        key = (name or "").strip()
        if not key:
            return None

        registry = get_builtin_tool_registry(refresh=refresh_registry)
        tool_cls = registry.get(key)
        if tool_cls is None:
            logger.warning("Tool %s not found in built-in registry for remount", key)
            return None

        try:
            tool = tool_cls()
            tool.validate_schema()
        except ValueError as exc:
            logger.warning("Failed schema validation for remount %s: %s", key, exc)
            return None
        except Exception as exc:
            logger.warning("Failed to remount tool %s: %s", key, exc)
            return None

        if key in self.tool_map:
            self.add_tool(tool, replace=True)
        else:
            self.add_tool(tool)
        return self.tool_map.get(key)

    def add_tool(self, tool: BaseTool, *, replace: bool = False) -> "ToolCollection":
        """Add a single tool to the collection.

        Parameters
        ----------
        tool:
            Tool instance to mount.
        replace:
            When True, replace an existing tool with the same name.

        Returns
        -------
        ToolCollection
            Self for chaining. Skips add when name exists and replace is False.

        Raises
        ------
        ValueError
            When the tool parameters schema is invalid for the LLM API.
        """
        try:
            tool.validate_schema()
        except ValueError as exc:
            raise ValueError(f"Cannot mount tool {tool.name!r}: {exc}") from exc

        if tool.name in self.tool_map:
            if replace:
                self.tools = tuple(
                    t if t.name != tool.name else tool for t in self.tools
                )
                self.tool_map[tool.name] = tool
                return self
            logger.warning("Tool %s already exists in collection, skipping", tool.name)
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """Add multiple tools to the collection.

        If any tool has a name conflict with an existing tool, it will be skipped and a warning will be logged.
        """
        for tool in tools:
            self.add_tool(tool)
        return self

    # ── Dynamic discovery ─────────────────────────────────────────────────

    @classmethod
    def discover(
        cls,
        *,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        refresh_registry: bool = False,
    ) -> "ToolCollection":
        """Dynamically discover and mount built-in tools.

        Scans ``weakagent.tools`` for all concrete :class:`BaseTool` subclasses
        and returns a ``ToolCollection`` containing their instances.

        Parameters
        ----------
        include:
            If given, only load tools whose ``name`` is in this list.
        exclude:
            If given, skip tools whose ``name`` is in this list.
        refresh_registry:
            Force re-scan of the package tree.

        Examples
        --------
        >>> # Load all discovered built-in tools
        >>> tools = ToolCollection.discover()

        >>> # Load only specific tools
        >>> tools = ToolCollection.discover(include=["terminate", "grep", "read_file"])

        >>> # Load all except some
        >>> tools = ToolCollection.discover(exclude=["create_chat_completion"])
        """
        registry = get_builtin_tool_registry(refresh=refresh_registry)
        instances: list[BaseTool] = []

        for tool_name, tool_cls in registry.items():
            if include is not None and tool_name not in include:
                continue
            if exclude is not None and tool_name in exclude:
                continue
            try:
                instances.append(tool_cls())
            except Exception as exc:
                logger.warning("Failed to instantiate tool %s: %s", tool_name, exc)

        logger.info(
            "ToolCollection.discover: loaded %d / %d discovered tools",
            len(instances),
            len(registry),
        )
        return cls(*instances)

    def discover_and_add(
        self,
        *,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        refresh_registry: bool = False,
    ) -> "ToolCollection":
        """Like :meth:`discover` but merges into *this* collection."""
        discovered = ToolCollection.discover(
            include=include,
            exclude=exclude,
            refresh_registry=refresh_registry,
        )
        for tool in discovered:
            self.add_tool(tool)
        return self
