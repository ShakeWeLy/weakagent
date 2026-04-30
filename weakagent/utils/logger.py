import logging
import os
from datetime import datetime


class _ColorFormatter(logging.Formatter):
    """
    Log formatter with fixed-width columns and ANSI colors.

    Output example:
    2026-04-29 16:56:13,640 | INFO    | weakagent.agent.toolcall      | ...
    """

    _LEVEL_WIDTH = 6
    _NAME_WIDTH = 30

    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"

    _FG_RED = "\x1b[31m"
    _FG_GREEN = "\x1b[32m"
    _FG_YELLOW = "\x1b[33m"
    _FG_CYAN = "\x1b[36m"
    _FG_MAGENTA = "\x1b[35m"
    _FG_WHITE = "\x1b[37m"
    _FG_GRAY = "\x1b[90m"

    def __init__(self) -> None:
        super().__init__()
        # Prefer colorama on Windows (better ANSI support).
        if os.name == "nt":
            try:
                import colorama  # type: ignore

                colorama.init()
            except Exception:
                pass

    @staticmethod
    def _format_ts(record: logging.LogRecord) -> str:
        # Fixed-width timestamp: "YYYY-MM-DD HH:MM:SS,mmm"
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"

    def _level_color(self, level: str) -> str:
        if level == "DEBUG":
            return self._DIM + self._FG_WHITE
        if level == "INFO":
            return self._FG_CYAN
        if level == "WARNING":
            return self._FG_YELLOW
        if level == "ERROR":
            return self._FG_RED
        if level == "CRITICAL":
            return self._FG_MAGENTA
        return self._FG_WHITE

    def format(self, record: logging.LogRecord) -> str:
        ts = self._format_ts(record)
        level = record.levelname
        name = record.name
        msg = record.getMessage()

        # Keep the separators aligned by forcing fixed-width columns.
        level_col = level.ljust(self._LEVEL_WIDTH)[: self._LEVEL_WIDTH]
        name_col = name.ljust(self._NAME_WIDTH)[: self._NAME_WIDTH]

        ts_color = self._DIM + self._FG_WHITE
        name_color = self._FG_WHITE
        is_toolcall = name.lower() == "weakagent.agent.toolcall"
        if is_toolcall:
            name_color = self._FG_GREEN
        if name.lower() == "weakagent.llm.llm":
            name_color = self._FG_GRAY
        prefix = (
            f"{ts_color}{ts}{self._RESET}"
            f" | {self._level_color(level)}{level_col}{self._RESET}"
            f" | {name_color}{name_col}{self._RESET}"
            f" | "
        )

        if is_toolcall:
            return prefix + f"{self._FG_GREEN}{msg}{self._RESET}"
        return prefix + msg

def get_logger(name="weakagent"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(_ColorFormatter())
        logger.addHandler(handler)

    return logger