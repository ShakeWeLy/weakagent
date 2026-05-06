import logging
import sys
import io
import os


class StandardFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class ColorFormatter(StandardFormatter):
    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"

    _FG_RED = "\x1b[31m"
    _FG_GREEN = "\x1b[32m"
    _FG_YELLOW = "\x1b[33m"
    _FG_CYAN = "\x1b[36m"
    _FG_MAGENTA = "\x1b[35m"
    _FG_WHITE = "\x1b[37m"

    def __init__(self) -> None:
        super().__init__()
        # Prefer colorama on Windows (better ANSI support).
        if os.name == "nt":
            try:
                import colorama  # type: ignore

                colorama.init()
            except Exception:
                pass

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
        base = StandardFormatter.format(self, record)
        return f"{self._level_color(record.levelname)}{base}{self._RESET}"

def get_logger(name="weakagent"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        stdout = sys.stdout
        if hasattr(stdout, "buffer"):
            stdout = io.TextIOWrapper(
                stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )

        # 控制台（彩色）
        console = logging.StreamHandler(stdout)
        console.setFormatter(ColorFormatter())

        # 文件（无颜色）
        file = logging.FileHandler("run.log", encoding="utf-8")
        file.setFormatter(StandardFormatter())

        logger.addHandler(console)
        logger.addHandler(file)

    return logger