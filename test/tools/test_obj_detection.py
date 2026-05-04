import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weakagent.config.settings import config
from weakagent.tools.images.obj_detection import ObjDetectionTool


IMAGE_PATH = PROJECT_ROOT / "Snipaste_2026-04-12_21-25-08.jpeg"

USER_PROMPT = (
    "识别图中的四个蓝色矩形框，输出每个框的 bbox_2d（0–1000 归一化）与简短中文 label。"
)


async def main():
    if not config.llm["default"].supports_images:
        raise SystemExit(
            "config.toml 里 [llm] 需要 supports_images = true（本测试走 default 模型多模态）。"
        )

    if not IMAGE_PATH.is_file():
        raise SystemExit(
            f"缺少测试图片，请将截图放到仓库根目录: {IMAGE_PATH}"
        )

    tool = ObjDetectionTool()
    result = await tool.execute(str(IMAGE_PATH), USER_PROMPT)

    print(result.output or result.error)
    if result.data:
        print(json.dumps(result.data, indent=2, ensure_ascii=False))

    assert result.success, result.error or result.output
    assert result.data is not None
    out = Path(result.data["output_image_path"])
    assert out.is_file(), f"expected annotated image at {out}"

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
