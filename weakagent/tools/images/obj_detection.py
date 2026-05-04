from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

from PIL import Image, ImageDraw, ImageFont

from weakagent.config.settings import config
from weakagent.llm import LLM
from weakagent.schemas.message import Message
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.utils.image import image_to_base64


SYSTEM_PROMPT = """
You analyze images and detect objects. Reply with JSON only (optional markdown code fence).
Each item: "label" (short English name) and "bbox_2d": [x1, y1, x2, y2] where coordinates
are normalized in 0–1000 relative to image width (x) and height (y); x1,y1 top-left, x2,y2 bottom-right.
Example:
```json
[
  {"label": "example", "bbox_2d": [100, 200, 300, 400]}
]
```
"""

def extract_json_substring(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else text


def parse_json_items(raw: Union[str, List[Any], Dict[str, Any], None]) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    sub = extract_json_substring(str(raw))
    if not sub:
        return []
    data = json.loads(sub)
    if isinstance(data, list):
        return data
    return [data]


def list_bbox_detections(raw: Union[str, List[Any], Dict[str, Any], None]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in parse_json_items(raw):
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        bb = item["bbox_2d"]
        out.append(
            {
                "label": item.get("label"),
                "bbox_2d": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
            }
        )
    return out


def bbox_norm_to_pixels(
    bbox_2d: List[float],
    width: int,
    height: int,
    *,
    norm_max: float = 1000.0,
    clamp: bool = True,
) -> Tuple[int, int, int, int]:
    x1 = int(bbox_2d[0] / norm_max * width)
    y1 = int(bbox_2d[1] / norm_max * height)
    x2 = int(bbox_2d[2] / norm_max * width)
    y2 = int(bbox_2d[3] / norm_max * height)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if clamp and width > 0 and height > 0:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 0, 255),
    (255, 165, 0),
    (128, 0, 128),
    (0, 191, 255),
]


def draw_bbox_2d_normalized(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    *,
    norm_max: float = 1000.0,
    clamp: bool = True,
    line_width: int = 3,
) -> Image.Image:
    img = image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, det in enumerate(detections):
        bb = det["bbox_2d"]
        ax1, ay1, ax2, ay2 = bbox_norm_to_pixels(bb, w, h, norm_max=norm_max, clamp=clamp)
        color = _COLORS[i % len(_COLORS)]
        draw.rectangle(((ax1, ay1), (ax2, ay2)), outline=color, width=line_width)
        label = det.get("label")
        if label:
            draw.text((ax1 + 4, ay1 + 2), str(label), fill=color, font=font)
    return img


class ObjDetectionTool(BaseTool):
    name: str = "obj_detection"
    description: str = "Use when you need to detect objects in an image and return the position of the objects"
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The path to the image to detect objects in",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to use for the object detection, such as 'get the xxx object's position in the image'",
            },
        },
        "required": ["image_path", "prompt"],
    }

    async def execute(self, image_path: str, prompt: Optional[str] = None) -> ToolExecutionResult:
        if not config.llm["default"].supports_images:
            return self.fail_response("Model does not support images")

        path = Path(image_path)
        if not path.is_file():
            return self.fail_response(f"Image not found: {image_path}")
        
        if prompt is None:
            return self.fail_response("Prompt is required")

        llm = LLM(config_name="default")
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        url = image_to_base64(image_path)
        messages = [
            Message.system_message(SYSTEM_PROMPT),
            Message.user_message(
                prompt,
                base64_image=url,
            ),
        ]
        answer = await llm.ask(messages=messages)
        detections = list_bbox_detections(answer)

        enriched: List[Dict[str, Any]] = []
        for det in detections:
            x1, y1, x2, y2 = bbox_norm_to_pixels(det["bbox_2d"], w, h)
            row = {
                "label": det.get("label"),
                "bbox_2d": det["bbox_2d"],
                "bbox_pixel": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            enriched.append(row)

        out_pil = draw_bbox_2d_normalized(pil_img, detections)
        out_path = path.parent / f"{path.stem}_obj_det{path.suffix}"
        out_pil.save(out_path)

        return self.success_response(
            {
                "detections": enriched,
                "output_image_path": str(out_path.resolve()),
            }
        )
