"""
Long-term memory demo.

Flow:
1) LLM decides whether a user message is worth saving
2) Worthy items are persisted to sqlite (`long_term_memory` table)
3) Stored memories can be formatted for system-prompt injection

Run from project root (requires valid [llm] / [llm.fast] in config.toml):

    python examples/7_long_memory_demo.py
"""

from __future__ import annotations

import asyncio
import json

from weakagent.llm.llm import LLM
from weakagent.llm.summarize import extract_long_memory
from weakagent.memory.long import LongMemory


# One message likely worth saving, one casual message that should be skipped.
SAMPLES_1 = [
  {
    "role": "system",
    "content": "你是 AI 助手。"
  },

  {
    "role": "user",
    "content": "我最近在做 YOLO 剪枝，推理速度一直提不上去。"
  },
  {
    "role": "assistant",
    "content": "剪枝后速度不提升，可能是因为算子结构没有真正减少，或者 TensorRT 没有成功优化。"
  },

  {
    "role": "user",
    "content": "我主要做工业视觉方向。"
  },
  {
    "role": "assistant",
    "content": "工业视觉通常会比较关注延迟、稳定性和部署成本。"
  },

  {
    "role": "user",
    "content": "今天 CUDA 又报错了，烦死。"
  },
  {
    "role": "assistant",
    "content": "先看看是驱动版本问题还是显存不足。"
  },

  {
    "role": "user",
    "content": "以后回答尽量深入一些，我不喜欢太浅。"
  },
  {
    "role": "assistant",
    "content": "好的，后面我会偏向原理和工程实现细节。"
  }
]

SAMPLES_2 = [
  {
    "role": "system",
    "content": "你是一个智能助手。"
  },
  {
    "role": "user",
    "content": "今天天气怎么样？"
  },
  {
    "role": "assistant",
    "content": "台北今天多云，气温约 24–28°C，有轻微降雨概率。"
  }
]

SAMPLES = [
    str(SAMPLES_1),
    str(SAMPLES_2),
]


async def main() -> None:
    llm = LLM(config_name="fast")
    user_id = "demo_user"

    print("=== Long memory demo ===\n")
    mem = LongMemory(user_id=user_id)

    for i, text in enumerate(SAMPLES, 1):
        print(f"[{i}] User: {text}")

        # Step 1: extraction only (no DB write)
        extracted = await extract_long_memory(llm, text)
        print(f"    extract: {json.dumps(extracted, ensure_ascii=False)}")

        # Step 2: extract + persist when should_save
        saved = await mem.extract_and_save(text, llm=llm)
        if saved.get("should_save"):
            print(f"    saved:  ({saved['memory_type']}) {saved['memory']}")
        else:
            print("    saved:  (skipped)")
        print()

    print("--- All entries for user ---")
    for e in mem.load_for_user(user_id):
        print(f"  [{e.memory_type} | {e.importance:.2f}] {e.content}")

    print("\n--- System context snippet ---")
    print(mem.to_system_context())


if __name__ == "__main__":
    asyncio.run(main())
