---
name: tool-usage-skill
description: >
  解释如何在 weakagent 中动态使用工具：发现可用工具、动态挂载/热加载、
  创建子代理并注入工具集，以及工具系统的整体架构。
metadata:
  weakagent:
    default_enabled: true
---

# Tool 动态使用指南

## 1. 工具发现 (Tool Discovery)

### 1.1 使用 list_tools 工具

`list_tools` 列出当前 agent **已挂载** 的工具和 **可添加** 的内置工具。

```
list_tools()
```

返回两部分：
- **Mounted tools**  — 当前 agent 可直接调用的工具
- **Available to add (built-in)**  — 注册在 `_BUILTIN_TOOL_REGISTRY` 中但未挂载的工具

### 1.2 底层发现机制

```python
# 扫描 weakagent.tools 包树，收集所有 BaseTool 子类
from weakagent.tools.tool_collection import get_builtin_tool_registry
registry = get_builtin_tool_registry()           # 懒加载
registry = get_builtin_tool_registry(refresh=True)  # 强制重新扫描
```

### 1.3 ToolCollection 发现

```python
from weakagent.tools.tool_collection import ToolCollection

# 加载所有发现的内置工具
tc = ToolCollection.discover()

# 只加载指定工具
tc = ToolCollection.discover(include=["terminate", "grep", "read"])

# 排除某些工具
tc = ToolCollection.discover(exclude=["create_chat_completion"])

# 合并到已有集合
existing_collection.discover_and_add(exclude=["bash"])
```

---

## 2. 工具挂载与热加载

### 2.1 运行时添加工具

`ToolCollection` 提供三个核心操作方法：

```python
# 按名称实例化并挂载内置工具
tc.add_tool_by_name("bash")

# 直接挂载工具实例
tc.add_tool(some_tool_instance)

# 替换已存在的同名工具
tc.add_tool(new_tool_instance, replace=True)

# 批量添加
tc.add_tools(tool1, tool2, tool3)
```

### 2.2 remount — 重新实例化替换

当工具的 Python 代码发生修改后，用 `remount_tool_by_name` 重新实例化并替换：

```python
# 从注册表中重新实例化工具，替换已挂载的同名工具
tc.remount_tool_by_name("hot_reload")
tc.remount_tool_by_name("hot_reload", refresh_registry=True)  # 先刷新注册表
```

### 2.3 hot_reload — 热加载 Python 模块

`hot_reload` 工具是开发期元工具，通过 `importlib.reload()` 重新加载模块：

```python
# 基本用法：重新加载指定模块
hot_reload(module_names=["weakagent.tools.tool.hot_reload"])

# 支持简写路径
hot_reload(module_names=["tools.memory.long", "tools.sub_agent.create_sub_agent"])

# 加载后重新挂载受影响的工具
hot_reload(
    module_names=["weakagent.tools.tool.hot_reload"],
    remount_tools=["hot_reload"]
)

# 使用预设：重新加载一组常用工具模块
hot_reload(use_tool_defaults=True)
```

**执行流程：**

```
hot_reload()
  ├─ reload_modules(names)          # importlib.reload 逐个模块
  │    ├─ normalize_module_name()    # 将短路径转为 weakagent.xxx
  │    └─ importlib.reload(mod)      # 实际重载
  ├─ get_builtin_tool_registry(refresh=True)  # 刷新注册表
  └─ collection.remount_tool_by_name()        # 重新挂载工具实例
```

**注意：** 热加载只刷新 Python 模块对象，正在运行的 asyncio 任务、pydantic 模型类、单例仍可能持有旧引用。推荐配合 `remount_tools` 参数一起使用。

---

## 3. 子代理工具注入

通过 `create_sub_agent` 的 `tools` 参数，可以指定子代理拥有哪些工具：

```python
# 创建一个只包含 terminate 和 grep 的子代理
create_sub_agent(
    sub_agent_name="toolcall",
    tools=["terminate", "grep"]
)

# 创建具备文件读写能力的子代理
create_sub_agent(
    sub_agent_name="toolcall",
    tools=["terminate", "grep", "list_files", "patch_file", "write_file"]
)

# 包含热加载能力的子代理
create_sub_agent(
    sub_agent_name="toolcall",
    tools=["terminate", "hot_reload", "create_sub_agent", "run_sub_agent"]
)
```

### 支持的工具有限列表：

| 工具名 | 说明 |
|--------|------|
| `terminate` | 终止交互（必选） |
| `run_sub_agent` | 运行子代理 |
| `create_sub_agent` | 创建子代理 |
| `hot_reload` | 热加载模块 |
| `bash` | 执行 Bash 命令 |

> **注意：** `create_sub_agent` 内部使用 `_build_tool_collection()` 构建 `ToolCollection`，仅支持上述白名单内的工具名。其余工具（`grep`, `list_files`, `patch_file`, `read`, `write_file` 等）通过 `available_tools` 动态发现机制挂载，不需要显式指定。

---

## 4. 工具系统架构

### 4.1 类层次

```
BaseTool (ABC)                  — weakagent/tools/base.py
  ├── name: str                 — 工具名称（唯一标识）
  ├── description: str          — 工具描述
  ├── parameters: dict          — JSON Schema 参数定义
  ├── execute(**kwargs)         — 核心执行逻辑
  └── execute_for_agent(...)    — 需要 agent 上下文的执行

ToolCollection                  — weakagent/tools/tool_collection.py
  ├── tools: tuple[BaseTool]    — 已挂载的工具列表
  ├── tool_map: dict[str, Tool] — 名称→工具 的映射
  ├── to_params()               — 转为 LLM API 参数格式
  ├── get_tool(name)            — 按名称获取工具
  ├── add_tool(tool, replace)   — 添加/替换工具
  ├── add_tool_by_name(name)    — 按注册名实例化并添加
  ├── remount_tool_by_name(...) — 重新实例化并替换
  ├── discover(...)             — 自动发现并加载内置工具
  └── list_tool_catalog()       — 列出已挂载和可添加的工具
```

### 4.2 工具注册流程

```
Python 模块扫描
  └─ _discover_builtin_tools()
       ├─ 遍历 weakagent.tools 包树
       ├─ 收集所有 BaseTool 非抽象子类
       └─ 存入 _BUILTIN_TOOL_REGISTRY

Agent 初始化
  └─ ToolCollection(*tools)
       ├─ validate_schema() 校验参数
       └─ 存入 self.tools / self.tool_map

Agent 工具循环
  └─ available_tools.execute(name, tool_input)
       ├─ tool_map[name] 查找工具
       └─ tool(**tool_input) 执行
```

---

## 5. 典型工作流

### 5.1 探索可用工具

```
1. 调用 list_tools()
2. 查看 Mounted tools 和 Available to add
3. 如需添加：使用 add_tool_by_name（代码层）或通过 agent 初始化配置
```

### 5.2 开发新工具 → 热加载

```
1. 在 weakagent/tools/xxx.py 中编写 BaseTool 子类
2. 使用 hot_reload(module_names=["tools.xxx"]) 重新加载
3. 使用 remount_tool_by_name("xxx") 重新挂载
4. 调用工具验证新逻辑
```

### 5.3 创建专用子代理

```
1. 定义子代理需要的工具列表
2. 使用 create_sub_agent(tools=[...]) 创建
3. 使用 run_sub_agent(sub_agent_id=...) 运行
```

---

## 6. 常见问题

**Q: list_tools 看到工具但无法调用？**
> 工具必须在 agent 的 `available_tools` (ToolCollection) 中挂载。通过 `add_tool_by_name()` 或初始化时传入。

**Q: hot_reload 后工具行为未更新？**
> 模块重载只刷新 Python 对象，已实例化的工具仍持有旧引用。需要同时传入 `remount_tools` 参数重新挂载。

**Q: 自定义工具不在 Available to add 中？**
> 确保工具类在 `weakagent.tools` 包树下，是 `BaseTool` 的非抽象子类，且具有唯一的 `name` 属性。
