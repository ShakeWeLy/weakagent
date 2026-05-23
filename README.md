# weakagent

轻量级多 Agent 运行时：支持自定义 Agent / Tool、父子 Agent 编排、SQLite 会话与定时任务调度。

## 快速开始

```bash
pip install -e .          # 开发安装
# 或: pip install dist/weakagent-<version>-py3-none-any.whl
```

复制并编辑项目根目录的 `config.toml`（至少配置 `[llm]` 的 `api_key`）。入口示例见 `main.py`。

---

## 1. 创建自定义 Agent

两种方式：**继承 Agent 类** + **向 Factory 注册类型**。

### 方式 A：简单 Agent（实现 `step`）

适合不依赖 LLM 工具链的演示或固定逻辑子任务。

```python
from weakagent.agent import AgentFactory, AgentRuntime, AgentSpec, BaseAgent
from weakagent.schemas.agent import AgentState

class EchoAgent(BaseAgent):
    async def step(self) -> str:
        # 从 messages 取最近 user 消息并返回
        self.state = AgentState.FINISHED
        return "handled"

factory = AgentFactory()
factory.register_spec(
    "echo",
    AgentSpec(
        agent_cls=EchoAgent,
        default_kwargs={"name": "echo_agent", "max_steps": 1},
        description="Echo demo agent",
    ),
)
runtime = await AgentRuntime.instance(factory=factory)
agent_id = runtime.create_agent("echo", name="my_echo")
```

完整示例：`examples/1_agent_runtime_demo.py`。

### 方式 B：ReAct / ToolCall Agent（挂载工具）

适合需要 LLM + 工具调用的对话型 Agent。

```python
from weakagent.agent import BriefReActAgent, AgentFactory, AgentRuntime, AgentSpec
from weakagent.tools import ToolCollection, Terminate, AskHumanTool

class ChatToolcallAgent(BriefReActAgent):
    name = "chat_toolcall"
    available_tools = ToolCollection(Terminate(), AskHumanTool())
    max_steps = 100

factory = AgentFactory()
factory.register_spec("chat_toolcall", AgentSpec(agent_cls=ChatToolcallAgent, ...))
```

项目入口：`main.py`（`ChatToolcallAgent` + `run_loop` 交互）。

内置 Factory 类型：`chat`、`toolcall`、`brief_react` / `react`、`multi_react`。自定义类型通过 `register_spec` 追加。

---

## 2. 使用 AgentRuntime

`AgentRuntime` 是**单例**，负责注册表、父子关系、同步/后台运行与清理。

```python
runtime = await AgentRuntime.instance(factory=factory)  # 或默认 AgentFactory()

# 创建并注册
agent_id = runtime.create_agent("echo", name="parent", system_prompt="...")
child_id = runtime.spawn_sub_agent(parent_id=agent_id, agent_type="echo", name="child")

# 已有实例也可 register
parent_id = runtime.register(my_agent, agent_type="multi_react")

# 运行
result = await runtime.run(agent_id, request="你好")           # 单次，阻塞到结束
await runtime.run_loop(agent_id)                               # 交互式 input 循环
task = runtime.run_in_background(agent_id, request="...")    # 后台 Task
await task

# 查询与清理
runtime.list_agents(parent_id=agent_id)
runtime.get_registered_agents()
await runtime.cleanup_all()
```

| 模式 | API | 说明 |
|------|-----|------|
| 单次 | `run(agent_id, request=...)` | 跑完一轮返回 `last_result` |
| 交互 | `run_loop(agent_id)` | 终端 `You>` 循环，结束时会话写入 `runtime_memory` |
| 后台 | `run_in_background` | `asyncio.gather` 并发多 Agent |
| 队列 | `start_queue_loop` + `put_request` / `get_result` | 见定时任务 `TaskCrudAgent` |

多 Agent 编排（`create_sub_agent` / `run_sub_agent`）：`examples/0_runtime_demo.py`、`examples/2_multi_react_create_then_run_demo.py`。

---

## 3. 新增自定义 Tool（Demo）

1. 继承 `BaseTool`，定义 `name`、`description`、`parameters`（OpenAI function schema）。
2. 实现 `async def execute(self, **kwargs) -> ToolExecutionResult`。
3. 放入 Agent 的 `ToolCollection(...)`。

```python
from weakagent.tools.base import BaseTool, ToolExecutionResult

class HelloTool(BaseTool):
    name = "hello"
    description = "Say hello to someone."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    async def execute(self, name: str) -> ToolExecutionResult:
        return self.success_response(f"Hello, {name}!")

# 在 Agent 中:
available_tools = ToolCollection(HelloTool(), Terminate())
```

参考实现：`weakagent/tools/terminate.py`。调度类工具见 `weakagent/tools/scheduler/`（需先 `set_task_store(store)` 绑定 `TaskStore`）。

---

## 4. 定时任务（Scheduler）

核心组件：

- **TaskStore**：SQLite 存任务（`config.toml` → `[scheduler].db_path`）
- **TaskRegistry + Dispatcher**：`task_type` → 执行器（如 `daily_summary`）
- **SchedulerRunner**：后台线程周期扫描 `pending` 任务并派发

```python
from weakagent.scheduler import TaskStore, TaskRegistry, Dispatcher, Scheduler, SchedulerRunner
from weakagent.scheduler.executors import DailySummaryExecutor

store = TaskStore(db_path="weakagent.sqlite3")
registry = TaskRegistry()
registry.register("daily_summary", DailySummaryExecutor)
scheduler = Scheduler(store=store, dispatcher=Dispatcher(registry=registry, store=store))

runner = SchedulerRunner(scheduler, interval=2.0)
runner.start()
# ... 稍后 runner.stop()
```

**三种使用方式**（见 `examples/5_scheduler_demo.py`）：

1. **Runner 线程**：`store.create_task("daily_summary", payload={...})` + `SchedulerRunner`
2. **CRUD 工具**：`CreateTaskTool` / `ListTasksTool` / `UpdateTaskTool` 等（Agent 或脚本直接 `execute`）
3. **TaskCrudAgent + 队列**：`runtime.start_queue_loop` + JSON `put_request`（`action`: create/list/update/delete）

自定义任务：实现 `BaseExecutor`，`registry.register("your_type", YourExecutor)`。

---

## 5. Memory 记了什么？

`BaseAgent` 上有多层记忆，职责不同：

| 类型 | 类 | 存什么 | 持久化 |
|------|-----|--------|--------|
| **短期** | `ShortMemory` | 当前 `run` 内的完整对话（user/assistant/tool），含工具链 | 内存，跑完可清空 |
| **会话** | `ConversationMemory` | 可选；完整消息历史、`session_id`、标题等 | SQLite `conversation_*` 表（`[conversation].db_path`） |
| **运行时** | `RuntimeMemory` | 每轮 **request + last_result**（不含中间 tool 细节） | SQLite `runtime_session*` 表；`run_loop` 结束可 `finalize_session` 摘要 |
| **工作** | `WorkingMemory` | 供摘要用的消息列表 | 内存，可 `summarize()` 压缩 |

**清理策略**（`BaseMemory.cleanup_if_needed`）：按轮次保留最近 N 轮、截断过长 tool 输出、超 token 窗口时丢弃旧轮或 LLM 摘要。配置项见 `keep_last_n`、`max_context_turns`、`cleanup_strategy` 等。

数据流简述：`runtime_memory` 在 `run` 开始时加载进 `short_memory`；单轮结束后把 `last_result` 写回 `runtime_memory`；`conversation` 在启用时逐条落库。

---

## 6. `config.toml` 简要配置

路径相对于**项目根**（`config.toml` 所在目录）。

```toml
# 默认 LLM（也可用 [llm.fast]、[llm.deepseek.pro] 等命名配置，Agent 里 config_name="fast"）
[llm]
model = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"
api_key = "YOUR_API_KEY"
max_tokens = 8192
temperature = 0.0

# 定时任务库
[scheduler]
db_path = "weakagent.sqlite3"

# 长会话库（可与 scheduler 共用同一文件）
[conversation]
db_path = "weakagent.sqlite3"

# 联网搜索（web_search 等）
[search]
engine = "bing"          # Google | Baidu | DuckDuckGo | Bing
fallback_engines = []
retry_delay = 15
max_retries = 2
lang = "zh"
country = "cn"
```

**说明：**

- 多模型：增加 `[llm.xxx]` 段，创建 `LLM(config_name="xxx")` 或 Agent 构造参数传入。
- Ollama：取消注释 `[llm]` 中 `api_type = 'ollama'` 示例并改 `base_url`。
- 勿将真实 `api_key` 提交到 Git；可用环境变量或本地未跟踪的 `config.local.toml`（若项目支持）。

---

## Examples 索引

| 文件 | 内容 |
|------|------|
| `examples/0_runtime_demo.py` | `multi_react` + `run_sub_agent` |
| `examples/1_agent_runtime_demo.py` | 自定义 `EchoAgent` + 父子并发 |
| `examples/2_multi_react_create_then_run_demo.py` | create → run 子 Agent |
| `examples/5_scheduler_demo.py` | 定时任务三种用法 |
| `examples/7_long_memory_demo.py` | 长期记忆提取与 SQLite 持久化 |
| `examples/8_skills_demo.py` | Skills 目录加载、系统提示注入、`read` 读 SKILL.md |
| `main.py` | 可运行聊天 Agent（`run_loop`） |

---

## 手动打包

在**项目根目录**（含 `pyproject.toml` 的目录）执行以下步骤。

### 1. 环境

- Python **3.10+**（`pyproject.toml` 要求）
- 建议使用虚拟环境，避免污染系统 Python

```bash
python -m venv .venv
# Windows (cmd):
# .venv\Scripts\activate
# Windows (Git Bash) / Linux / macOS:
source .venv/Scripts/activate   # Git Bash on Windows
# 或: source .venv/bin/activate
```

### 2. 安装打包工具

任选其一：

**方式 A：`build`（官方推荐，生成 sdist + wheel）**

```bash
pip install build
```

**方式 B：`uv`（若已安装 uv）**

```bash
# 无需单独装 build，直接构建
uv build
```

### 3. 执行构建

**使用 `build`：**

```bash
python -m build
```

默认会生成：

- `dist/<项目名>-<版本>.tar.gz`：源码包（sdist）
- `dist/<项目名>-<版本>-py3-none-any.whl`：内置发行包（wheel）

**仅打 wheel 或仅打 sdist（可选）：**

```bash
python -m build --wheel      # 只要 wheel
python -m build --sdist      # 只要 sdist
```

**使用 `uv`：**

```bash
uv build
```

产物同样在 `dist/` 目录下。

### 4. 清理后重新打包（可选）

若曾构建过，可先删除再打包，避免混用旧文件：

```bash
rm -rf build dist *.egg-info
# Windows PowerShell: Remove-Item -Recurse -Force build,dist,*egg-info -ErrorAction SilentlyContinue
```

然后重复第 3 步。

### 5. 本地验证安装

从 wheel 安装到当前环境：

```bash
pip install dist/weakagent-0.2.5-py3-none-any.whl
```

版本号以 `pyproject.toml` 里 `[project]` 的 `version` 为准，请替换为实际生成的 wheel 文件名。

或从源码包安装：

```bash
pip install dist/weakagent-0.2.5.tar.gz
```

开发阶段也可**可编辑安装**（不生成 `dist`，直接链到源码）：

```bash
pip install -e .
```

---

**说明：** 发布到 PyPI 时，除上述构建外，还需配置 PyPI 账号与 `twine upload dist/*` 等步骤；仅本地或内网分发时，分发 `dist/` 下的文件即可。
