# 热拔插 MOD 插件库开发规范

> **版本**: 1.0.0  
> **适用系统**: AGI Agent 插件系统  
> **最后更新**: 2026-07-05

---

## 目录

1. [概述](#1-概述)
2. [插件架构](#2-插件架构)
3. [插件生命周期](#3-插件生命周期)
4. [接口定义](#4-接口定义)
5. [数据格式规范](#5-数据格式规范)
6. [钩子系统](#6-钩子系统)
7. [依赖管理](#7-依赖管理)
8. [错误处理](#8-错误处理)
9. [版本兼容性](#9-版本兼容性)
10. [最佳实践](#10-最佳实践)
11. [示例插件](#11-示例插件)

---

## 1. 概述

本规范定义了 AGI Agent 热拔插插件系统的开发标准、接口约定和兼容性要求。所有插件必须严格遵循本规范，以确保系统与插件的高适配率和最小化兼容性问题。

### 1.1 设计原则

- **完全动态加载**: 插件无需修改系统主体代码和前端代码，放置到指定目录即可被自动发现
- **热拔插**: 支持运行时加载、激活、停用、卸载、重载，无需重启服务
- **零侵入**: 插件与系统通过标准接口交互，互不干扰
- **资源安全**: 卸载时必须完全释放所有资源，无内存泄漏
- **错误隔离**: 单个插件故障不影响系统主体和其他插件

### 1.2 系统版本

| 系统版本 | 兼容插件版本 | 说明 |
|---------|-------------|------|
| 1.0.0   | 1.0.0       | 初始版本 |

---

## 2. 插件架构

### 2.1 文件结构

```
agi_agent/
├── plugins/
│   ├── __init__.py          # 插件包导出
│   ├── plugin_base.py       # 插件基类（PeripheralPlugin）
│   ├── plugin_manager.py    # 插件管理器
│   └── mods/                # 插件放置目录
│       ├── noise_sensor.py  # 示例：噪声传感器
│       ├── temperature_sensor.py
│       └── data_processor.py
```

### 2.2 插件文件命名

- 文件名: `{plugin_name}.py`
- 全小写，使用下划线分隔
- 不得以下划线开头（`_` 开头的文件会被忽略）
- 文件名与插件内部 `name` 属性建议保持一致

### 2.3 类结构

每个插件文件必须包含以下之一：
- 名为 `Plugin` 的类，继承 `PeripheralPlugin`
- 名为 `create_plugin` 的工厂函数，返回 `PeripheralPlugin` 实例

**推荐使用工厂函数**，便于在构造时进行复杂初始化。

---

## 3. 插件生命周期

```
  发现(discovered)
      │
      ▼
  加载(on_load) ───► 已加载(loaded)
      │                    │
      │                    ├─► 激活(on_activate)
      │                    │      │
      │                    │      ▼
      │                    │   活跃(active)
      │                    │      │
      │                    │   停用(on_deactivate)
      │                    │      │
      │                    ▼      ▼
      └──────────────► 卸载(on_unload)
                           │
                           ▼
                      已释放(freed)
```

### 3.1 生命周期阶段详解

| 阶段 | 触发时机 | 对应方法 | 状态 | 必须完成的工作 |
|------|---------|---------|------|--------------|
| 发现 | 文件放入 mods 目录 | 无 | discovered | 系统自动扫描，元信息检查 |
| 加载 | 调用 load_plugin() | `on_load()` | loaded | 初始化资源、读取配置、建立连接 |
| 激活 | 调用 activate_plugin() | `on_activate()` | active | 启动后台线程、开启数据流 |
| 停用 | 调用 deactivate_plugin() | `on_deactivate()` | loaded | 暂停后台线程、停止数据流 |
| 卸载 | 调用 unload_plugin() | `on_unload()` | unloaded | **彻底释放所有资源** |

### 3.2 状态枚举

```python
class PluginStatus(Enum):
    UNLOADED = "unloaded"   # 未加载
    LOADED = "loaded"       # 已加载，未激活
    ACTIVE = "active"       # 活跃，正常工作
    ERROR = "error"         # 错误状态
```

---

## 4. 接口定义

### 4.1 必须实现的方法（4个）

#### 4.1.1 on_load()

```python
def on_load(self) -> bool:
    """插件加载时调用。
    
    Returns:
        bool: 加载成功返回 True，失败返回 False。
              返回 False 时系统会拒绝加载该插件。
    """
```

**用途**: 初始化资源、读取配置、建立连接、分配内存。

**注意事项**:
- 必须是幂等的（多次调用结果一致）
- 失败时必须自行清理已分配的资源
- 不得阻塞主线程超过 5 秒
- 需要后台线程的，在此处创建但**不要启动**

---

#### 4.1.2 on_unload()

```python
def on_unload(self) -> bool:
    """插件卸载时调用。
    
    Returns:
        bool: 卸载成功返回 True，失败返回 False。
              即使返回 False，系统也会强制清理模块引用。
    """
```

**用途**: 释放所有资源。**这是最重要的方法，必须彻底释放。**

**必须释放的资源清单**:
- [ ] 所有线程（设置停止标志并 join）
- [ ] 所有网络连接（socket、HTTP 客户端等）
- [ ] 所有文件句柄（确保 close）
- [ ] 所有数据库连接
- [ ] 所有定时器/调度器
- [ ] 所有缓存数据（清空 list/dict/deque）
- [ ] 所有外部进程（如果启动了子进程）
- [ ] 所有注册的回调/监听器
- [ ] 大型 numpy/torch 张量（设为 None 触发 GC）

**反模式（禁止）**:
```python
# ❌ 错误：什么都不做
def on_unload(self):
    return True

# ❌ 错误：仅释放部分资源
def on_unload(self):
    self.data = None  # 线程还在跑！
    return True
```

---

#### 4.1.3 process(input_data)

```python
def process(self, input_data: Any) -> Any:
    """处理输入数据。
    
    Args:
        input_data: 输入数据。可以是 numpy.ndarray、dict 或其他类型，
                   具体取决于插件注册的钩子点。
    
    Returns:
        Any: 处理结果。将被注入到智能体的对应模块。
             如果是数据处理类插件，应返回与输入类型一致的数据。
    """
```

**用途**: 插件的核心功能实现。

**注意事项**:
- 必须是线程安全的
- 执行时间应尽可能短（建议 < 10ms）
- 不得修改输入数据的原始引用（先 copy 再修改）
- 异常必须在内部捕获或允许向上传播（系统会捕获并记录错误计数）

---

#### 4.1.4 get_data()

```python
def get_data(self) -> Dict[str, Any]:
    """获取插件当前数据状态。
    
    Returns:
        Dict[str, Any]: 插件当前状态的字典表示。
                        必须可 JSON 序列化。
    """
```

**用途**: 向系统和前端暴露插件内部状态。

**返回值要求**:
- 必须是字典类型
- 所有值必须可 JSON 序列化
- 不得包含循环引用
- 建议包含：当前值、历史统计、状态标志等

---

### 4.2 可选实现的方法

#### 4.2.1 on_activate()

```python
def on_activate(self) -> bool:
    """插件激活前调用。返回 False 则激活失败。"""
```

用于：启动后台线程、开始数据流、打开硬件设备等。

#### 4.2.2 on_deactivate()

```python
def on_deactivate(self) -> bool:
    """插件停用后调用。返回 False 不影响状态迁移。"""
```

用于：暂停后台线程、停止数据流、关闭硬件设备等。

#### 4.2.3 on_structure_change(new_dim)

```python
def on_structure_change(self, new_dim: int) -> bool:
    """系统结构维度变化时调用。"""
```

用于：调整插件内部的数组维度、网络层大小等以适配新结构。

---

### 4.3 构造函数参数

插件的 `__init__` 方法不得要求任何参数（使用工厂函数时），或者必须有默认值（使用类继承时）。

所有配置通过 `self.config` 字典传递，由系统在加载后注入。

---

## 5. 数据格式规范

### 5.1 输入数据类型

| 钩子点 | 输入类型 | 说明 |
|--------|---------|------|
| PRE_PERCEPTION | numpy.ndarray | 原始观测向量 |
| POST_PERCEPTION | numpy.ndarray | 感知模块输出的特征 |
| PRE_COGNITION | numpy.ndarray | 认知模块输入特征 |
| POST_COGNITION | dict | 认知推理结果 |
| PRE_ACTION | numpy.ndarray | 动作生成前的状态 |
| POST_ACTION | numpy.ndarray | 最终动作向量 |
| PERIODIC | dict | 周期触发，含当前 step |
| ON_STRUCTURE_CHANGE | int | 新的维度值 |

### 5.2 输出数据格式

- **传感器类插件**: 返回 numpy.ndarray，维度与输入一致或追加新维度
- **处理类插件**: 返回与输入相同类型的数据
- **分析类插件**: 返回 dict，包含分析结果

### 5.3 返回值约束

- numpy 数组的 dtype 必须是 float32 或 float64
- 字典的键必须是字符串
- 嵌套深度不超过 3 层
- 不得包含自定义类实例（使用基本类型）

---

## 6. 钩子系统

### 6.1 钩子点定义

```python
class PluginHookPoint(Enum):
    PRE_PERCEPTION = "pre_perception"       # 感知前
    POST_PERCEPTION = "post_perception"     # 感知后
    PRE_COGNITION = "pre_cognition"         # 认知前
    POST_COGNITION = "post_cognition"       # 认知后
    PRE_ACTION = "pre_action"               # 动作前
    POST_ACTION = "post_action"             # 动作后
    PERIODIC = "periodic"                   # 周期触发
    ON_STRUCTURE_CHANGE = "on_structure_change"  # 结构变更
```

### 6.2 注册钩子

在构造函数中声明：

```python
def __init__(self):
    super().__init__(
        ...
        hook_points=[PluginHookPoint.PRE_PERCEPTION, PluginHookPoint.PRE_COGNITION]
    )
```

### 6.3 实现钩子方法

钩子方法命名规则：`hook_{hook_name}`

```python
def hook_pre_perception(self, input_data):
    """感知前处理。返回处理后的数据。"""
    return self.process(input_data)

def hook_pre_cognition(self, input_data):
    """认知前处理。"""
    return input_data
```

如果未实现对应的 `hook_xxx` 方法，则该钩子点不会被调用（即使注册了也会被跳过）。

### 6.4 执行顺序

- 同一钩子点的多个插件按 `priority` 从高到低执行
- 前一个插件的输出作为下一个插件的输入（链式调用）
- 高优先级插件先执行，低优先级后执行

---

## 7. 依赖管理

### 7.1 声明依赖

在构造函数的 `dependencies` 参数中声明：

```python
def __init__(self):
    super().__init__(
        ...
        dependencies=["base_sensor", "data_processor"]
    )
```

### 7.2 依赖规则

- 依赖的插件必须已经加载（不要求激活）
- 依赖不满足时，系统会拒绝加载当前插件
- 不支持循环依赖（A 依赖 B，B 不能再依赖 A）
- 不支持可选依赖（要么全满足，要么不加载）

### 7.3 外部 Python 包依赖

插件需要的第三方 Python 包，需在插件文件顶部 `import`，若未安装会导致加载失败。

**建议**: 在 `on_load` 中检查依赖是否可用，给出明确错误信息，而不是让 import 直接崩溃。

```python
def on_load(self) -> bool:
    try:
        import serial
        self.serial_lib = serial
    except ImportError:
        self._last_error = "缺少依赖库: pyserial，请先安装 pip install pyserial"
        return False
    return True
```

---

## 8. 错误处理

### 8.1 错误计数

系统会自动统计插件的错误次数，记录在 `plugin._error_count` 和 `plugin._last_error` 中。

### 8.2 错误处理原则

1. **插件内部错误**: 尽可能自行捕获并恢复，不得让异常蔓延到系统层
2. **致命错误**: 无法恢复时，设置 `self.status = PluginStatus.ERROR`
3. **资源泄漏防护**: 无论是否发生错误，`on_unload` 都必须保证资源释放

### 8.3 推荐的错误处理模式

```python
def process(self, input_data):
    try:
        # 核心逻辑
        result = self._do_work(input_data)
        return result
    except DeviceNotFoundError as e:
        self._last_error = f"设备未找到: {e}"
        self._error_count += 1
        # 降级：返回原始数据，不影响系统
        return input_data
    except Exception as e:
        self._last_error = str(e)
        self._error_count += 1
        return input_data
```

---

## 9. 版本兼容性

### 9.1 版本号规范

采用语义化版本号：`主版本.次版本.修订版本`

- **主版本**: 不兼容的接口变更
- **次版本**: 向下兼容的功能新增
- **修订版本**: 向下兼容的问题修复

### 9.2 声明兼容版本

```python
def __init__(self):
    super().__init__(
        ...
        compatible_versions=["1.0.0", "1.1.0", "1.2.0"]
    )
```

### 9.3 兼容性检查

系统加载插件时会检查：
- 插件声明的 `compatible_versions` 中是否包含当前系统版本
- 不包含则拒绝加载，并给出明确错误信息

### 9.4 废弃策略

- 接口变更前，至少保留一个次要版本的过渡期
- 过渡期内标记为 `@deprecated`，但仍可正常使用
- 主版本号升级时，移除所有废弃接口

---

## 10. 最佳实践

### 10.1 性能优化

| 优化点 | 建议 |
|--------|------|
| 启动速度 | `on_load` 中只做必要初始化，延迟加载重资源到 `on_activate` |
| 内存占用 | 及时清理不再使用的数据，使用 deque 限制历史长度 |
| CPU 占用 | 后台线程使用合理的 sleep 间隔，避免忙等待 |
| 锁粒度 | 尽量减小锁的范围，避免长时间持有全局锁 |

### 10.2 线程安全

- 所有被外部调用的方法都必须考虑线程安全
- 使用 `threading.Lock` 保护共享状态
- 避免死锁：按固定顺序获取多个锁

### 10.3 配置管理

- 所有可调参数放入 `self.config`
- 提供合理的默认值
- 配置变更后应立即生效（热更新）

### 10.4 日志与调试

- 不要使用 `print` 输出日志（会污染控制台）
- 建议使用 Python 标准 `logging` 模块，logger 名称使用插件名
- 调试信息使用 DEBUG 级别，正常运行时不输出

---

## 11. 示例插件

### 11.1 最小模板

```python
from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint
from typing import Dict, Any
import numpy as np


class MyPlugin(PeripheralPlugin):
    """我的插件。"""

    def __init__(self):
        super().__init__(
            name="my_plugin",
            version="1.0.0",
            description="插件描述",
            plugin_type="sensor",  # sensor / processor / analyzer / actuator
            priority=PluginPriority.NORMAL,
            config={"param1": "value1"},
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[PluginHookPoint.PRE_PERCEPTION]
        )

    def on_load(self) -> bool:
        return True

    def on_unload(self) -> bool:
        return True

    def hook_pre_perception(self, input_data):
        return self.process(input_data)

    def process(self, input_data: Any) -> Any:
        return input_data

    def get_data(self) -> Dict[str, Any]:
        return {"status": "ok"}


def create_plugin():
    return MyPlugin()
```

### 11.2 完整示例

参考 `plugins/mods/` 目录下的示例插件：

- `noise_sensor.py` - 环境噪声传感器
- `temperature_sensor.py` - 温度传感器
- `data_processor.py` - 数据预处理

---

## 附录 A: API 速查表

### PluginManager API

| 方法 | 说明 |
|------|------|
| `scan_available_plugins()` | 扫描所有可用插件 |
| `load_plugin(plugin_name=...)` | 加载指定插件 |
| `unload_plugin(plugin_name)` | 卸载插件 |
| `activate_plugin(plugin_name)` | 激活插件 |
| `deactivate_plugin(plugin_name)` | 停用插件 |
| `reload_plugin(plugin_name)` | 热重载插件 |
| `load_all_from_dir()` | 加载所有插件 |
| `activate_all()` | 激活所有 |
| `deactivate_all()` | 停用所有 |
| `get_active_plugins()` | 获取活跃插件列表 |
| `process_with_plugins(data, hook_point)` | 执行插件链 |
| `invoke_hook(hook_point, data)` | 调用指定钩子 |
| `notify_structure_change(new_dim)` | 通知结构变更 |
| `shutdown()` | 关闭管理器，卸载所有插件 |

### Web API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/plugins/available` | GET | 列出可用插件 |
| `/api/plugins/loaded` | GET | 列出已加载插件 |
| `/api/plugins/load?plugin_name=xxx` | POST | 加载插件 |
| `/api/plugins/{name}/unload` | POST | 卸载插件 |
| `/api/plugins/{name}/activate` | POST | 激活插件 |
| `/api/plugins/{name}/deactivate` | POST | 停用插件 |
| `/api/plugins/{name}/reload` | POST | 热重载 |
| `/api/plugins/load_all` | POST | 全部加载 |
| `/api/plugins/activate_all` | POST | 全部激活 |
| `/api/plugins/deactivate_all` | POST | 全部停用 |
| `/api/plugins/data` | GET | 获取插件数据 |
| `/api/plugins/status` | GET | 管理器状态 |

---

## 附录 B: 常见问题

**Q: 插件文件放入目录后多久会被发现？**  
A: 文件监视线程每 2 秒扫描一次，最多 2 秒内自动发现。

**Q: 修改插件代码后会自动重载吗？**  
A: 会的。系统检测到文件哈希变化后自动热重载，保留原配置。

**Q: 删除插件文件会自动卸载吗？**  
A: 会的。文件被删除后自动卸载并释放资源。

**Q: 插件抛出异常会导致系统崩溃吗？**  
A: 不会。系统在调用插件的所有地方都有 try/except 保护，仅记录错误计数。

**Q: 如何让插件在系统启动时自动加载？**  
A: 放入 mods 目录即可被发现，但不会自动激活。如需自动激活，可在 WebUI 中点击"全部加载"+"全部激活"，或调用对应 API。

---

*本文档为 AGI Agent 插件系统的权威开发规范，所有插件开发必须严格遵循。*
