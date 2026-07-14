# 智能体记忆底座 v2.0 架构设计文档

## 1. 系统目标

构建分层、持久化、可检索、可遗忘、可溯源、支持多模态、兼容长短期分离的工业级智能体记忆底座，作为智能体感知、思考、决策、自我迭代的核心数据层，支撑单智能体/多智能体集群、对话、工具调用、任务复盘、人格维持、长期生命周期记忆。

## 2. 核心设计原则与技术映射

### 2.1 分层隔离

| 层级 | 名称 | 生命周期 | 存储方式 | 调度策略 |
|------|------|----------|----------|----------|
| L0 | 瞬时记忆 | 毫秒级 | 内存队列 | 实时写入/自动丢弃 |
| L1 | 上下文记忆 | 分钟级 | 内存字典 | 滑动窗口淘汰 |
| L2 | 短期工作记忆 | 小时级 | SQLite + 缓存 | TTL过期清理 |
| L3 | 中期记忆 | 天/月级 | SQLite + 快照 | 定时归档 |
| L4 | 学习记忆 | 周/季度级 | SQLite + 文件 | 分类存储 |
| L5 | 永久记忆 | 永久 | 向量数据库 + 知识图谱 | 语义索引 |

**技术方案**：各层级独立存储引擎，通过 `StorageBackend` 抽象层隔离，生命周期调度由 `MemoryLifecycleManager` 统一管理，互不阻塞。

### 2.2 语义驱动

**技术方案**：
- 所有记忆条目必须携带 embedding 向量（可选但推荐）
- 检索优先使用向量相似度匹配（余弦相似度）
- 支持模糊关联、跨上下文召回
- 向量索引采用分层策略：L2-L4 使用局部索引，L5 使用专业向量数据库

### 2.3 可控遗忘

**技术方案**：
- 内置记忆衰减机制（基于艾宾浩斯遗忘曲线）
- 重要性权重计算：访问频率 × 时间衰减 × 场景匹配度
- 遗忘插件可替换，支持自定义遗忘策略
- 避免无限膨胀：各层级有最大条目限制

### 2.4 完整溯源

**技术方案**：
- 每条记忆绑定完整元数据：来源、时间、置信度、版本
- 修改日志记录：每次更新记录操作类型、操作时间、操作人
- 支持审计查询：按时间、来源、类别检索修改历史
- 记忆快照备份：定时创建状态快照

### 2.5 多模态统一

**技术方案**：
- 统一 `MemoryPayload` 模型支持：文本、图像特征、音频向量、工具结构化数据、数值指标
- 每种数据类型有对应的编码/解码处理器（插件化）
- 统一存储范式：结构化元数据 + 二进制/序列化内容 + 向量嵌入

### 2.6 分布式扩展

**技术方案**：
- `StorageBackend` 抽象层支持单机轻量化部署 + 分布式集群双模式
- 兼容向量库（Qdrant/Pinecone/FAISS）、关系库（PostgreSQL/SQLite）、时序库
- 分布式模式下支持数据分片和副本同步

### 2.7 安全隔离

**技术方案**：
- 记忆权限分级：PUBLIC、INTERNAL、CONFIDENTIAL、RESTRICTED
- 敏感数据脱敏：内置脱敏管道，支持自定义脱敏规则
- 记忆快照备份与恢复机制
- 访问控制列表（ACL）管理

### 2.8 低耦合插件化

**技术方案**：
- 定义 `MemoryPlugin` 抽象基类
- 插件类型：压缩、摘要、聚类、遗忘、检索、自省
- `PluginRegistry` 统一管理插件注册与发现
- 插件热插拔：运行时动态加载/卸载

## 3. 核心架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      MemoryBase v2.0                          │
├─────────────────────────────────────────────────────────────────┤
│                     插件层 (Plugins)                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Retrieval│ │Forgetting│ │Consolidation│ │Compression│ │Summary│ │
│  │ Plugin  │ │ Plugin  │ │ Plugin    │ │ Plugin   │ │Plugin  │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │
│       │           │           │           │           │       │
├───────┼───────────┼───────────┼───────────┼───────────┼───────┤
│                     核心管理层 (Core)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         MemoryHarness (记忆检索与调度管理器)             │   │
│  │  - 跨层级检索    - 综合评分    - 插件调度    - 生命周期  │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
├─────────────────────────┼───────────────────────────────────────┤
│                     数据层 (Data)                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MemoryPayload (多模态统一数据模型)                      │   │
│  │  MemoryEntry (记忆条目)                                  │   │
│  │  MemoryMetadata (元数据)                                 │   │
│  │  MemoryAccessControl (安全层)                            │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
├─────────────────────────┼───────────────────────────────────────┤
│                     存储层 (Storage)                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │  L0-L1       │  L2-L3       │  L4          │  L5          │  │
│  │  内存存储    │  SQLite      │  文件+SQLite │  向量数据库  │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│                         │                                       │
│              ┌───────────┴───────────┐                          │
│              │   StorageBackend      │                          │
│              │   (抽象存储接口)       │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

| 组件 | 职责 | 关键接口 |
|------|------|----------|
| MemoryPlugin | 插件抽象基类 | `process()`, `configure()` |
| PluginRegistry | 插件注册与管理 | `register()`, `get()`, `list()` |
| MemoryPayload | 多模态数据模型 | `encode()`, `decode()`, `to_dict()` |
| MemoryEntry | 记忆条目 | `update_access()`, `generate_hash()` |
| MemoryMetadata | 元数据 | `to_dict()`, `from_dict()` |
| StorageBackend | 存储后端抽象 | `add()`, `get()`, `search()`, `delete()` |
| MemoryHarness | 核心调度器 | `retrieve()`, `add_memory()`, `cleanup()` |
| MemoryAccessControl | 安全控制 | `check_permission()`, `sanitize()` |

## 4. 数据模型设计

### 4.1 MemoryPayload（多模态数据载荷）

```python
@dataclass
class MemoryPayload:
    data_type: DataType  # TEXT, IMAGE, AUDIO, STRUCTURED, NUMERIC
    content: Any          # 原始内容
    encoding: str        # 编码格式
    embedding: Optional[List[float]]  # 向量表示
    metadata: Dict[str, Any]  # 附加元数据
```

### 4.2 MemoryEntry（记忆条目）

```python
@dataclass
class MemoryEntry:
    memory_id: str
    payload: MemoryPayload
    metadata: MemoryMetadata
    version: int = 1
    related_ids: List[str] = field(default_factory=list)
    audit_log: List[AuditRecord] = field(default_factory=list)
```

### 4.3 MemoryMetadata（元数据）

```python
@dataclass
class MemoryMetadata:
    tier: MemoryTier
    category: MemoryCategory
    permission_level: PermissionLevel  # 新增：权限级别
    importance: float = 0.5           # 新增：重要性权重
    decay_rate: float = 0.95          # 新增：衰减率
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    accessed_count: int = 0
    scene_tags: List[str] = field(default_factory=list)
    source_agent: str = ""
    task_id: str = ""
```

### 4.4 AuditRecord（审计记录）

```python
@dataclass
class AuditRecord:
    operation_type: str  # CREATE, UPDATE, DELETE, ACCESS
    timestamp: float = field(default_factory=time.time)
    operator: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
```

## 5. 插件系统设计

### 5.1 MemoryPlugin 基类

```python
class MemoryPlugin(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_type(self) -> PluginType:
        pass
    
    @abstractmethod
    def process(self, entry: MemoryEntry) -> MemoryEntry:
        pass
    
    def configure(self, config: Dict[str, Any]):
        pass
```

### 5.2 插件类型

| 类型 | 用途 | 示例 |
|------|------|------|
| RETRIEVAL | 检索策略 | 语义检索、关键词检索、混合检索 |
| FORGETTING | 遗忘策略 | 艾宾浩斯遗忘、LRU淘汰、重要性过滤 |
| CONSOLIDATION | 巩固策略 | 记忆升级、摘要生成、知识融合 |
| COMPRESSION | 压缩策略 | 无损压缩、有损压缩、量化 |
| SUMMARY | 摘要策略 | 文本摘要、关键信息提取 |
| CLUSTERING | 聚类策略 | 相似记忆分组、主题聚类 |
| INTROSPECTION | 自省策略 | 记忆质量评估、矛盾检测 |

## 6. 安全层设计

### 6.1 权限级别

| 级别 | 说明 | 访问范围 |
|------|------|----------|
| PUBLIC | 公开记忆 | 所有智能体可访问 |
| INTERNAL | 内部记忆 | 同集群智能体可访问 |
| CONFIDENTIAL | 机密记忆 | 授权智能体可访问 |
| RESTRICTED | 受限记忆 | 仅创建者可访问 |

### 6.2 脱敏规则

```python
class SanitizationRule:
    - 规则ID
    - 匹配模式（正则表达式）
    - 替换策略（掩码/哈希/删除）
    - 适用数据类型
```

## 7. 分布式扩展设计

### 7.1 StorageBackend 抽象

```python
class StorageBackend(ABC):
    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 20) -> List[MemoryEntry]:
        pass
    
    @abstractmethod
    def semantic_search(self, embedding: List[float], limit: int = 20) -> List[MemoryEntry]:
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
```

### 7.2 支持的后端类型

| 后端 | 适用场景 | 依赖 |
|------|----------|------|
| LocalMemoryBackend | L0-L1 内存存储 | 无 |
| SQLiteBackend | L2-L4 持久化存储 | sqlite3 |
| QdrantBackend | L5 向量存储 | qdrant-client |
| PostgreSQLBackend | 分布式持久化 | psycopg2 |
| RedisBackend | 缓存层 | redis |

## 8. 实施计划

### 阶段一：核心抽象层（第1-2周）

- 定义 `MemoryPlugin` 基类和 `PluginRegistry`
- 定义 `MemoryPayload` 多模态数据模型
- 定义 `StorageBackend` 抽象接口
- 定义 `MemoryAccessControl` 安全层

### 阶段二：数据模型升级（第3-4周）

- 扩展 `MemoryEntry` 和 `MemoryMetadata`
- 实现 `AuditRecord` 审计记录
- 实现多模态编码/解码处理器

### 阶段三：存储层重构（第5-6周）

- 实现 `LocalMemoryBackend`
- 实现 `SQLiteBackend`
- 实现 `QdrantBackend`（可选）
- 各层级存储引擎适配

### 阶段四：插件化改造（第7-8周）

- 将现有 `retrieval.py` 重构为插件
- 将现有 `forgetting.py` 重构为插件
- 将现有 `consolidation.py` 重构为插件
- 实现新插件类型（压缩、摘要、聚类、自省）

### 阶段五：安全与分布式（第9-10周）

- 实现 `MemoryAccessControl`
- 实现脱敏管道
- 实现分布式模式支持
- 实现备份与恢复机制

### 阶段六：集成测试（第11-12周）

- 全链路集成测试
- 性能基准测试
- 向后兼容性测试
- 文档完善

## 9. 向后兼容性

- 现有 `MemoryEntry` 和 `MemoryMetadata` 保持兼容
- 新增字段默认值保证旧代码不报错
- 存储格式自动迁移
- 提供数据迁移工具

## 10. 性能指标

| 指标 | 目标值 |
|------|--------|
| 单条记忆写入延迟 | < 10ms |
| 语义检索延迟（10000条） | < 50ms |
| 跨层级检索延迟 | < 200ms |
| 并发写入吞吐量 | > 1000条/秒 |
| 内存占用（L1） | < 100MB |

---

**文档版本**: v1.0  
**创建日期**: 2026-07-10  
**适用项目**: AGI Agent Project Phoenix