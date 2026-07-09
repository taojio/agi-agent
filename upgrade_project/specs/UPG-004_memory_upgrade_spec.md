# UPG-004 记忆系统升级规格文档

## 1. 升级目标

### 1.1 现状
- 已有五级记忆体系（L1-L5）
- 基础 CRUD + 简单搜索
- 缺乏自适应遗忘与巩固机制

### 1.2 升级目标
- 语义检索增强（向量相似度 + 关联记忆推荐）
- 自适应遗忘机制（艾宾浩斯遗忘曲线）
- 记忆自动巩固与层级迁移
- 记忆关联网络
- 记忆统计与可视化增强

## 2. 架构设计

### 2.1 新增组件
- MemoryRetriever: 语义检索引擎
- MemoryForgetting: 自适应遗忘机制
- MemoryConsolidation: 记忆巩固与迁移
- MemoryGraph: 记忆关联网络

### 2.2 API 端点
- POST /api/memory/search - 语义搜索
- GET /api/memory/{id} - 获取记忆详情
- PUT /api/memory/{id} - 更新记忆
- DELETE /api/memory/{id} - 删除记忆
- GET /api/memory/related/{id} - 关联记忆推荐
- POST /api/memory/consolidate - 触发记忆巩固
- GET /api/memory/graph - 记忆网络图数据

## 3. 实施步骤
- [x] 规格设计
- [ ] 记忆检索增强模块
- [ ] 自适应遗忘机制
- [ ] 记忆巩固与迁移
- [ ] 记忆 API 端点增强
- [ ] 单元测试
