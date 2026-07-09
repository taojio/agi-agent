# UPG-001 安全系统升级规格文档

> 模块ID：UPG-001  
> 优先级：P0（最高）  
> 预计周期：6周  
> 版本：v1.0  

---

## 1. 升级目标

将系统安全评分从 5/10 提升至 9/10，建立多层纵深防御体系，实现可证明的安全约束。

### 1.1 核心目标

| 目标 | 当前状态 | 目标状态 |
|------|----------|----------|
| 认证机制 | 无 | JWT + 刷新令牌 |
| 授权机制 | 无 | RBAC 5级角色 |
| 输入验证 | 几乎无 | 全局校验中间件 |
| SQL注入防护 | 部分拼接 | ORM + 参数化查询 |
| XSS防护 | 无 | 输出编码 + CSP |
| 速率限制 | 无 | 分级限流 |
| CORS配置 | 宽松 | 严格白名单 |
| 审计日志 | 基础 | 全面审计追踪 |
| 敏感数据保护 | 明文存储 | 加密存储 + 脱敏 |

### 1.2 OWASP Top 10 覆盖

| 风险项 | 当前状态 | 升级后 |
|--------|----------|--------|
| A01 失效的访问控制 | 🔴 高危 | ✅ 修复 |
| A02 加密失效 | 🟠 高 | ✅ 修复 |
| A03 注入 | 🟠 高 | ✅ 修复 |
| A04 不安全设计 | 🟠 高 | 🟡 缓解 |
| A05 安全配置错误 | 🟡 中 | ✅ 修复 |
| A06 脆弱且过时的组件 | 🟡 中 | 🟡 缓解 |
| A07 识别和认证失败 | 🔴 高危 | ✅ 修复 |
| A08 软件和数据完整性失败 | 🟡 中 | ✅ 修复 |
| A09 安全日志和监控失败 | 🟡 中 | ✅ 修复 |
| A10 服务端请求伪造 | ⚪ 低 | ✅ 修复 |

---

## 2. 架构设计

### 2.1 安全架构分层

```
┌─────────────────────────────────────────────────────────┐
│                    API 网关层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ CORS 加固 │ │ 速率限制  │ │ 请求日志  │ │ WAF 规则  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                    认证授权层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ JWT 认证  │ │ RBAC授权 │ │ 会话管理 │ │ API Key  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                    输入验证层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 类型校验  │ │ 长度限制  │ │ 格式校验  │ │ 内容过滤  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                    数据安全层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 加密存储  │ │ 数据脱敏  │ │ SQL注入防 │ │ 路径遍历防 │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                    安全监控层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 审计日志  │ │ 异常检测  │ │ 告警通知  │ │ 安全报表  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

| 子模块 | 文件路径 | 职责 |
|--------|----------|------|
| 用户模型 | `security/models.py` | 用户、角色、权限数据模型 |
| JWT认证 | `security/jwt_auth.py` | Token生成、验证、刷新 |
| RBAC授权 | `security/rbac.py` | 角色权限管理、权限检查 |
| 输入验证 | `security/validation.py` | 全局输入校验中间件 |
| 速率限制 | `security/rate_limiter.py` | API限流、防暴力破解 |
| 安全头部 | `security/headers.py` | CSP、HSTS、CORS等安全头 |
| 数据保护 | `security/data_protection.py` | 加密、脱敏、哈希 |
| 审计日志 | `security/audit_logger.py` | 安全事件审计追踪 |
| 异常处理 | `security/exceptions.py` | 安全异常统一处理 |

---

## 3. 详细设计

### 3.1 用户认证系统 (JWT)

#### 3.1.1 用户模型

```python
class User:
    id: UUID
    username: str           # 唯一用户名
    email: str              # 唯一邮箱
    password_hash: str      # bcrypt哈希
    salt: str               # 密码盐值
    role: str               # 角色名
    is_active: bool         # 是否激活
    is_verified: bool       # 是否已验证
    last_login_at: datetime
    created_at: datetime
    updated_at: datetime
    metadata: dict          # 扩展字段
```

#### 3.1.2 角色定义（5级）

| 角色 | 权限等级 | 说明 |
|------|----------|------|
| `super_admin` | 🔴 L5 | 超级管理员，系统所有权限 |
| `admin` | 🟠 L4 | 管理员，用户管理、系统配置 |
| `operator` | 🟡 L3 | 操作员，日常运维操作 |
| `viewer` | 🔵 L2 | 只读用户，查看数据和报表 |
| `guest` | ⚪ L1 | 访客，仅基本访问权限 |

#### 3.1.3 JWT Token 结构

```python
# Access Token (15分钟过期)
{
    "sub": "user_id",
    "username": "xxx",
    "role": "admin",
    "type": "access",
    "iat": 1234567890,
    "exp": 1234568790,
    "jti": "unique_token_id"
}

# Refresh Token (7天过期)
{
    "sub": "user_id",
    "type": "refresh",
    "iat": 1234567890,
    "exp": 1235172690,
    "jti": "unique_refresh_id"
}
```

#### 3.1.4 API 端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/api/auth/register` | POST | ❌ | 用户注册（需管理员审批） |
| `/api/auth/login` | POST | ❌ | 用户登录，返回Token |
| `/api/auth/refresh` | POST | Refresh Token | 刷新Access Token |
| `/api/auth/logout` | POST | ✅ | 登出，注销Token |
| `/api/auth/me` | GET | ✅ | 获取当前用户信息 |
| `/api/auth/password/change` | POST | ✅ | 修改密码 |
| `/api/auth/password/reset` | POST | ❌ | 请求密码重置 |
| `/api/admin/users` | GET | L4 | 用户列表 |
| `/api/admin/users/{id}` | PUT | L4 | 修改用户信息 |
| `/api/admin/users/{id}/role` | PATCH | L5 | 修改用户角色 |

### 3.2 RBAC 权限系统

#### 3.2.1 权限模型

采用 `角色 → 权限 → 资源` 三级模型：

```
角色(Role) ──n:n── 权限(Permission) ──n:n── 资源(Resource)
```

#### 3.2.2 权限定义示例

| 权限代码 | 说明 | 资源类型 |
|----------|------|----------|
| `memory:read` | 读取记忆 | memory |
| `memory:write` | 写入记忆 | memory |
| `memory:delete` | 删除记忆 | memory |
| `soul:read` | 读取SOUL | soul |
| `soul:write` | 修改SOUL | soul |
| `evolution:run` | 运行进化 | evolution |
| `agent:control` | 控制Agent | agent |
| `admin:user:manage` | 用户管理 | admin |
| `admin:system:config` | 系统配置 | admin |
| `security:audit:view` | 查看审计 | security |

#### 3.2.3 权限检查中间件

```python
# 使用方式
@app.get("/api/memory/list")
@require_permission("memory:read")
async def list_memories(...):
    ...
```

### 3.3 输入验证系统

#### 3.3.1 验证规则

- **类型验证**：字符串、数字、布尔值、列表、字典
- **长度验证**：最小长度、最大长度
- **格式验证**：邮箱、URL、IP、手机号、UUID、JSON
- **范围验证**：数值范围、枚举值
- **内容验证**：敏感词过滤、XSS检测、SQL注入检测
- **文件验证**：类型、大小、MIME校验

#### 3.3.2 Pydantic 模型验证

所有 API 输入/输出使用 Pydantic 模型定义，自动进行类型和格式验证。

### 3.4 速率限制系统

#### 3.4.1 限流策略

| 端点类型 | 限制 | 窗口 | 说明 |
|----------|------|------|------|
| 登录接口 | 5次/IP | 1分钟 | 防暴力破解 |
| 注册接口 | 3次/IP | 1小时 | 防恶意注册 |
| 普通API | 100次/用户 | 1分钟 | 正常使用限制 |
| 管理API | 30次/用户 | 1分钟 | 敏感操作限流 |
| 文件上传 | 10次/用户 | 1小时 | 防存储耗尽 |

#### 3.4.2 实现方案

- 基于 Redis 的滑动窗口算法
- 支持 IP 级和用户级双重限流
- 超限返回 429 Too Many Requests
- 响应头包含限流状态信息

### 3.5 安全头部

| 头部 | 值 | 说明 |
|------|-----|------|
| `X-Content-Type-Options` | `nosniff` | 防止MIME类型嗅探 |
| `X-Frame-Options` | `DENY` | 防止点击劫持 |
| `X-XSS-Protection` | `1; mode=block` | XSS防护 |
| `Content-Security-Policy` | 严格策略 | 内容安全策略 |
| `Strict-Transport-Security` | `max-age=31536000` | HSTS强制HTTPS |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | 引用策略 |
| `Permissions-Policy` | 受限 | 浏览器权限限制 |

### 3.6 审计日志系统

#### 3.6.1 日志事件类型

| 类型 | 级别 | 说明 |
|------|------|------|
| `auth.login` | INFO | 用户登录 |
| `auth.login.failed` | WARNING | 登录失败 |
| `auth.logout` | INFO | 用户登出 |
| `auth.password.change` | INFO | 密码修改 |
| `auth.token.refresh` | DEBUG | Token刷新 |
| `user.created` | INFO | 用户创建 |
| `user.role.changed` | WARNING | 角色变更 |
| `user.deactivated` | WARNING | 用户停用 |
| `security.boundary.violation` | CRITICAL | 安全边界违规 |
| `security.rate_limit.exceeded` | WARNING | 超限流 |
| `system.config.changed` | WARNING | 系统配置变更 |
| `data.export` | INFO | 数据导出 |
| `data.delete` | WARNING | 数据删除 |

#### 3.6.2 审计日志字段

```python
{
    "event_id": "uuid",
    "event_type": "auth.login",
    "severity": "INFO",
    "timestamp": "2026-07-10T...",
    "user_id": "xxx",
    "username": "xxx",
    "role": "admin",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "resource": "/api/auth/login",
    "method": "POST",
    "status_code": 200,
    "duration_ms": 123,
    "details": { ... },
    "request_id": "req_xxx"
}
```

---

## 4. 实施步骤

### 阶段一：基础框架（第1周）
- [ ] 创建安全模块目录结构
- [ ] 用户模型与数据库表设计
- [ ] 密码哈希与验证工具
- [ ] 基础异常类定义

### 阶段二：JWT认证（第2周）
- [ ] JWT Token 生成与验证
- [ ] 登录/注册/登出 API
- [ ] Token 刷新机制
- [ ] Token 黑名单（登出失效）
- [ ] 认证依赖注入

### 阶段三：RBAC授权（第3周）
- [ ] 角色与权限数据模型
- [ ] 权限检查中间件
- [ ] 角色权限管理 API
- [ ] 默认角色与权限初始化

### 阶段四：输入验证（第4周）
- [ ] 全局请求验证中间件
- [ ] Pydantic 模型全面覆盖
- [ ] XSS 过滤与输出编码
- [ ] SQL 注入防护检查
- [ ] 路径遍历防护

### 阶段五：限流与安全头（第5周）
- [ ] 速率限制中间件
- [ ] CORS 严格白名单
- [ ] 安全响应头中间件
- [ ] CSP 策略配置

### 阶段六：审计与加固（第6周）
- [ ] 审计日志系统
- [ ] 敏感数据加密存储
- [ ] 安全扫描与漏洞修复
- [ ] 文档与测试
- [ ] 上线验证

---

## 5. 依赖与风险

### 5.1 新增依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `pyjwt` | >=2.8 | JWT Token 处理 |
| `passlib[bcrypt]` | >=1.7 | 密码哈希 |
| `python-multipart` | >=0.0.6 | 文件上传处理 |
| `slowapi` | >=0.1.9 | 速率限制 |
| `pydantic` | >=2.0 | 数据验证 |
| `email-validator` | >=2.0 | 邮箱验证 |
| `cryptography` | >=41.0 | 加密算法 |

### 5.2 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 改造期间API不可用 | 中 | 高 | 渐进式迁移，新旧API并行 |
| 密码迁移困难 | 低 | 中 | 支持新旧哈希共存，登录时自动升级 |
| 性能影响 | 低 | 中 | 缓存验证结果，异步写审计日志 |
| 用户体验下降 | 中 | 中 | 合理的Token过期时间，无感刷新 |

---

## 6. 验证标准

### 6.1 功能验证
- [ ] 用户注册/登录/登出流程完整
- [ ] JWT Token 生成、验证、刷新正常
- [ ] RBAC 权限控制正确
- [ ] 输入验证拦截恶意输入
- [ ] 速率限制生效
- [ ] 安全头部正确设置
- [ ] 审计日志完整记录

### 6.2 安全验证
- [ ] OWASP ZAP 扫描无高危漏洞
- [ ] SQL注入测试全部拦截
- [ ] XSS测试全部过滤
- [ ] 暴力破解测试成功限流
- [ ] 越权访问测试全部拒绝

### 6.3 性能验证
- [ ] 认证中间件延迟 < 5ms
- [ ] 权限检查延迟 < 2ms
- [ ] 整体 API 延迟增加 < 10%
