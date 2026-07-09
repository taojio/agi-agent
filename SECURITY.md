# 安全政策 / Security Policy

## 支持的版本 / Supported Versions

以下版本的 AGI Agent 项目目前接受安全更新：

The following versions of the AGI Agent project are currently being supported with security updates:

| 版本 / Version | 支持状态 / Supported |
|---------------|---------------------|
| Latest | ✅ 支持 / Yes |
| Previous minor releases | ⚠️ 有限支持 / Limited |

## 报告漏洞 / Reporting a Vulnerability

我们非常重视安全问题。如果您发现了安全漏洞，请按照以下方式报告：

We take security issues very seriously. If you discover a security vulnerability, please report it as described below.

### 如何报告 / How to Report

1. **通过 GitHub Issues 提交**：在项目仓库中创建一个新的 Issue，标记为 `security` 标签
2. **描述漏洞**：提供漏洞的详细描述，包括：
   - 漏洞类型（例如：SQL注入、XSS、路径遍历等）
   - 复现步骤
   - 影响范围
   - 可能的修复建议

### 通过 GitHub Issues 提交：在项目仓库中创建一个新的 Issue，标记为 `security` 标签

- 漏洞类型（例如：SQL注入、XSS、路径遍历等）
- 复现步骤
- 影响范围
- 可能的修复建议

### 响应时间 / Response Time

我们承诺在收到漏洞报告后的 48 小时内确认收到。我们会尽快调查并提供修复方案。

We commit to acknowledging receipt of the vulnerability report within 48 hours. We will investigate and provide a fix as soon as possible.

## 漏洞处理流程 / Vulnerability Handling Process

1. **确认**：确认漏洞报告并进行初步评估
2. **评估**：评估漏洞的严重程度和影响范围
3. **修复**：开发修复方案并进行测试
4. **验证**：验证修复效果
5. **发布**：发布安全更新
6. **通知**：通知社区关于安全漏洞和修复的信息

## 安全最佳实践 / Security Best Practices

### 代码层面 / Code Level

- 所有用户输入必须进行验证和清理
- 使用参数化查询防止 SQL 注入
- 对敏感数据进行加密存储
- 使用 HTTPS 保护通信
- 实施适当的访问控制

### 部署层面 / Deployment Level

- 保持依赖库和系统更新
- 限制不必要的网络访问
- 配置适当的防火墙规则
- 定期备份数据
- 监控系统日志

### 贡献者指南 / Contributor Guidelines

所有贡献者在提交代码时应遵守以下安全原则：

- 遵循安全编码最佳实践
- 避免硬编码敏感信息
- 对用户输入进行验证
- 考虑潜在的安全漏洞
- 添加适当的错误处理

## 安全联系信息 / Security Contact

如有任何安全相关问题，请通过以下方式联系我们：

- GitHub Issues: https://github.com/taojio/agi-agent/issues
- 项目讨论区: https://github.com/taojio/agi-agent/discussions

## 免责声明 / Disclaimer

本项目按"原样"提供，不提供任何形式的保证。使用本软件的风险由用户自行承担。

This project is provided "as is" without warranty of any kind. Users assume all risks associated with using this software.