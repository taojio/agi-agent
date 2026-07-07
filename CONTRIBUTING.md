# 贡献指南 / Contributing Guide

## 欢迎贡献 / Welcome

感谢你对 AGI Agent 项目的兴趣！我们欢迎各种形式的贡献，包括代码、文档、测试、问题报告等。

Thank you for your interest in the AGI Agent project! We welcome contributions in various forms, including code, documentation, tests, and issue reports.

## 贡献方式 / Ways to Contribute

### 1. 报告问题 / Reporting Issues

如果你发现了 bug 或有功能建议，请在 GitHub Issues 中提交。请提供以下信息：

If you find a bug or have a feature request, please submit it in GitHub Issues. Please provide the following information:

- 问题描述 (Issue description)
- 复现步骤 (Steps to reproduce)
- 预期行为 (Expected behavior)
- 实际行为 (Actual behavior)
- 环境信息 (Environment information)
- 截图或日志 (Screenshots or logs)

### 2. 提交代码 / Submitting Code

#### 分支策略 / Branch Strategy

- `main` - 主分支，稳定版本
- `develop` - 开发分支，最新功能
- `feature/xxx` - 功能分支，新功能开发
- `bugfix/xxx` - Bug修复分支

#### 提交流程 / Submission Process

1. Fork 项目仓库
2. 创建新分支: `git checkout -b feature/your-feature-name`
3. 提交更改: `git commit -m "feat: add new feature"`
4. 推送到你的 fork: `git push origin feature/your-feature-name`
5. 创建 Pull Request

#### 提交规范 / Commit Guidelines

请遵循 Conventional Commits 规范：

Please follow the Conventional Commits specification:

- `feat:` - 新功能 (New feature)
- `fix:` - Bug修复 (Bug fix)
- `docs:` - 文档更新 (Documentation update)
- `refactor:` - 代码重构 (Code refactoring)
- `test:` - 测试代码 (Test code)
- `chore:` - 杂项维护 (Miscellaneous maintenance)

示例:
```
feat: add file ingestion system
fix: resolve memory leak in agent step
docs: update API documentation
```

### 3. 编写文档 / Writing Documentation

良好的文档对项目至关重要。你可以：

Good documentation is crucial for the project. You can:

- 完善 README.md
- 编写 API 文档
- 添加代码注释
- 编写教程和示例

## 开发规范 / Development Guidelines

### 代码风格 / Code Style

- Python 代码请遵循 PEP 8 规范
- 使用类型注解 (Type hints)
- 代码注释清晰
- 函数和变量命名有意义

### 测试要求 / Testing Requirements

- 所有新功能必须添加单元测试
- 所有测试必须通过
- 测试覆盖率应尽可能高

### 安全要求 / Security Requirements

- 不要提交敏感信息
- 使用安全的依赖库
- 遵循安全最佳实践

## 代码审查 / Code Review

所有 Pull Request 都需要经过代码审查才能合并。审查内容包括：

All Pull Requests require code review before merging. Review items include:

- 代码质量和风格
- 功能正确性
- 安全性
- 性能
- 测试覆盖

## 行为准则 / Code of Conduct

请遵守项目的 [行为准则](CODE_OF_CONDUCT.md)。

Please adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md).

## 联系我们 / Contact

如果你有任何问题，请通过以下方式联系我们：

If you have any questions, please contact us through:

- GitHub Issues: https://github.com/taojio/agi-agent/issues
- 讨论区: https://github.com/taojio/agi-agent/discussions

---

## 贡献流程总结 / Contribution Process Summary

```
1. 选择一个 issue 或提出新的功能想法
2. Fork 仓库并创建分支
3. 实现功能或修复 bug
4. 编写测试用例
5. 提交代码并创建 Pull Request
6. 参与代码审查
7. 代码合并到主分支
```

```
1. Select an issue or propose a new feature idea
2. Fork the repository and create a branch
3. Implement the feature or fix the bug
4. Write test cases
5. Commit code and create a Pull Request
6. Participate in code review
7. Code is merged into the main branch
```
