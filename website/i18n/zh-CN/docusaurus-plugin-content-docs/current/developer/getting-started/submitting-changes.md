# 提交更改

本指南介绍向 Rayforge 贡献代码改进的流程。

## 创建功能分支

为您的更改创建一个描述性分支：

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/issue-number-description
```

## 进行更改

- 遵循现有代码风格和约定
- 编写清晰、专注的提交和清晰的消息
- 为新功能添加测试
- 根据需要更新文档

## 测试您的更改

运行完整测试套件以确保没有问题：

```bash
# 运行所有测试和检查
pixi run test
pixi run lint
```

## 与上游同步

在创建 pull request 之前，与上游仓库同步：

```bash
# 获取最新更改
git fetch upstream

# 将您的分支变基到最新的 main
git rebase upstream/main
```

## 提交 Pull Request

1. 将您的分支推送到您的 fork：
   ```bash
   git push origin feature/your-feature-name
   ```

2. 在 GitHub 上创建 pull request，包含：
   - 描述更改的清晰标题
   - 详细说明您更改了什么以及为什么
   - 引用任何相关问题
   - 如果更改影响 UI，附上截图

## 代码审查流程

- 所有 pull request 在合并前需要审查
- 及时处理反馈并进行请求的更改
- 保持讨论专注和建设性

## 合并要求

Pull request 在以下情况下合并：

- [ ] 通过所有自动化测试
- [ ] 遵循项目的编码风格
- [ ] 为新功能包含适当的测试
- [ ] 如需要更新文档
- [ ] 至少获得一位维护者的批准

## 其他指南

### 提交消息

使用清晰、描述性的提交消息：

- 以大写字母开头
- 第一行保持在 50 个字符以内
- 使用祈使语气（"Add feature" 而不是 "Added feature"）
- 如需要，在正文中包含更多细节

### 小型、专注的更改

保持 pull request 专注于单个功能或修复。大型更改应分解为更小的逻辑部分。

:::tip 先讨论
对于重大更改，请先开 [issue](https://github.com/barebaric/rayforge/issues) 讨论您的方法，然后再投入大量时间。
:::


:::note 需要帮助？
如果您对贡献流程的任何部分不确定，不要犹豫在 issue 或讨论中寻求帮助。
:::
