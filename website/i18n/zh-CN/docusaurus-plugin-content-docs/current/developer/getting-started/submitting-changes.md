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

2. 在 GitHub 上创建 pull request 并填写 pull request 模板：
   - 清晰的标题，以及说明更改内容及其所解决的问题
   - 为什么选择这种特定的架构方案，以及否决了哪些备选方案
   - 本地测试该更改的确切步骤
   - 引用任何相关问题
   - 如果更改影响 UI，附上截图
   - 勾选模板清单中的所有复选框

<!-- prettier-ignore-start -->
:::warning[模板强制执行]
未填写模板的 pull request 会被机器人自动关闭。编辑 pull request 描述，回答每个部分并勾选每个
复选框后，pull request 会自动重新打开。
:::
<!-- prettier-ignore-end -->

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

## AI 辅助代码

我们欢迎使用 AI 工具来辅助编写代码。但是，你提交的代码完全由你负责。你必须完全理解这些代码，并且能够解释其中的架构决策。

AI 生成的代码常常引入隐蔽的 bug、架构不匹配和安全漏洞。看起来是未经审查的原始 AI 输出的 pull
request 将被关闭，以尊重维护者的时间。

## 其他指南

### 提交消息

使用清晰、描述性的提交消息：

- 以大写字母开头
- 第一行保持在 50 个字符以内
- 使用祈使语气（"Add feature" 而不是 "Added feature"）
- 如需要，在正文中包含更多细节

### 小型、专注的更改

保持 pull request 专注于单个功能或修复。大型更改应分解为更小的逻辑部分。

<!-- prettier-ignore-start -->
:::tip[先讨论]
对于重大更改，请先开 [issue](https://github.com/barebaric/rayforge/issues)
讨论您的方法，然后再投入大量时间。
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[需要帮助？]
如果您对贡献流程的任何部分不确定，不要犹豫在 issue 或讨论中寻求帮助。
:::
<!-- prettier-ignore-end -->
