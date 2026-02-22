# 获取代码

本指南介绍如何获取 Rayforge 源代码进行开发。

## Fork 仓库

在 GitHub 上 fork [Rayforge 仓库](https://github.com/barebaric/rayforge) 以创建您自己的副本，您可以在其中进行更改。

## 克隆您的 Fork

```bash
git clone https://github.com/YOUR_USERNAME/rayforge.git
cd rayforge
```

## 添加上游仓库

将原始仓库添加为上游远程，以便跟踪更改：

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## 验证仓库

检查远程配置是否正确：

```bash
git remote -v
```

您应该看到您的 fork（origin）和上游仓库。

## 下一步

获取代码后，继续 [环境设置](setup) 配置您的开发环境。
