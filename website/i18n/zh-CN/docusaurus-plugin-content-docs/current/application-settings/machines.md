---
description:
  "在 Rayforge 中管理机器——为您的项目添加、配置、导出、导入以及在不同的激光切割机和雕刻机之间切换。"
---

# 机器

![机器设置](/screenshots/app-settings-machines.webp)

**机器**
页面显示所有已配置机器的列表。每个条目显示机器名称，并带有编辑或删除按钮。当前活动的机器用勾选图标标记。

从主窗口通过 **机器 → Manage Machines…** 打开该页面，或在应用设置中选择
**机器**（<kbd>ctrl+comma</kbd>）。

## 添加机器

添加机器最快的方式是在主窗口中选择 **机器 → Add
Machine…**，它会直接打开配置向导。在机器页面中，点击列表底部的 **Add Machine** 按钮，然后：

1. 从列表中选择一个设备配置文件作为模板——每个配置文件会预设机器设置和 G-code 方言

![添加机器](/screenshots/app-settings-machines-add.webp)

2. [机器设置对话框](../machine/general.md) 将打开，您可以在其中调整配置

或者：

- 点击 **Device Not Listed** 启动
  [配置向导](../getting-started/first-time-setup.md)，引导您逐步配置机器
- 点击 **Import from File…**
  从之前导出的配置文件或 LightBurn 设备配置文件 (.lbdev) 中添加机器。LightBurn 配置文件包含相机校准和激光设置，这些设置将应用到新机器上。

## 编辑机器

点击机器旁边的编辑图标以打开 [机器设置对话框](../machine/general.md)。

## 机器备注

位于 **General** 之后的 **Notes**
类别汇集了设备配置文件提供的指导以及您为该机器编写的备注。设备备注为只读；**My Notes**
可使用 Markdown 编辑。支持标题、段落、粗体、斜体、列表、链接、行内代码、代码块以及使用 `:::details`
的可展开部分。

如果配置文件包含设置说明，它们也会显示在配置向导最后的 **Review & Name**
页面，并在机器创建后继续保留在 **Machine Settings → Notes**
中。个人备注按机器单独保存，审核配置文件更新时不会被替换。

## 切换活动机器

**机器 → Select Machine** 子菜单列出了所有已配置的机器；活动机器带有勾选标记。选择另一个条目即可切换到该机器。您也可以使用主窗口标题栏中的机器下拉菜单。选择会在会话之间记住。

## 删除机器

1. 点击机器旁边的删除图标
2. 确认删除

<!-- prettier-ignore-start -->
:::warning
删除机器无法撤销。如果您想保留配置，请先导出配置文件。
:::
<!-- prettier-ignore-end -->
