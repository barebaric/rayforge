---
description: "首次配置您的激光切割机或雕刻机。使用配置向导创建您的机器，然后连接并准备好在 Rayforge 中进行切割。"
---

# 首次设置

安装 Rayforge 后，您需要配置激光切割机或雕刻机。本指南将带您通过配置向导创建您的第一台机器并建立连接。

## 第 1 步：启动 Rayforge

从应用程序菜单启动 Rayforge，或在终端中运行 `rayforge`。您应该看到带有空白画布的主界面。

## 第 2 步：使用向导创建机器

导航到 **设置 → 机器** 或按 <kbd>ctrl+comma</kbd> 打开设置对话框，然后选择 **机器** 页面。

![机器设置](/screenshots/app-settings-machines.png)

点击 **Add Machine** 打开机器选择器。

![添加机器](/screenshots/app-settings-machines-add.png)

配置向导将打开，并根据您的选择调整其显示的步骤：

- 选择**内置配置文件**会预填控制器、工作区域和机头——向导会直接跳到旋转模块、相机和审核步骤
- **导入配置文件**会保留硬件和机头步骤，以便您纠正导入过程中的任何错误
- **Device Not Listed** 会引导您完成每一步，包括控制器和 AI 规格查询步骤

### 选择起点

选择一个内置设备配置文件以预填控制器、工作区域和机头设置，或点击 **Device Not Listed** 手动配置所有内容。您也可以 **Import from File…** 导入之前导出的配置文件或带有相机校准和激光设置的 LightBurn 设备配置文件 (.lbdev)。

![向导 — 选择起点](/screenshots/config-wizard-profile.png)

### 选择控制器

选择与您机器控制器板匹配的固件或协议系列（GRBL、Marlin、Smoothie、Ruida、OctoPrint、…）。如果您只想将 G-code 导出到文件而不驱动物理机器，请选择 **None — G-code export only**。当您从内置配置文件或导入开始时，此步骤会被跳过。

![向导 — 选择控制器](/screenshots/config-wizard-controller.png)

### 连接

输入您的机器所需的连接参数。具体字段取决于您选择的控制器：

- **串口驱动程序** — USB 设备路径（例如 Linux 上的 `/dev/ttyUSB0`、Windows 上的 `COM3`）和波特率
- **网络驱动程序** — 主机地址和端口（例如 `192.168.1.100`）
- **OctoPrint** — 服务器 URL 和 API 密钥

![向导 — 连接](/screenshots/config-wizard-connect.png)

### 发现设备

当您的控制器支持时，向导会提供连接设备并自动读取其配置的选项——工作区域、速度、加速度和固件功能。点击 **Probe Now** 自动检测这些值，或使用 **Next** 在后续步骤中手动输入。

![向导 — 发现设备](/screenshots/config-wizard-probe.png)

### AI 提供方

仅当尚未配置 AI 提供方时显示。输入兼容 OpenAI 的端点（基础 URL 和 API 密钥），以便下一步可以查询已知商业机器的规格。跳过此步骤可手动输入这些值。

![向导 — AI 提供方](/screenshots/config-wizard-ai-provider.png)

### AI 规格查询

如果您的机器是已知的商业型号，AI 可以从制造商的文档中预填其规格。输入供应商和型号，然后点击 **Look Up Specs**。建议的值以开关行形式出现并默认接受——关闭您不想应用的任何内容。

![向导 — AI 规格查询](/screenshots/config-wizard-ai-lookup.png)

### 硬件

配置机器的物理设置：

- **轴** — X/Y 工作区域范围和坐标原点 (0,0) 角
- **轴方向** — 如果坐标出现负值，则反转某个轴
- **Z 轴** — 机器是否有 Z 轴（对焦电机、可移动床台）；如果不存在，
  则不生成 Z 移动，3D 画布将内容分层放置在雕刻平面上
- **面板方向** — 旋转平面工作区在屏幕上的呈现方式（原生、向左旋转、
  向右旋转）；旋转层需要原生
- **工作区域** — 工作表面不可使用空间周围的边距
- **软限位** — 点动的可选安全边界
- **速度** — 最大空移速度、最大切割速度和加速度
- **行为** — 启动时归零和单轴归零

![向导 — 硬件](/screenshots/config-wizard-hardware.png)

### 机头

声明安装在龙门架上的部件——Laser Head 或 Spindle Head——并设置其参数。对于激光：最大功率（S 值）、光斑大小、PWM 频率和焦距。对于主轴：最大和最小 RPM。

![向导 — 机头](/screenshots/config-wizard-head.png)

### 旋转模块

可选地设置旋转附件：类型（卡爪或滚轮）、轴（A/B/C）、模式（真正的第 4 轴 vs. 轴替换）、几何形状和反向方向标志。跳过此步骤可稍后从机器设置中添加旋转模块。

![向导 — 旋转模块](/screenshots/config-wizard-rotary.png)

### 相机

可选地启用您想用于预览和对齐的任何相机。当您启用相机并继续时，[相机向导](../machine/camera.md#第-2-步相机向导) 将打开，引导您完成图像设置、镜头校准和图像对齐。您可以跳过此步骤，稍后从机器的相机设置中设置相机。

![向导 — 相机](/screenshots/config-wizard-camera.png)

### 审核与命名

为机器命名并审核您已配置的所有内容的摘要——驱动程序、连接、工作区域、速度、机头、旋转模块和相机。向导还会提示任何警告，例如缺少驱动程序或未设置工作区域。

![向导 — 审核与命名](/screenshots/config-wizard-review.png)

点击 **Create Machine** 完成。机器设置对话框将为您的机器打开，您可以在其中调整向导预填的任何设置。有关详细信息，请参阅[机器设置](../machine/general.md)页面。

## 第 3 步：自动连接

Rayforge 在应用程序启动时自动连接到您的机器（如果机器已开机并连接）。您不需要手动点击连接按钮。

连接状态显示在主窗口左下角，带有状态图标和标签，显示当前状态（已连接、正在连接、已断开、错误等）。

:::success 已连接！
如果您的机器显示"已连接"状态，您就可以开始使用 Rayforge 了！
:::

---

## 连接问题故障排除

### 找不到设备

- **Linux（串口）**：将用户添加到 `dialout` 组。对于**基于 Debian
  的发行版上的 Snap 和非 Snap 安装**，都需要此操作以避免 AppArmor
  DENIED 消息：
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  注销并重新登录以使更改生效。

- **Snap 包**：除了上述 `dialout` 组外，还需确保您已授予串口权限：
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**：检查设备管理器以确认设备被识别并记下 COM 端口号。

### 连接被拒绝

- 验证 IP 地址和端口号是否正确
- 确保机器已开机并连接到网络
- 如果使用网络连接，检查防火墙设置

### 机器无响应

- 尝试不同的波特率（有些设备使用 `9600` 或 `57600`）
- 检查电缆是否松动或连接不良
- 重启激光切割机并重试

有关更多帮助，请参阅[连接问题](../troubleshooting/connection.md)。

---

**下一步：**[快速入门指南 →](quick-start)
