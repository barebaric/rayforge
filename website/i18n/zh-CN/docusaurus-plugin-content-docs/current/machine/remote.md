---
description:
  "通过放在硬件旁边的小型计算机运行机器，而 Rayforge 在其他地方运行。任务在断开连接后继续执行，
  重新连接时可精确同步进度。"
---

# 远程设备

通过放在硬件旁边的小型计算机（例如车间里的 Raspberry Pi）运行机器，而 Rayforge 在您的笔记本或
台式机上运行。任务在设备主机上执行，即使 Rayforge 断开连接也会继续；重新连接时可精确同步进度。

## 工作原理

- **raydriver 守护进程**在硬件旁边托管机器连接。
- Rayforge 连接到*守护进程上的设备*——它无需知道设备使用哪种协议。守护进程会解析协议，并告诉
  Rayforge 该设备能做什么。
- 远程机器使用单一的 **Remote machine (raydriver)** 驱动。特定协议的驱动（GRBL、Smoothie、Marlin）
  用于直接连接到您计算机的硬件。

## 设置设备主机

1. 在设备主机上安装 raydriver：`pip install raydriver`。已包含网络发现。
2. 运行 `raydriver daemon add /dev/ttyUSB0`（使用您机器的串口；`raydriver scan` 会列出它们）。
   这会创建带生成**令牌**的配置并注册设备。对于非默认固件，添加 `--protocol`/`--baudrate`。
3. 运行 `raydriver daemon install` 安装并启动后台服务（Linux 上为 systemd，macOS 上为 launchd，
   Windows 上为登录任务），或在前台运行 `raydriver daemon`。
4. 若需局域网访问，使用 `raydriver daemon --setup` 打印的 `openssl` 单行命令创建证书，并在配置中
   设置 `tls_cert` 和 `tls_key`。非回环地址上的守护进程在没有令牌和 TLS 时拒绝启动。

完整参考请参阅
[raydriver 守护进程指南](https://github.com/barebaric/raydriver/blob/main/docs/daemon.md)。

### 获取令牌

令牌是 Rayforge 用于连接的共享密钥。它在添加第一个设备时生成，并在那时打印一次。要再次打印，
请**在设备主机上**运行：

```bash
raydriver daemon status
```

复制 `token:` 后面的值——您将在下方把它粘贴到 Rayforge 中。

## 连接 Rayforge

机器向导会在网络中发现守护进程并列出其设备；选择一个即可添加。或者，使用 **Remote machine
(raydriver)** 驱动手动添加机器，输入守护进程的主机名、端口和令牌；然后从守护进程读取的列表中
选择设备：

![向导 — 远程连接](/screenshots/remote-wizard-connect.webp)

- **守护进程主机名 / 端口**——运行守护进程的计算机（默认端口 `8765`）。
- **令牌**——设备主机上 `raydriver daemon status` 显示的值。
- **设备**——守护进程上的机器，设置主机名和令牌后从其设备列表中读取。
- TLS 字段仅对通过网络访问的守护进程显示；同一台计算机（`127.0.0.1`）上的守护进程不需要证书。

探测可通过远程连接进行，因此向导能识别未连接到您计算机的硬件。

## 机器设置

远程机器的设置显示守护进程端点和设备，而非本地串口：

![机器设置 — 远程机器](/screenshots/remote-settings.webp)

## 任务与重新连接

- 已提交的任务会在设备主机上运行至完成，即使 Rayforge 关闭或网络中断。
- 重新连接即可查看精确进度。同一时间只有一个客户端持有控制权；其他客户端可以观看。

## 设备主机上的摄像头

视频不通过控制连接传输。使用 Rayforge 的[网络摄像头](camera.md)，并配合设备主机上可访问的流：

- **IP 摄像头**：直接添加其流 URL。
- **设备主机上的 USB 摄像头**：运行标准流媒体程序（如 `ustreamer`），并将其 MJPEG URL 作为
  HTTP 流摄像头添加。

守护进程会通告已配置的摄像头端点，以便向导预填 URL。

## 安全说明

- 仅限局域网（或隧道）——无云、无账户。
- 任何非回环守护进程都必须使用令牌和 TLS。
- 将摄像头流限制在局域网内或置于防火墙之后；它们与 TLS 控制通道相互独立。
