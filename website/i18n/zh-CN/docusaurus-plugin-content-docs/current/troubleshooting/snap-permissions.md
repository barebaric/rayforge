# Snap 权限（Linux）

本页面介绍在 Linux 上将 Rayforge 作为 Snap 软件包安装时如何配置权限。

## 什么是 Snap 权限？

Snap 是出于安全考虑在沙箱中运行的容器化应用程序。默认情况下，它们对系统资源的访问有限。要使用某些功能（如激光控制器的串口），您必须明确授予权限。

## 必需权限

Rayforge 需要连接这些 Snap 接口才能完全运行：

| 接口              | 用途                             | 是否必需？           |
| ----------------- | -------------------------------- | -------------------- |
| `serial-port`     | 访问 USB 串口设备（激光控制器）  | **是**（用于机器控制）|
| `home`            | 读写主目录中的文件               | 自动连接             |
| `removable-media` | 访问外部驱动器和 USB 存储        | 可选                 |
| `network`         | 网络连接（用于更新等）           | 自动连接             |

---

## 授予串口访问权限

**这是 Rayforge 最重要的权限。**

### 前提条件：dialout 组成员资格

在基于 Debian 的发行版上，即使使用 Snap 软件包，您的用户也必须是
`dialout` 组的成员。如果没有此组成员资格，您在尝试访问串口时可能
会收到 AppArmor DENIED 消息。

```bash
# 将您的用户添加到 dialout 组
sudo usermod -a -G dialout $USER
```

**重要：** 您必须注销并重新登录（或重启）才能使组更改生效。

### 检查当前权限

```bash
# 查看 Rayforge 的所有连接
snap connections rayforge
```

查找 `serial-port` 接口。如果显示"disconnected"或"-"，您需要连接它。

### 连接串口接口

```bash
# 授予串口访问权限
sudo snap connect rayforge:serial-port
```

**您只需要执行一次。** 该权限在应用更新和重启后保持有效。

### 验证连接

```bash
# 检查 serial-port 是否已连接
snap connections rayforge | grep serial-port
```

预期输出：
```
serial-port     rayforge:serial-port     :serial-port     -
```

如果您看到插头/插槽指示符，则连接已激活。

---

## 授予可移动媒体访问权限

如果您想从 USB 驱动器或外部存储导入/导出文件：

```bash
# 授予可移动媒体访问权限
sudo snap connect rayforge:removable-media
```

现在您可以访问 `/media` 和 `/mnt` 中的文件。

---

## Snap 权限故障排除

### 串口仍然无法工作

**连接接口后：**

1. **重新插拔 USB 设备：**
   - 拔掉激光控制器
   - 等待 5 秒
   - 重新插入

2. **重启 Rayforge：**
   - 完全关闭 Rayforge
   - 从应用程序菜单重新启动或：
     ```bash
     snap run rayforge
     ```

3. **检查端口是否出现：**
   - 打开 Rayforge → 设置 → 机器
   - 在下拉列表中查找串口
   - 应该看到 `/dev/ttyUSB0`、`/dev/ttyACM0` 或类似的

4. **验证设备存在：**
   ```bash
   # 列出 USB 串口设备
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### 尽管接口已连接仍显示"Permission Denied"

这种情况很少见，但可能发生在以下情况：

1. **Snap 安装损坏：**
   ```bash
   # 重新安装 snap
   sudo snap refresh rayforge --devmode
   # 或如果失败：
   sudo snap remove rayforge
   sudo snap install rayforge
   # 重新连接接口
   sudo snap connect rayforge:serial-port
   ```

2. **udev 规则冲突：**
   - 检查 `/etc/udev/rules.d/` 中的自定义串口规则
   - 它们可能与 Snap 的设备访问冲突

3. **AppArmor 拒绝：**
   ```bash
   # 检查 AppArmor 拒绝
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   如果您看到串口的拒绝，可能存在 AppArmor 配置文件冲突。

### 无法访问主目录外的文件

**按设计**，Snap 无法访问主目录外的文件，除非您授予 `removable-media`。

**变通方案：**

1. **将文件移动到主目录：**
   ```bash
   # 将 SVG 文件复制到 ~/Documents
   cp /some/other/location/*.svg ~/Documents/
   ```

2. **授予 removable-media 访问权限：**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **使用 Snap 的文件选择器：**
   - 内置文件选择器具有更广泛的访问权限
   - 通过 文件 → 打开 打开文件，而不是命令行参数

---

## 手动接口管理

### 列出所有可用接口

```bash
# 查看系统上的所有 Snap 接口
snap interface
```

### 断开接口

```bash
# 断开 serial-port（如果需要）
sudo snap disconnect rayforge:serial-port
```

### 断开后重新连接

```bash
sudo snap connect rayforge:serial-port
```

---

## 替代方案：从源代码安装

如果 Snap 权限对您的工作流程限制太多：

**选项 1：从源代码构建**

```bash
# 克隆仓库
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# 使用 pixi 安装依赖
pixi install

# 运行 Rayforge
pixi run rayforge
```

**优点：**
- 无权限限制
- 完全系统访问
- 更容易调试
- 最新开发版本

**缺点：**
- 手动更新（git pull）
- 更多依赖需要管理
- 无自动更新

**选项 2：使用 Flatpak（如果可用）**

Flatpak 有类似的沙箱机制，但有时有不同的权限模型。检查 Rayforge 是否提供 Flatpak 软件包。

---

## Snap 权限最佳实践

### 只连接您需要的

不要连接您不使用的接口：

- ✅ 如果使用激光控制器则连接 `serial-port`
- ✅ 如果从 USB 驱动器导入则连接 `removable-media`
- ❌ 不要"以防万一"连接所有接口 - 这违背了安全目的

### 验证 Snap 来源

始终从官方 Snap Store 安装：

```bash
# 检查发布者
snap info rayforge
```

查找：
- 已验证的发布者
- 官方仓库来源
- 定期更新

---

## 理解 Snap 沙箱

### Snap 默认可以访问什么？

**允许：**
- 主目录中的文件
- 网络连接
- 显示/音频

**未经明确许可不允许：**
- 串口（USB 设备）
- 可移动媒体
- 系统文件
- 其他用户的主目录

### 为什么这对 Rayforge 很重要

Rayforge 需要：

1. **主目录访问**（自动授予）
   - 保存项目文件
   - 读取导入的 SVG/DXF 文件
   - 存储首选项

2. **串口访问**（必须授予）
   - 与激光控制器通信
   - **这是关键权限**

3. **可移动媒体**（可选）
   - 从 USB 驱动器导入文件
   - 将 G-code 导出到外部存储

---

## 调试 Snap 问题

### 启用详细 Snap 日志

```bash
# 带调试输出运行 Snap
snap run --shell rayforge
# 在 snap shell 内：
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### 检查 Snap 日志

```bash
# 查看 Rayforge 日志
snap logs rayforge

# 实时跟踪日志
snap logs -f rayforge
```

### 检查系统日志中的拒绝

```bash
# 查找 AppArmor 拒绝
sudo journalctl -xe | grep DENIED | grep rayforge

# 查找 USB 设备事件
sudo journalctl -f -u snapd
# 然后插入您的激光控制器
```

---

## 获取帮助

如果您仍有 Snap 相关问题：

1. **首先检查权限：**
   ```bash
   snap connections rayforge
   ```

2. **尝试串口测试：**
   ```bash
   # 如果您安装了 screen 或 minicom
   sudo snap connect rayforge:serial-port
   # 然后在 Rayforge 中测试
   ```

3. **报告问题时包含：**
   - `snap connections rayforge` 的输出
   - `snap version` 的输出
   - `snap info rayforge` 的输出
   - 您的 Ubuntu/Linux 发行版版本
   - 确切的错误消息

4. **考虑替代方案：**
   - 从源代码安装（见上文）
   - 使用不同的软件包格式（AppImage、Flatpak）

---

## 快速参考命令

```bash
# 授予串口访问权限（最重要）
sudo snap connect rayforge:serial-port

# 授予可移动媒体访问权限
sudo snap connect rayforge:removable-media

# 检查当前连接
snap connections rayforge

# 查看 Rayforge 日志
snap logs rayforge

# 刷新/更新 Rayforge
sudo snap refresh rayforge

# 删除并重新安装（最后手段）
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## 相关页面

- [连接问题](connection) - 串口连接故障排除
- [调试模式](debug) - 启用诊断日志
- [安装](../getting-started/installation) - 安装指南
- [常规设置](../machine/general) - 机器设置
