# G 代码设置

机器设置中的 G 代码页面配置 Rayforge 如何为您的机器生成 G 代码。

![G 代码设置](/screenshots/machine-gcode.png)

## G 代码方言

选择与您的控制器固件匹配的 G 代码方言。不同的控制器使用略有不同的命令和格式。

### 可用方言

- **Grbl (Compat)**：业余激光切割机的标准 GRBL 方言。使用 M3/M5 进行激光控制。
- **Grbl (Compat, no Z axis)**：与 Grbl (Compat) 相同，但没有 Z 轴命令。用于仅 2D 的机器。
- **GRBL Dynamic**：使用 GRBL 的动态激光功率模式进行可变功率雕刻。
- **GRBL Dynamic (no Z axis)**：没有 Z 轴命令的动态模式。
- **Smoothieware**：用于 Smoothieboard 和类似控制器。
- **Marlin**：用于基于 Marlin 的控制器。

:::info
方言影响激光功率、移动和其他命令在输出 G 代码中的格式。
:::

## 方言前导和后缀

每个方言包含可在作业开始和结束时运行的可自定义前导和后缀 G 代码。

### 前导

在每个作业开始时执行的 G 代码命令，在任何切割操作之前。常见用途包括设置单位（G21 表示毫米）、定位模式（G90 表示绝对）和初始化机器状态。

### 后缀

在每个作业结束时执行的 G 代码命令，在所有切割操作之后。常见用途包括关闭激光（M5）、返回原点（G0 X0 Y0）和停放头。

## 另请参阅

- [G 代码基础](../general-info/gcode-basics) - 理解 G 代码
- [G 代码方言](../reference/gcode-dialects) - 详细的方言差异
- [钩子和宏](hooks-macros) - 自定义 G 代码注入点
