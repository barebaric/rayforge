---
description:
  "Rayforge 草图绘制器中的草图参数、约束表达式和文本框模板：用命名值和公式驱动几何图形和标签。"
---

# 表达式与参数

当草图的尺寸由命名值而不是硬编码的数字驱动时，草图才真正成为参数化草图。草图绘制器在两处支持这一点：尺寸约束接受**表达式**，文本框接受
**模板表达式**。两者都由求解器求值，因此每当某个值发生变化，草图都会自动更新。

## 草图参数

每个草图都有自己的参数列表，显示在草图编辑器左侧的 **草图参数**
面板中。**添加参数**可以创建一个参数，可在整数、浮点数、滑块或单行文本之间选择。每个参数都有一个名称——即
`key` 列——表达式引用的就是这个名称。

对于壁厚可变的盒子，一个典型的设置是两个参数：`width` 和
`thickness`。此时还没有任何东西约束几何图形；在表达式使用它们之前，参数只是数字的名称。

## 约束中的表达式

双击一个尺寸约束（参见[约束](constraints.md)），输入一个表达式而不是纯数字：

```
width / 2
```

约束的值就成为该表达式的结果，并在草图每次求解时重新计算。更改 `width`
参数，受约束的几何图形就会随之更新——现在一次编辑就能更新所有引用它的尺寸。由表达式驱动的约束会以橙色绘制标记，标签显示计算出的值。

表达式可以将参数与算术运算和标准 Python 数学函数组合使用：

```
width - 2 * thickness
sqrt(area) / 2
2 * pi * radius
```

像 `sqrt`、`sin`、`cos` 和 `tan` 这样的函数，以及像 `pi` 这样的常量，都来自 Python 的 `math`
模块——这个模块加上参数本身，正是约束表达式所能引用的全部内容。字符串参数也可以被引用，这主要用于文本框。

## 文本框中的模板表达式 {#template-expressions-in-text-boxes}

文本框会在求解时解析花括号括起来的表达式，因此标签和雕刻文本会显示实时值：

```
W = {width}, H = {height}
```

任何参数都可以按名称替换，结果可以用冒号后的 Python 格式说明符进行格式化：

- `{width}` — 参数 `width` 的当前值
- `{name}` — 字符串类型参数的值
- `{width:.1f}` — 一位小数
- `{timestamp():.0f}` — 函数结果无小数

这里同样可以使用数学运算，既可以写成 `{width * 2}` 这样的表达式，也可以通过 `{sqrt(area):.2f}`
这样的函数使用。与约束表达式相比，文本模板拥有更丰富的工具箱：除了数学模块之外，它们还暴露下方的内置函数，并且可以为它们注册自定义函数（见[下文](#custom-template-functions)）。

### 内置模板函数

| 函数            | 返回类型   | 描述                                        |
| --------------- | ---------- | ------------------------------------------- |
| `{today()}`     | `date`     | 当前 UTC 日期（如 `2026-08-26`）            |
| `{date()}`      | `date`     | `today()` 的别名                            |
| `{now()}`       | `datetime` | 当前 UTC 日期和时间                         |
| `{time()}`      | `time`     | 当前 UTC 时间（如 `15:30:00.123456+00:00`） |
| `{timestamp()}` | `float`    | Unix 时间戳（自纪元以来的秒数）             |
| `{uuid4()}`     | `str`      | 8 字符十六进制字符串（如 `a1b2c3d4`）       |
| `{uuid8()}`     | `str`      | `uuid4()` 的别名                            |
| `{uuid()}`      | `str`      | 完整 UUID v4 字符串（36 字符）              |

典型用途包括每次求解生成唯一序列号（`零件 #{uuid4()}`）、实时尺寸标签（`宽={width:.1f} 高={height:.1f}`）、日期戳（`日期：{today()}`）、生产计数器（`{name} - {count:.0f}个`），或用于生产日志的 Unix 时间戳（`{timestamp():.0f}`）。

## 自定义模板函数 {#custom-template-functions}

你可以注册自己的函数，在文本框模板中使用。这对于从数据库获取序列号、读取外部数据或生成自定义标签非常有用。

### 编写注册脚本

创建一个 Python 文件（如 `~/.config/rayforge/my_functions.py`）：

```python
"""Register custom template functions for text box expressions."""
import sqlite3

from sketcher.core.template_functions import register_template_function

DB_PATH = "/home/you/production.db"


def next_serial() -> str:
    """Fetch and reserve the next serial number from the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE counters SET value = value + 1 "
            "WHERE name = 'serial' RETURNING value"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()

register_template_function("next_serial", next_serial)
```

对每个函数调用
`register_template_function(name, callable)`。函数可以做任何 Python 能做的事——打开文件、连接数据库、调用 API——并且它在
**每次渲染**时都会被调用，所以应该很快（如果底层数据在渲染之间不会变化，可以使用缓存）。如果你的 callable 是线程安全的，函数就是线程安全的。

### 运行带有脚本的 Rayforge

使用 `--script` 标志在窗口打开之前加载你的函数：

```bash
rayforge --script ~/.config/rayforge/my_functions.py mydoc.ryp
```

这会在启动早期运行你的脚本——在加载插件之前、在创建主窗口之前——因此函数在草图首次求解时就已可用。

### 在文本框中使用函数

在草图绘制器中，创建一个内容如下的文本框：

```
{next_serial()}
```

格式说明符也可以使用：

```
{next_serial():>20}
```

### 编程式注册函数

如果你正在编写插件或可重用库，可以从任何在草图求解之前运行的 Python 代码中调用
`register_template_function`：

```python
from sketcher.core.template_functions import register_template_function

register_template_function("part_number", lambda: f"P-{hash('x') % 10000:04d}")
```

### 内置函数不能被删除

内置函数（`today`、`now`、`uuid`
等）不能被注销。如果需要改变它们的行为，请用不同的名称注册一个函数。
