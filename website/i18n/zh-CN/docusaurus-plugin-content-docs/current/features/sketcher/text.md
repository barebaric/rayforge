---
description: "Rayforge 参数化 2D 草图绘制器中的文本框模板和自定义模板函数。"
---

# 文本模板

文本框支持用花括号括起来的模板表达式。这些表达式在求解时使用当前参数值进行
解析，因此当您更改尺寸或输入变量时，文本会自动更新。

## 变量替换

按名称引用任何草图参数或输入变量：

- `{width}` — 参数"width"的当前值
- `{name}` — 字符串类型输入参数的值
- `{count:.0f}` — 使用 Python 格式说明符格式化（无小数）

## 数学表达式

您可以在模板中使用数学函数：

- `{sqrt(area):.2f}` — "area"的平方根，格式化为 2 位小数
- `{width * 2}` — 算术表达式

标准数学函数（`sqrt`、`sin`、`cos`、`tan`、`pi` 等）均可用。

## 内置函数

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

## 格式说明符

Python 格式说明符可用于任何表达式结果：

- `{width:.1f}` — 一位小数
- `{timestamp():.0f}` — 时间戳无小数
- `{today()}` — 默认字符串表示

## 使用示例

- `零件 #{uuid4()}` — 每次求解时生成唯一序列号
- `宽={width:.1f} 高={height:.1f}` — 实时尺寸标签
- `日期：{today()}` — 为每个零件标注日期
- `{name} - {count:.0f}个` — 组合字符串和数字参数
- `{timestamp():.0f}` — 用于生产日志的 Unix 时间戳

## 自定义模板函数

你可以注册自己的函数用于文本模板。这对于从数据库获取
序列号、读取外部数据或生成自定义标签非常有用。

### 编写注册脚本

创建一个 Python 文件（如
`~/.config/rayforge/my_functions.py`）：

```python
"""为文本模板注册自定义函数。"""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

DB_PATH = "/home/you/production.db"


def next_serial() -> str:
    """从数据库获取下一个序列号。"""
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

关键点：

- 对每个函数调用 `register_template_function(name, callable)`。
- 你的函数可以做任何 Python 能做的事：打开文件、连接
  数据库、调用 API 等。
- 函数在**每次渲染**时都会被调用，所以应该很快。
- 如果你的 callable 是线程安全的，函数就是线程安全的。

### 运行带有脚本的 Rayforge

使用 `--script` 标志在窗口打开前加载你的函数：

```bash
rayforge --script ~/.config/rayforge/my_functions.py \
    my_document.ryp
```

这会在启动早期运行你的脚本——在加载插件之前、在创建
主窗口之前——这样函数在草图首次求解时就可用了。

### 在文本框中使用函数

创建一个文本框：

```
{next_serial()}
```

格式说明符也有效：

```
{next_serial():>20}
```

### 编程式注册函数

如果你正在编写插件或可重用库，可以从任何在草图
求解之前运行的 Python 代码中调用 `register_template_function`：

```python
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function(
    "part_number",
    lambda: f"P-{hash('x') % 10000:04d}"
)
```

### 内置函数不能被删除

内置函数（`today`、`now`、`uuid` 等）不能被注销。
如果需要改变它们的行为，请用不同的名称注册一个函数。
