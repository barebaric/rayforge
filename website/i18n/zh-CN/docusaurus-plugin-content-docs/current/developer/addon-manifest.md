# 插件清单文件

每个插件都需要在其根目录中有一个 `rayforge-addon.yaml` 文件。此清单文件告诉 Rayforge 关于您的插件的信息——其名称、提供的内容以及如何加载它。

## 基本结构

以下是包含所有常用字段的完整清单：

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## 必填字段

### `name`

您的插件的唯一标识符。这必须是有效的 Python 模块名称——只能包含字母、数字和下划线，且不能以数字开头。

```yaml
name: my_custom_addon
```

### `display_name`

在 UI 中显示的可读名称。可以包含空格和特殊字符。

```yaml
display_name: "My Custom Addon"
```

### `description`

简要描述您的插件的功能。这会显示在插件管理器中。

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

您的插件目标的 API 版本。这必须至少为 1（最低支持版本），最多为当前版本（9）。使用高于支持的版本将导致您的插件验证失败。

```yaml
api_version: 9
```

请参阅[钩子](./addon-hooks.md#api-version-history)文档了解每个版本的变更内容。

### `author`

关于插件作者的信息。`name` 字段是必需的；`email` 是可选的，但建议提供以便用户联系您。

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## 可选字段

### `url`

指向您的插件主页或仓库的 URL。

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Rayforge 本身的版本约束。指定您的插件所需的最低版本。

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

对其他插件的依赖。列出带有版本约束的插件名称。

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

您的插件版本号。这通常从 git 标签自动确定，但您可以明确指定。使用语义版本控制（例如 `1.0.0`）。

```yaml
version: 1.0.0
```

## 入口点

`provides` 部分定义您的插件为 Rayforge 提供的内容。

### 后端（Backend）

后端模块在主进程和工作进程中都会加载。将其用于机器驱动程序、步骤类型、ops 生成器和任何核心功能。

```yaml
provides:
  backend: my_addon.backend
```

该值是相对于您的插件目录的点分 Python 模块路径。

### 前端（Frontend）

前端模块仅在主进程中加载。将其用于 UI 组件、GTK 小部件以及任何需要主窗口的内容。

```yaml
provides:
  frontend: my_addon.frontend
```

### 资源（Assets）

您可以捆绑 Rayforge 将识别的资源文件。每个资源都有一个路径和类型：

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

`path` 相对于您的插件根目录且必须存在。资源类型由 Rayforge 定义，可能包括机器配置文件、材料库或模板等内容。

## 许可证信息

`license` 字段描述您的插件的许可证方式。对于免费插件，只需使用 SPDX 标识符指定许可证名称：

```yaml
license:
  name: MIT
```

常见的 SPDX 标识符包括 `MIT`、`Apache-2.0`、`GPL-3.0` 和 `BSD-3-Clause`。

## 付费插件

Rayforge 通过 Gumroad 许可证验证支持付费插件。如果您想销售您的插件，可以将其配置为在运行之前需要有效的许可证。

### 基本付费配置

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

当 `required` 为 true 时，Rayforge 将在加载您的插件之前检查有效许可证。`purchase_url` 会显示给没有许可证的用户。

### Gumroad 产品 ID

添加您的 Gumroad 产品 ID 以启用许可证验证：

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

对于多个产品 ID（例如不同的定价层级）：

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### 完整的付费插件示例

以下是付费插件的完整清单：

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### 在代码中检查许可证状态

在您的插件代码中，您可以检查许可证是否有效：

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## 验证规则

Rayforge 在加载插件时验证您的清单。以下是规则：

`name` 必须是有效的 Python 标识符（字母、数字、下划线，无前导数字）。`api_version` 必须是介于 1 和当前版本之间的整数。`author.name` 不能为空或包含占位符文本如 "your-github-username"。入口点必须是有效的模块路径且模块必须存在。资源路径必须是相对的（没有 `..` 或前导 `/`）且文件必须存在。

如果验证失败，Rayforge 会记录错误并跳过您的插件。在开发期间检查控制台输出以发现这些问题。
