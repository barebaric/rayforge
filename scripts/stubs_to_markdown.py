"""Generate Docusaurus markdown API docs from Python .pyi type stubs.

Parses each .pyi stub file using the `ast` module (stdlib), extracts
docstrings and signatures, and writes one markdown page per module into
the output directory.

Re-export-only modules (modules whose body is only imports, docstring,
and __all__) are skipped -- they exist solely for backward compatibility
and have no unique content to document.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path


def module_name_from_path(rel_path: str, root_module: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    if parts[-1] == "__init__.pyi":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".pyi", "")
    dotted = ".".join(parts)
    if not dotted:
        return root_module
    if not dotted.startswith(root_module):
        return f"{root_module}.{dotted}"
    return dotted


def output_path_from_rel(
    rel_path: str, output_dir: Path, root_module: str
) -> Path:
    mod = module_name_from_path(rel_path, root_module)
    return output_dir / f"{mod}.md"


def find_stub_files(stubs_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for f in sorted(stubs_dir.rglob("*.pyi")):
        rel = f.relative_to(stubs_dir).as_posix()
        results.append((rel, f))
    return results


def get_docstring(body: list[ast.stmt]) -> str | None:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        val = body[0].value
        if isinstance(val.value, str):
            return val.value
    return None


def trailing_docstring(body: list[ast.stmt], idx: int) -> str | None:
    nxt = idx + 1
    if nxt < len(body):
        stmt = body[nxt]
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            val = stmt.value
            if isinstance(val.value, str):
                return val.value
    return None


SPHINX_REF_RE = re.compile(
    r":(?:class|meth|func|attr|data|mod|obj|py:class|py:meth|"
    r"py:func|py:attr|py:data):`([^`]+)`"
)


FENCE_PLACEHOLDER = "\x00FENCE\x00"

def convert_sphinx(text: str) -> str:
    text = SPHINX_REF_RE.sub(r"**\1**", text)
    # Protect triple-backtick code fences from the `` -> ` replacement
    text = text.replace("```", FENCE_PLACEHOLDER)
    text = text.replace("``", "`")
    text = text.replace(FENCE_PLACEHOLDER, "```")
    return text


PARAM_RE = re.compile(r"^:param\s+(\w[\w_]*):\s*(.*)")
RETURN_RE = re.compile(r"^:returns?:\s*(.*)")
RAISE_RE = re.compile(r"^:raises?\s+(\w[\w_]*):\s*(.*)")


def convert_docstring_sections(text: str) -> tuple[str, dict[str, str]]:
    """Parse docstring and return (description_text, param_descriptions)."""
    lines = text.split("\n")
    result = []
    param_docs: dict[str, str] = {}
    i = 0
    in_args = False
    in_returns = False
    in_raises = False
    in_attrs = False
    in_example = False
    example_lines = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("Attributes:") or stripped.startswith(
            "Attributes:"
        ):
            in_attrs = True
            in_args = False
            in_returns = False
            in_raises = False
            result.append("")
            result.append("**Attributes:**")
            result.append("")
            i += 1
            continue

        if in_attrs:
            if stripped == "":
                in_attrs = False
                result.append("")
                i += 1
                continue
            m = re.match(r"^\s+(\w[\w_]*):\s*(.*)", line)
            if m:
                aname = m.group(1)
                adesc = convert_sphinx(m.group(2))
                result.append(f"* ``{aname}`` — {adesc}")
            elif stripped:
                result.append(stripped)
            i += 1
            continue

        if stripped == "Example::" or stripped.startswith("Example::"):
            in_example = True
            example_lines = []
            result.append("")
            result.append("**Example:**")
            result.append("")
            i += 1
            continue

        if in_example:
            if stripped == "" and example_lines:
                pass
            elif stripped.startswith("```") or stripped.startswith(".."):
                in_example = False
                result.extend(example_lines)
                continue
            else:
                example_lines.append(line if stripped else "")
            i += 1
            if i >= len(lines) or (
                i < len(lines) and lines[i].strip() == "" and not in_example
            ):
                continue
            if (
                i < len(lines)
                and not lines[i].startswith(" ")
                and not lines[i].strip() == ""
            ):
                in_example = False
                result.extend(example_lines)
                continue
            i += 1
            continue

        if stripped.startswith("Args:") or stripped.startswith("Parameters:"):
            in_args = True
            in_returns = False
            in_raises = False
            i += 1
            continue

        if stripped.startswith("Returns:") or stripped.startswith("Return:"):
            in_args = False
            in_returns = True
            in_raises = False
            result.append("")
            ret_desc = convert_sphinx(stripped.split(":", 1)[1].strip())
            result.append(f"**Returns:** {ret_desc}")
            i += 1
            continue

        if stripped.startswith("Raises:") or stripped.startswith("Raise:"):
            in_args = False
            in_returns = False
            in_raises = True
            i += 1
            continue

        if in_args or in_raises:
            if stripped == "":
                in_args = False
                in_raises = False
                i += 1
                continue
            m = re.match(r"^\s+(\w[\w_]*):\s*(.*)", line)
            if m:
                pname = m.group(1)
                pdesc = convert_sphinx(m.group(2))
                if in_args:
                    param_docs[pname] = pdesc
                else:
                    result.append(f"* ``{pname}`` — {pdesc}")
            elif stripped:
                result.append(stripped)
            i += 1
            continue

        if in_returns:
            if stripped == "" and (
                i + 1 >= len(lines) or lines[i + 1].strip() == ""
            ):
                in_returns = False
                result.append("")
                i += 1
                continue
            if stripped == "":
                result.append("")
                i += 1
                continue
            if (
                not line.startswith(" ")
                and not stripped.startswith(":")
                and not stripped.startswith("``")
            ):
                in_returns = False
                continue
            stripped_desc = stripped
            if stripped_desc.startswith(":") and ":" in stripped_desc[1:]:
                in_returns = False
                continue
            result.append(convert_sphinx(line.rstrip()))
            i += 1
            continue

        m = PARAM_RE.match(stripped)
        if m:
            param_docs[m.group(1)] = convert_sphinx(m.group(2))
            i += 1
            continue

        m = RETURN_RE.match(stripped)
        if m:
            result.append("")
            result.append(f"**Returns:** {convert_sphinx(m.group(1))}")
            i += 1
            continue

        m = RAISE_RE.match(stripped)
        if m:
            result.append("")
            result.append(
                f"**Raises:** ``{m.group(1)}`` — {convert_sphinx(m.group(2))}"
            )
            i += 1
            continue

        result.append(convert_sphinx(line.rstrip()))
        i += 1

    if in_example and example_lines:
        result.extend(example_lines)

    return str("\n".join(result)), param_docs


def dedent_doc(text: str) -> str:
    lines = text.split("\n")
    if len(lines) <= 1:
        return text
    non_empty = [ln for ln in lines[1:] if ln.strip()]
    if not non_empty:
        return lines[0].strip()
    indent = min((len(ln) - len(ln.lstrip()) for ln in non_empty), default=0)
    result = [lines[0]]
    for ln in lines[1:]:
        if ln.strip():
            result.append(ln[indent:] if len(ln) > indent else ln)
        else:
            result.append("")
    return "\n".join(result)


def format_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    if not doc:
        return "", {}
    text = dedent_doc(doc)
    text = convert_sphinx(text)
    text, param_docs = convert_docstring_sections(text)
    text = text.strip()
    return text, param_docs


def clean_type(text: str) -> str:
    """Strip superfluous module prefixes from type annotations."""
    text = text.replace("builtins.", "")
    text = text.replace("typing.", "")
    text = text.replace("raygeo.", "")
    return text


def format_annotation(node: ast.expr | None) -> str:
    if node is None:
        return ""
    try:
        return clean_type(ast.unparse(node))
    except Exception:
        return ""


def format_default(node: ast.expr | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return ""
    try:
        return f" = {ast.unparse(node)}"
    except Exception:
        return ""


def format_function_signature(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    skip_self: bool = True,
) -> str:
    parts = []
    args = func.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(
        args.defaults
    )

    start = (
        1
        if (skip_self and args.args and args.args[0].arg in ("self", "cls"))
        else 0
    )
    for idx in range(start, len(args.args)):
        arg = args.args[idx]
        d = (
            defaults[idx]
            if idx >= len(args.args) - len(args.defaults)
            else None
        )
        sig = arg.arg
        if arg.annotation:
            sig += str(f": {format_annotation(arg.annotation)}")
        def_str = format_default(d)
        if def_str:
            sig += def_str
        parts.append(sig)

    if args.vararg:
        var = str(f"*{args.vararg.arg}")
        if args.vararg.annotation:
            var += str(f": {format_annotation(args.vararg.annotation)}")
        parts.append(var)

    seen_kwonly = False
    for idx, arg in enumerate(args.kwonlyargs):
        if not seen_kwonly and not args.vararg:
            parts.append("*")
            seen_kwonly = True
        d = args.kw_defaults[idx] if idx < len(args.kw_defaults) else None
        sig = arg.arg
        if arg.annotation:
            sig += str(f": {format_annotation(arg.annotation)}")
        def_str = format_default(d)
        if def_str:
            sig += def_str
        parts.append(sig)

    if args.kwarg:
        kw = str(f"**{args.kwarg.arg}")
        if args.kwarg.annotation:
            kw += str(f": {format_annotation(args.kwarg.annotation)}")
        parts.append(kw)

    sig = str(f"({', '.join(parts)})")
    if func.returns:
        sig += str(f" -> {format_annotation(func.returns)}")
    return sig


def extract_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict]:
    """Extract structured parameter info from AST function definition."""
    args = func.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(
        args.defaults
    )
    params = []

    start = 1 if (args.args and args.args[0].arg in ("self", "cls")) else 0
    for idx in range(start, len(args.args)):
        arg = args.args[idx]
        ann = format_annotation(arg.annotation)
        def_str = ""
        if idx >= len(args.args) - len(args.defaults):
            def_node = defaults[idx]
            if def_node is not None:
                def_str = format_default(def_node)
        type_str = f"{ann}" if ann else ""
        if def_str:
            type_str = (
                f"{type_str}{def_str}" if type_str else def_str.strip(" = ")
            )
        params.append({"name": arg.arg, "type": type_str})

    if args.vararg:
        var = f"*{args.vararg.arg}"
        ann = format_annotation(args.vararg.annotation)
        params.append({"name": var, "type": ann})

    for arg in args.kwonlyargs:
        ann = format_annotation(arg.annotation)
        params.append({"name": arg.arg, "type": ann})

    if args.kwarg:
        kw = f"**{args.kwarg.arg}"
        ann = format_annotation(args.kwarg.annotation)
        params.append({"name": kw, "type": ann})

    return params


def is_property(item: ast.stmt) -> bool:
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in item.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "property":
                return True
            if isinstance(dec, ast.Name) and dec.id == "property":
                return True
    return False


def is_setter(item: ast.stmt) -> bool:
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in item.decorator_list:
            if isinstance(dec, ast.Attribute) and "setter" in dec.attr:
                return True
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and "setter" in dec.func.attr
            ):
                return True
    return False


def is_classmethod(item: ast.stmt) -> bool:
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in item.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "classmethod":
                return True
            if isinstance(dec, ast.Attribute) and dec.attr == "classmethod":
                return True
    return False


def collect_members(body: list[ast.stmt]) -> dict:
    members = {
        "type_aliases": [],
        "constants": [],
        "classes": [],
        "functions": [],
        "re_exports": [],
    }
    idx = -1
    for item in body:
        idx += 1
        if isinstance(item, ast.AnnAssign) and isinstance(
            item.target, ast.Name
        ):
            name = item.target.id
            ann_str = format_annotation(item.annotation)

            is_alias = "TypeAlias" in ann_str or (
                isinstance(item.annotation, ast.Subscript)
                and isinstance(item.annotation.value, ast.Name)
                and item.annotation.value.id in ("TypeAlias",)
            )

            doc = trailing_docstring(body, idx)
            if is_alias or "TypeAlias" in ann_str:
                value_str = (
                    format_annotation(item.value)
                    if item.value is not None
                    else ann_str
                )
                members["type_aliases"].append(
                    {
                        "name": name,
                        "type": value_str,
                        "doc": doc,
                    }
                )
            else:
                members["constants"].append(
                    {
                        "name": name,
                        "type": ann_str,
                        "doc": doc,
                    }
                )

        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    doc = trailing_docstring(body, idx)
                    members["constants"].append(
                        {
                            "name": target.id,
                            "type": "",
                            "doc": doc,
                        }
                    )
            doc = trailing_docstring(body, idx)
            if (
                doc
                and not members["type_aliases"]
                and not members["constants"]
            ):
                pass

        elif isinstance(item, ast.ClassDef):
            doc = get_docstring(item.body)
            cls_info = {
                "name": item.name,
                "doc": doc,
                "methods": [],
                "properties": [],
                "nested_classes": [],
            }
            for sub in item.body:
                if isinstance(sub, ast.ClassDef):
                    sub_doc = get_docstring(sub.body)
                    cls_info["nested_classes"].append(
                        {
                            "name": sub.name,
                            "doc": sub_doc,
                            "methods": [],
                            "properties": [],
                        }
                    )
                elif is_property(sub) and isinstance(sub, ast.FunctionDef):
                    sub_doc = get_docstring(sub.body)
                    ret = format_annotation(sub.returns) if sub.returns else ""
                    cls_info["properties"].append(
                        {
                            "name": sub.name,
                            "doc": sub_doc,
                            "signature": ret,
                        }
                    )
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_setter(sub):
                        continue
                    sub_doc = get_docstring(sub.body)
                    sig = format_function_signature(sub)
                    cls_info["methods"].append(
                        {
                            "name": sub.name,
                            "doc": sub_doc,
                            "signature": sig,
                            "params": extract_params(sub),
                            "is_classmethod": is_classmethod(sub),
                        }
                    )
            members["classes"].append(cls_info)

        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = get_docstring(item.body)
            sig = format_function_signature(item)
            members["functions"].append(
                {
                    "name": item.name,
                    "doc": doc,
                    "signature": sig,
                    "params": extract_params(item),
                }
            )

        elif isinstance(item, ast.ImportFrom):
            if item.module:
                names = [a.name for a in item.names]
                members["re_exports"].append(
                    {
                        "module": item.module,
                        "names": names,
                    }
                )

    return members


def is_reexport_only(tree: ast.Module) -> bool:
    """Return True if the module contains nothing but imports + docstring.

    A "re-export-only" module exists solely for backward compatibility.
    Such modules should not produce a doc page.
    """
    has_content = False
    for item in tree.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    break
            else:
                has_content = True
                break
        else:
            has_content = True
            break
    return not has_content


def generate_page(mod: str, content: str, position: int = 10) -> str:
    return f"""---
title: {mod}
sidebar_label: {mod}
sidebar_position: {position}
---

{content}
"""


def render_members(members: dict, mod_doc: str | None) -> str:
    lines = []

    if mod_doc:
        lines.append(convert_sphinx(mod_doc))
        lines.append("")

    public_aliases = sorted(
        (a for a in members["type_aliases"] if not a["name"].startswith("_")),
        key=lambda x: x["name"],
    )
    public_consts = sorted(
        (c for c in members["constants"] if not c["name"].startswith("_")),
        key=lambda x: x["name"],
    )
    has_alias = bool(public_aliases)
    has_consts = bool(public_consts)
    has_classes = bool(members["classes"])
    has_funcs = bool(members["functions"])
    has_re = bool(members["re_exports"])

    def _render_params(
        params: list[dict],
        param_docs: dict[str, str],
        signature: str = "",
    ) -> None:
        if not params and "->" not in signature:
            return
        lines.append("")
        lines.append("| Parameter | Type | Description |")
        lines.append("| --- | --- | --- |")
        for p in params:
            ptype = p.get("type", "")
            ptype_md = f"`{ptype}`" if ptype else ""
            desc = param_docs.get(p["name"], "")
            lines.append(f"| `{p['name']}` | {ptype_md} | {desc} |")
        if "->" in signature:
            ret = signature.split("->", 1)[1].strip()
            lines.append(f"| *Returns* | `{ret}` |  |")
        lines.append("")

    if has_alias:
        lines.append("## Type Aliases")
        lines.append("")
        for a in public_aliases:
            lines.append(f"### {a['name']}")
            lines.append("")
            lines.append(f"Type: `{a['type']}`")
            lines.append("")
            doc, _ = format_docstring(a["doc"])
            if doc:
                lines.append(doc)
                lines.append("")
        lines.append("")

    if has_consts:
        lines.append("## Constants")
        lines.append("")
        for c in public_consts:
            lines.append(f"### {c['name']}")
            lines.append("")
            if c["type"]:
                lines.append(f"``{c['type']}``")
                lines.append("")
            doc, _ = format_docstring(c["doc"])
            if doc:
                lines.append(doc)
                lines.append("")
        lines.append("")

    if has_classes:
        for cls in sorted(members["classes"], key=lambda x: x["name"]):
            lines.append(f"## {cls['name']}")
            lines.append("")
            doc, _ = format_docstring(cls["doc"])
            if doc:
                lines.append(doc)
                lines.append("")

            public_props = sorted(
                (
                    p
                    for p in cls["properties"]
                    if not p["name"].startswith("_")
                ),
                key=lambda x: x["name"],
            )
            public_methods = sorted(
                (m for m in cls["methods"] if not m["name"].startswith("_")),
                key=lambda x: x["name"],
            )
            public_nested = sorted(
                (
                    n
                    for n in cls["nested_classes"]
                    if not n["name"].startswith("_")
                ),
                key=lambda x: x["name"],
            )

            for p in public_props:
                sig = f": {p['signature']}" if p.get("signature") else ""
                lines.append(f"### `{p['name']}`")
                lines.append("")
                if sig:
                    lines.append(f"``{p['name']}{sig}``")
                    lines.append("")
                pd, _ = format_docstring(p["doc"])
                if pd:
                    lines.append(pd)
                    lines.append("")

            for m in public_methods:
                dec = "@classmethod " if m.get("is_classmethod") else ""
                lines.append(f"### `{m['name']}()`")
                lines.append("")
                lines.append(f"``{dec}{m['name']}{m['signature']}``")
                lines.append("")
                mdoc, param_docs = format_docstring(m["doc"])
                if mdoc:
                    lines.append(mdoc)
                    lines.append("")
                _render_params(
                    m.get("params", []), param_docs, m.get("signature", "")
                )

            for nc in public_nested:
                lines.append(f"### {nc['name']}")
                lines.append("")
                ndoc, _ = format_docstring(nc["doc"])
                if ndoc:
                    lines.append(ndoc)
                    lines.append("")
                nc_public_methods = sorted(
                    (
                        m
                        for m in nc["methods"]
                        if not m["name"].startswith("_")
                    ),
                    key=lambda x: x["name"],
                )
                nc_public_props = sorted(
                    (
                        p
                        for p in nc["properties"]
                        if not p["name"].startswith("_")
                    ),
                    key=lambda x: x["name"],
                )
                for p in nc_public_props:
                    sig = f": {p['signature']}" if p.get("signature") else ""
                    lines.append(f"#### `{p['name']}`")
                    lines.append("")
                    if sig:
                        lines.append(f"``{p['name']}{sig}``")
                        lines.append("")
                    pd, _ = format_docstring(p["doc"])
                    if pd:
                        lines.append(pd)
                        lines.append("")
                for m in nc_public_methods:
                    dec = "@classmethod " if m.get("is_classmethod") else ""
                    lines.append(f"#### `{m['name']}()`")
                    lines.append("")
                    lines.append(f"``{dec}{m['name']}{m['signature']}``")
                    lines.append("")
                    mdoc, param_docs = format_docstring(m["doc"])
                    if mdoc:
                        lines.append(mdoc)
                        lines.append("")
                    _render_params(
                        m.get("params", []),
                        param_docs,
                        m.get("signature", ""),
                    )
        lines.append("")

    if has_funcs:
        lines.append("## Functions")
        lines.append("")
        for f in sorted(members["functions"], key=lambda x: x["name"]):
            lines.append(f"### `{f['name']}()`")
            lines.append("")
            lines.append(f"``{f['name']}{f['signature']}``")
            lines.append("")
            doc, param_docs = format_docstring(f["doc"])
            if doc:
                lines.append(doc)
                lines.append("")
            _render_params(
                f.get("params", []), param_docs, f.get("signature", "")
            )
        lines.append("")

    if (
        has_re
        and not has_funcs
        and not has_classes
        and not has_alias
        and not has_consts
    ):
        lines.append("## Re-exports")
        lines.append("")
        for r in sorted(members["re_exports"], key=lambda x: x["module"]):
            items = ", ".join(r["names"])
            lines.append(
                f"This module re-exports from ``{r['module']}``: {items}."
            )
            lines.append("")

    return str("\n".join(lines))


def process_file(
    rel_path: str,
    filepath: Path,
    root_module: str,
    position: int = 10,
) -> str:
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(
            f"  Warning: syntax error in {rel_path}, skipping", file=sys.stderr
        )
        return ""

    mod_doc = get_docstring(tree.body)
    members = collect_members(tree.body)
    content = render_members(members, mod_doc)
    mod = module_name_from_path(rel_path, root_module)
    return generate_page(mod, content, position)


def generate(
    stubs_dir: Path,
    output_dir: Path,
    root_module: str = "",
) -> None:
    """Generate markdown API docs from .pyi stubs in stubs_dir.

    :param stubs_dir: Directory containing .pyi stub files.
    :param output_dir: Directory to write generated .md files.
    :param root_module: Root Python module name. Inferred from stubs_dir
        name if empty.
    """
    if not root_module:
        root_module = stubs_dir.name

    if not stubs_dir.exists():
        print(f"Input directory not found: {stubs_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_stub_files(stubs_dir)
    if not files:
        print("No .pyi stub files found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    for old in output_dir.glob("*.md"):
        old.unlink()

    all_mods = [module_name_from_path(rel, root_module) for rel, _ in files]

    def has_children(mod: str) -> bool:
        prefix = mod + "."
        return any(m.startswith(prefix) for m in all_mods)

    valid = []
    for rel_path, filepath in files:
        mod = module_name_from_path(rel_path, root_module)
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            print(
                f"  Warning: syntax error in {rel_path}, skipping",
                file=sys.stderr,
            )
            continue
        if is_reexport_only(tree) and not has_children(mod):
            print(f"  {mod} -> skipped (re-export only)")
            continue
        valid.append((rel_path, filepath, mod))

    valid.sort(key=lambda x: ("" if x[2] == root_module else x[2], x[2]))

    for idx, (rel_path, filepath, mod) in enumerate(valid, start=1):
        page = process_file(rel_path, filepath, root_module, idx)
        if not page.strip():
            continue
        out_path = output_path_from_rel(rel_path, output_dir, root_module)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"  {mod} -> {out_path}")

    total = len([f for f in output_dir.iterdir() if f.suffix == ".md"])
    print(f"\nGenerated {total} API doc pages in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown API docs from Python .pyi stubs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing .pyi stub files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write generated .md files.",
    )
    parser.add_argument(
        "--root-module",
        default="",
        help=(
            "Root Python module name (e.g. 'raygeo'). "
            "If empty, inferred from the input directory name."
        ),
    )
    args = parser.parse_args()
    generate(
        Path(str(args.input)),
        Path(str(args.output)),
        str(args.root_module),
    )


if __name__ == "__main__":
    main()
