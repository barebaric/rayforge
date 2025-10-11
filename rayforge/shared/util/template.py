import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.encoder.gcode import GcodeContext
    from ...machine.models.machine import Machine
    from ...machine.models.macro import Macro


class TemplateFormatter:
    """
    Expands a macro by processing variable placeholders (e.g., {obj.attr})
    and @include(Macro Name) directives.
    """

    def __init__(self, machine: "Machine", context_obj: "GcodeContext"):
        """
        Initializes the formatter.

        Args:
            machine: The machine object, needed to look up macros by name.
            context_obj: The object against which variable paths will be
              resolved.
        """
        self._machine = machine
        self._context = context_obj

    def _resolve_variable(self, path: str) -> str:
        """Resolves a dot-notation path like 'machine.dimensions[0]'."""
        # This is the original _resolve method, now with a clearer name.
        try:
            current = self._context
            parts = re.split(r"\.|(\[\d+\])", path)
            clean_parts = [p for p in parts if p]

            for part in clean_parts:
                if part.startswith("[") and part.endswith("]"):
                    index = int(part[1:-1])
                    current = current[index]  # type: ignore
                else:
                    current = getattr(current, part)
            return str(current)
        except (AttributeError, TypeError, IndexError):
            return f"{{{path}}}"

    def format_string(self, template_string: str) -> str:
        """Formats a single line by replacing all variable placeholders."""
        return re.sub(
            r"\{(.+?)\}",
            lambda m: self._resolve_variable(m.group(1)),
            template_string,
        )

    def expand_macro(self, macro: "Macro") -> List[str]:
        """
        Public entry point to fully expand a macro.

        Args:
            macro: The top-level macro (e.g., from a hook) to expand.

        Returns:
            A list of fully expanded G-code lines.
        """
        # The call_stack tracks macro names to prevent infinite recursion.
        return self._recursive_expand(macro, call_stack=set())

    def _recursive_expand(
        self, macro: "Macro", call_stack: set[str]
    ) -> List[str]:
        """
        Recursively expands a macro, processing includes and formatting
        variables.
        """
        output_lines: List[str] = []

        if macro.name in call_stack:
            error_msg = (
                f"; ERROR: Circular dependency detected. Macro "
                f"'{macro.name}' was included again."
            )
            return [error_msg]

        call_stack.add(macro.name)

        for line in macro.code:
            match = re.match(r"^\s*@include\((.*?)\)\s*$", line)
            if match:
                macro_name = match.group(1).strip()
                found_macro = next(
                    (
                        m
                        for m in self._machine.macros.values()
                        if m.name == macro_name
                    ),
                    None,
                )

                if found_macro and found_macro.enabled:
                    # Recurse: expand the included macro
                    expanded_lines = self._recursive_expand(
                        found_macro, call_stack
                    )
                    output_lines.extend(expanded_lines)
                else:
                    output_lines.append(
                        f"; WARNING: Macro '{macro_name}' "
                        f" not found or disabled."
                    )
            else:
                # This is a normal G-code line, format it
                output_lines.append(self.format_string(line))

        # Backtrack: remove the macro from the stack after processing
        call_stack.remove(macro.name)
        return output_lines
