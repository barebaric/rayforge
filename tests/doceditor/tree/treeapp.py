#!/usr/bin/env python3
"""Standalone harness for the tree widgets; not collected by pytest.

Run with: pixi run python tests/doceditor/tree/treeapp.py [parts|operations]
"""

# flake8: noqa: E402
import logging
import sys
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, Gtk

from rayforge.core.color import COLOR_PALETTE
from rayforge.core.doc import Doc
from rayforge.core.group import Group
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workflow import Workflow
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.tree.operations_tree import OperationsTree
from rayforge.doceditor.tree.parts_tree import PartsTree

base_path = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContourStep(Step):
    TYPELABEL = "Contour"
    ICON = "cut-symbolic"

    def __init__(self, name: str | None = None):
        super().__init__(typelabel=self.TYPELABEL, name=name)


class EngraveStep(Step):
    TYPELABEL = "Engrave"
    ICON = "engrave-symbolic"

    def __init__(self, name: str | None = None):
        super().__init__(typelabel=self.TYPELABEL, name=name)


def build_demo_doc() -> Doc:
    doc = Doc()
    doc.set_children([])

    parts = Layer("Parts")
    doc.add_layer(parts)
    parts_workflow = parts.workflows[0]
    ribs = Group("Ribs")
    parts.add_child(ribs)
    for name, width, height in (("Rib L", 70, 40), ("Rib R", 70, 40)):
        workpiece = WorkPiece(name)
        workpiece.natural_width_mm = width
        workpiece.natural_height_mm = height
        ribs.add_child(workpiece)
    fuselage = WorkPiece("Fuselage")
    fuselage.natural_width_mm = 130
    fuselage.natural_height_mm = 60
    parts.add_child(fuselage)
    parts_workflow.add_step(ContourStep(name="Contour outer"))

    names = Layer("Names")
    doc.add_layer(names)
    names_workflow = names.workflows[0]
    names_workflow.set_color(COLOR_PALETTE[1])
    for name in ("Rib L", "Rib R", "Fuselage"):
        workpiece = WorkPiece(f"{name} name")
        workpiece.natural_width_mm = 24
        workpiece.natural_height_mm = 5
        names.add_child(workpiece)
    names_workflow.add_step(EngraveStep(name="Engrave"))

    inner = ContourStep(name="Contour inner")
    inner.workpiece_uids = [wp.uid for wp in ribs.all_workpieces]
    parts_workflow.add_step(inner)

    stock_asset = StockAsset(name="Balsa 3 mm")
    stock_asset.thickness = 3.0
    doc.add_asset(stock_asset)
    parts.add_child(
        StockItem(stock_asset_uid=stock_asset.uid, name="Balsa 3 mm")
    )
    return doc


class TreeApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="org.rayforge.treeapp")
        self.doc = build_demo_doc()
        self.selected_item = None
        self.part_counter = 0

    def do_activate(self):
        win = Gtk.ApplicationWindow(
            application=self,
            default_width=1050,
            default_height=720,
            title="Tree widget harness",
        )
        header = Gtk.HeaderBar()
        header.set_title_widget(Gtk.Label(label="Tree widget harness"))
        win.set_titlebar(header)

        columns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        win.set_child(columns)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar.set_margin_top(8)
        sidebar.set_margin_start(8)
        sidebar.set_margin_end(8)
        switcher = Gtk.StackSwitcher()
        stack = Gtk.Stack()
        switcher.set_stack(stack)

        self.parts_tree = PartsTree()
        self.operations_tree = OperationsTree()
        self.parts_tree.set_doc(self.doc)
        self.operations_tree.set_doc(self.doc)
        stack.add_titled(self.parts_tree, "parts", "Parts")
        stack.add_titled(self.operations_tree, "operations", "Operations")
        page = sys.argv[1] if len(sys.argv) > 1 else "parts"
        stack.set_visible_child_name(page)
        if page == "operations" and self.doc.layers:
            first_workflow = self.doc.layers[0].workflows[0]
            assert first_workflow is not None
            self.operations_tree.expand(first_workflow.uid)
        sidebar.append(switcher)
        sidebar.append(stack)
        sidebar.set_size_request(300, -1)
        columns.append(sidebar)

        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        columns.append(separator)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        controls.set_margin_top(12)
        controls.set_margin_start(12)
        controls.set_margin_end(12)
        info = Gtk.Label(label="Nothing selected", xalign=0)
        info.set_wrap(True)
        controls.append(info)

        for label, handler in (
            ("Add part", self._on_add_part),
            ("Duplicate selected", self._on_duplicate),
            ("Delete selected", self._on_delete),
            ("Remove selected workflow", self._on_remove_workflow),
        ):
            button = Gtk.Button(label=label)
            button.set_halign(Gtk.Align.START)
            button.connect("clicked", handler)
            controls.append(button)

        columns.append(controls)

        self.info = info
        self.parts_tree.selection_changed.connect(self._on_selection)
        self.operations_tree.selection_changed.connect(self._on_selection)
        self.parts_tree.visibility_toggled.connect(self._on_visibility)
        self.parts_tree.rename_requested.connect(self._on_rename)
        self.operations_tree.rename_requested.connect(self._on_rename)
        self.operations_tree.workflow_requested.connect(
            self._on_workflow_requested
        )
        self.operations_tree.workflow_activated.connect(
            self._on_workflow_activated
        )

        win.present()

    def _on_selection(self, sender, spec):
        self.selected_item = self.doc.find_descendant_by_uid(spec.uid)
        self.info.set_label(f"{spec.kind}: {spec.label} ({spec.uid})")

    def _on_visibility(self, sender, item, on):
        item.set_visible(on)

    def _on_rename(self, sender, uid, name):
        item = self.doc.find_descendant_by_uid(uid)
        if item is not None:
            item.name = name

    def _on_add_part(self, button):
        self.part_counter += 1
        workpiece = WorkPiece(f"Part {self.part_counter}")
        workpiece.natural_width_mm = 30
        workpiece.natural_height_mm = 20
        self.doc.active_layer.add_child(workpiece)

    def _on_duplicate(self, button):
        item = self.selected_item
        if item is None or item.parent is None:
            return
        duplicate = item.duplicate()
        item.parent.add_child(duplicate)

    def _on_delete(self, button):
        item = self.selected_item
        if item is None or item.parent is None:
            return
        item.parent.remove_child(item)
        self.selected_item = None
        self.info.set_label("Nothing selected")

    def _on_remove_workflow(self, button):
        if self.selected_item is None:
            return
        if not isinstance(self.selected_item, Workflow):
            return
        layer = self.selected_item.parent
        if layer is not None:
            layer.remove_child(self.selected_item)
        self.selected_item = None
        self.info.set_label("Nothing selected")

    def _on_workflow_requested(self, sender, color):
        layer = self.doc.active_layer
        workflow = Workflow(f"Workflow {len(layer.workflows) + 1}")
        workflow.set_color(color)
        layer.add_child(workflow)

    def _on_workflow_activated(self, sender, workflow):
        self.info.set_label(f"Workflow picked: {workflow.name}")


if __name__ == "__main__":
    app = TreeApp()
    app.run([])
