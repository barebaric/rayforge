# rayforge/widgets/about.py

import sys
import platform
import webbrowser
import logging
from importlib.metadata import version, PackageNotFoundError
from gi.repository import Gtk, Adw, GLib  # type: ignore
from ... import __version__
from ...icons import get_icon


logger = logging.getLogger(__name__)
_not_found_str = _("Not found")


def _get_version(package_name: str) -> str:
    """Safely retrieves the version of a Python package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return _not_found_str


def get_dependency_info() -> dict:
    """
    Gathers version information for the application's key dependencies.
    This function only includes information that can be dynamically queried
    at runtime.
    """
    info = {}

    # System
    info[_("System")] = [
        ("Python", sys.version.split(" ")[0]),
        (
            "Platform",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        ),
    ]

    # UI Toolkit
    try:
        import gi

        pygobject_ver = gi.__version__
    except (ImportError, AttributeError):
        pygobject_ver = _not_found_str

    ui_deps = [
        (
            "GTK",
            f"{Gtk.get_major_version()}."
            f"{Gtk.get_minor_version()}."
            f"{Gtk.get_micro_version()}",
        ),
        (
            "LibAdwaita",
            f"{Adw.get_major_version()}."
            f"{Adw.get_minor_version()}."
            f"{Adw.get_micro_version()}",
        ),
    ]
    if pygobject_ver != _not_found_str:
        ui_deps.append(("PyGObject", pygobject_ver))
    info[_("UI Toolkit")] = ui_deps

    # Graphics & Imaging
    graphics_deps = []
    try:
        import cairo

        graphics_deps.append(("PyCairo", cairo.version))
        graphics_deps.append(("libcairo", cairo.cairo_version_string()))
    except ImportError:
        pass

    try:
        import pyvips

        pyvips_ver = _get_version("pyvips")
        graphics_deps.append(("pyvips", pyvips_ver))
    except Exception as e:
        msg = f"failed to find pyvips version: {e}"
        logger.warning(msg)
        graphics_deps.append(("pyvips", msg))
        pyvips = None

    try:
        libvips_ver = pyvips.version(0) if pyvips else None
        graphics_deps.append(("libvips", libvips_ver))
    except Exception as e:
        msg = f"failed to find libvips version: {e}"
        logger.warning(msg)
        graphics_deps.append(("libvips", msg))

    for pkg_name, display_name in [
        ("opencv-python", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pypotrace", "pypotrace"),
    ]:
        ver = _get_version(pkg_name)
        graphics_deps.append((display_name, ver))

    if graphics_deps:
        info[_("Graphics & Imaging")] = graphics_deps

    comm_deps = []
    for pkg in [
        "ezdxf",
        "pypdf",
        "PyYAML",
        "pyserial-asyncio",
        "aiohttp",
        "websockets",
    ]:
        ver = _get_version(pkg)
        comm_deps.append((pkg, ver))

    if comm_deps:
        info[_("File Formats & Communication")] = comm_deps

    return {k: v for k, v in info.items() if v}


class AboutDialog(Adw.Window):
    """
    A custom 'About' dialog that uses a ViewStack to navigate between
    the main page and a detailed system information page.
    """

    def __init__(self, **kwargs):
        super().__init__(modal=True, **kwargs)
        self.set_default_size(500, 700)
        self.set_hide_on_close(True)

        self._build_ui()

    def _on_copy_info_clicked(self, button: Gtk.Button):
        lines = [f"## Rayforge {__version__ or _not_found_str}", ""]
        dep_info = get_dependency_info()
        for category, deps in dep_info.items():
            lines.append(f"### {category}")
            for name, ver in deps:
                lines.append(f"{name}: {ver}")
            lines.append("")
        full_text = "\n".join(lines).strip()
        clipboard = self.get_display().get_clipboard()
        clipboard.set(full_text)
        button.set_icon_name("object-select-symbolic")
        GLib.timeout_add(
            2000,
            lambda: button.set_icon_name("edit-copy-symbolic")
            and GLib.SOURCE_REMOVE,
        )
        # Also give feedback on the headerbar copy button if it exists
        if hasattr(self, "header_copy_button"):
            self.header_copy_button.set_icon_name("object-select-symbolic")
            GLib.timeout_add(
                2000,
                lambda: self.header_copy_button.set_icon_name(
                    "edit-copy-symbolic"
                )
                and GLib.SOURCE_REMOVE,
            )

    def _build_main_page(self):
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.set_halign(Gtk.Align.FILL)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)

        hero_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        hero_box.set_vexpand(True)
        hero_box.set_valign(Gtk.Align.CENTER)
        hero_box.set_halign(Gtk.Align.CENTER)
        content_box.append(hero_box)

        icon = Gtk.Image.new_from_icon_name("com.barebaric.rayforge")
        icon.set_pixel_size(128)
        hero_box.append(icon)

        title = Gtk.Label()
        title.set_markup("<span size='xx-large' weight='bold'>Rayforge</span>")
        title.set_margin_top(6)
        hero_box.append(title)

        version_label = Gtk.Label()
        version_label.set_markup(
            f"<small>Version {__version__ or _not_found_str}</small>"
        )
        hero_box.append(version_label)

        copyright_label = Gtk.Label(label="© 2025 Samuel Abels")
        hero_box.append(copyright_label)

        links_box = Gtk.Box(halign=Gtk.Align.CENTER, margin_top=12)
        links_box.add_css_class("linked")
        hero_box.append(links_box)

        website_button = Gtk.Button.new_with_label(_("Website"))
        website_button.connect(
            "clicked",
            lambda _: webbrowser.open("https://github.com/barebaric/rayforge"),
        )
        links_box.append(website_button)

        issues_button = Gtk.Button.new_with_label(_("Report an Issue"))
        issues_button.connect(
            "clicked",
            lambda _: webbrowser.open(
                "https://github.com/barebaric/rayforge/issues"
            ),
        )
        links_box.append(issues_button)

        prefgroup = Adw.PreferencesGroup()
        content_box.append(prefgroup)
        dev_row = Adw.ActionRow(
            title=_("Lead Developer"), subtitle="Samuel Abels"
        )
        prefgroup.add(dev_row)

        license_row = Adw.ActionRow(title=_("License"), subtitle="MIT X11")
        license_row.set_activatable(True)
        license_row.add_suffix(get_icon("open-in-new"))
        license_row.connect(
            "activated",
            lambda _: webbrowser.open("https://opensource.org/license/mit"),
        )
        prefgroup.add(license_row)

        sys_info_row = Adw.ActionRow(
            title=_("System Information"),
            subtitle=_("Versions of libraries and components"),
        )
        sys_info_row.set_activatable(True)

        inline_copy_button = Gtk.Button.new_from_icon_name(
            "edit-copy-symbolic"
        )
        inline_copy_button.set_valign(Gtk.Align.CENTER)
        inline_copy_button.add_css_class("flat")
        inline_copy_button.set_tooltip_text(_("Copy System Information"))
        inline_copy_button.connect("clicked", self._on_copy_info_clicked)
        sys_info_row.add_suffix(inline_copy_button)

        sys_info_row.add_suffix(
            Gtk.Image.new_from_icon_name("go-next-symbolic")
        )
        sys_info_row.connect(
            "activated",
            lambda w: self.view_stack.set_visible_child_name("sysinfo"),
        )
        prefgroup.add(sys_info_row)

        return content_box

    def _build_sysinfo_page(self):
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)
        scrolled_window.set_child(content_box)

        dep_info = get_dependency_info()
        for category, deps in dep_info.items():
            escaped_category = GLib.markup_escape_text(category)
            group = Adw.PreferencesGroup()
            group.set_title(escaped_category)
            content_box.append(group)

            for name, ver in deps:
                row = Adw.ActionRow(title=name, subtitle=str(ver))
                group.add(row)

        return scrolled_window

    def _on_view_changed(self, stack, param):
        visible_page = stack.get_visible_child_name()
        is_main = visible_page == "main"

        self.back_button.set_visible(not is_main)
        self.header_copy_button.set_visible(not is_main)
        self.header_bar.set_title_widget(
            self.main_title if is_main else self.sysinfo_title
        )

    def _build_ui(self):
        self.header_bar = Adw.HeaderBar()
        self.main_title = Adw.WindowTitle(title=_("About Rayforge"))
        self.sysinfo_title = Adw.WindowTitle(title=_("System Information"))

        self.back_button = Gtk.Button.new_from_icon_name(
            "go-previous-symbolic"
        )
        self.back_button.connect(
            "clicked", lambda w: self.view_stack.set_visible_child_name("main")
        )
        self.header_bar.pack_start(self.back_button)

        self.header_copy_button = Gtk.Button.new_from_icon_name(
            "edit-copy-symbolic"
        )
        self.header_copy_button.set_tooltip_text(_("Copy System Information"))
        self.header_copy_button.connect("clicked", self._on_copy_info_clicked)
        self.header_bar.pack_end(self.header_copy_button)

        self.view_stack = Adw.ViewStack()
        self.view_stack.add_named(self._build_main_page(), "main")
        self.view_stack.add_named(self._build_sysinfo_page(), "sysinfo")
        self.view_stack.connect(
            "notify::visible-child-name", self._on_view_changed
        )

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(self.header_bar)
        main_box.append(self.view_stack)
        self.set_content(main_box)
        self._on_view_changed(self.view_stack, None)  # Set initial state
