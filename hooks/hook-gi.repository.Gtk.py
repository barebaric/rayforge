#-----------------------------------------------------------------------------
# Custom hook for Gtk 4.0 (overrides PyInstaller's default Gtk 3.0 hook)
#-----------------------------------------------------------------------------

import os
import os.path

from PyInstaller.compat import is_win
from PyInstaller.utils.hooks import get_hook_config
from PyInstaller.utils.hooks.gi import GiModuleInfo, collect_glib_etc_files, collect_glib_share_files, \
    collect_glib_translations


def hook(hook_api):
    # Use GTK 4.0 instead of the default 3.0
    module_info = GiModuleInfo('Gtk', '4.0', hook_api=hook_api)
    if not module_info.available:
        return

    binaries, datas, hiddenimports = module_info.collect_typelib_data()

    # Collect fontconfig data
    datas += collect_glib_share_files('fontconfig')

    # Icons, themes, translations
    icon_list = get_hook_config(hook_api, "gi", "icons")
    if icon_list is not None:
        for icon in icon_list:
            datas += collect_glib_share_files(os.path.join('icons', icon))
    else:
        datas += collect_glib_share_files('icons')

    # Themes
    theme_list = get_hook_config(hook_api, "gi", "themes")
    if theme_list is not None:
        for theme in theme_list:
            datas += collect_glib_share_files(os.path.join('themes', theme))
    else:
        datas += collect_glib_share_files('themes')

    # Translations - use gtk40 for GTK 4.0
    lang_list = get_hook_config(hook_api, "gi", "languages")
    datas += collect_glib_translations('gtk40', lang_list)

    # These only seem to be required on Windows
    if is_win:
        datas += collect_glib_etc_files('fonts')
        datas += collect_glib_etc_files('pango')
        datas += collect_glib_share_files('fonts')

    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
