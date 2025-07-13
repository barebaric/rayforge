from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks.gi import collect_glib_share_files


datas = []
datas += collect_data_files('gi')
datas += collect_glib_share_files('girepository-1.0')


def pre_safe_import_module(api):
    api.add_runtime_module('gi.repository.Gtk')
