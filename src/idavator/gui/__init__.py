"""IDA GUI: lifting viewer and LLVM IR drop actions."""

from idavator import __version__


def gui_caption(title: str) -> str:
    """Window / dialog title with package version (from ``idavator.__version__``)."""
    return f"{title} — v{__version__}"


def PLUGIN_ENTRY():
    from idavator.gui.plugin import IDAvatorPlugin

    return IDAvatorPlugin()
