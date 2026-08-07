# =============================================================================
# hyperloop_theme
# =============================================================================
#
# The pink visual theme for the HyperLoop GUI: color tokens and the ttk
# style configuration that applies them. Split out of Hyperloop.py to keep
# that file under the repository's source-hygiene line limit and to keep
# styling concerns separate from GUI construction logic.
#
# =============================================================================

import tkinter as tk
from tkinter import ttk

_PAGE = "#fdf1f5"
_SURFACE = "#fffafc"
_PRIMARY_INK = "#6b1236"
_SECONDARY_INK = "#9c4267"
_MUTED_INK = "#b97b93"
_GRIDLINE = "#f3c6d9"
_BUTTON_WASH = "#fbe4ec"
_BUTTON_WASH_ACTIVE = "#f6c9da"
_ACCENT = "#c2255c"
_ACCENT_ACTIVE = "#a51e4d"
_ACCENT_TEXT = "#ffffff"
_STATUS_GOOD = "#0ca30c"  # not pinked -- success/failure must stay distinct
_STATUS_CRITICAL = "#d03b3b"
_HEADING_FONT = "Helvetica"


def _apply_theme(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass  # fall back to whatever theme this platform ships by default

    root.configure(background=_PAGE)

    style.configure("TFrame", background=_SURFACE)
    style.configure(
        "TLabel", background=_SURFACE, foreground=_SECONDARY_INK,
        font=(_HEADING_FONT, 11),
    )
    style.configure(
        "TLabelframe", background=_SURFACE, bordercolor=_GRIDLINE,
        relief="solid", borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label", background=_SURFACE, foreground=_PRIMARY_INK,
        font=(_HEADING_FONT, 10, "bold"),
    )
    style.configure(
        "TButton", background=_BUTTON_WASH, foreground=_PRIMARY_INK,
        bordercolor=_GRIDLINE, padding=3, font=(_HEADING_FONT, 10),
    )
    style.map(
        "TButton",
        background=[("active", _BUTTON_WASH_ACTIVE), ("disabled", _SURFACE)],
        foreground=[("disabled", _MUTED_INK)],
    )
    style.configure(
        "Accent.TButton", background=_ACCENT, foreground=_ACCENT_TEXT,
        font=(_HEADING_FONT, 10, "bold"), padding=5,
    )
    style.map(
        "Accent.TButton",
        background=[("active", _ACCENT_ACTIVE), ("disabled", _MUTED_INK)],
    )
    style.configure(
        "TEntry", fieldbackground=_SURFACE, bordercolor=_GRIDLINE,
        font=(_HEADING_FONT, 11),
    )
    style.configure(
        "TCombobox", fieldbackground=_SURFACE, background=_BUTTON_WASH,
        arrowcolor=_PRIMARY_INK, bordercolor=_GRIDLINE,
        font=(_HEADING_FONT, 11),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", _SURFACE)],
        background=[("readonly", _BUTTON_WASH)],
    )
    root.option_add("*TCombobox*Listbox.background", _SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", _PRIMARY_INK)
    root.option_add("*TCombobox*Listbox.selectBackground", _ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", _ACCENT_TEXT)
    style.configure(
        "Header.TLabel", background=_SURFACE, foreground=_PRIMARY_INK,
        font=(_HEADING_FONT, 13, "bold"),
    )
    style.configure(
        "Subheader.TLabel", background=_SURFACE, foreground=_MUTED_INK,
        font=(_HEADING_FONT, 10),
    )
