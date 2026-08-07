# =============================================================================
# hyperloop_help
# =============================================================================
#
# The two instructions popups shown from the HyperLoop GUI's header: a
# general usage overview, and Mr. Smith's specific walkthrough for testing
# the real neural-network (Iris) example. Split out of Hyperloop.py to keep
# that file under the repository's source-hygiene line limit and to keep
# static help content separate from the main GUI construction logic.
#
# =============================================================================

import tkinter as tk
from tkinter import ttk

from hyperloop_theme import _PAGE, _SURFACE, _PRIMARY_INK, _HEADING_FONT

_INSTRUCTIONS_TEXT = (
    "GENERAL USAGE\n"
    "1. Worker Configuration -- pick a worker script (or Browse for "
    "your own), plus its metrics flag and timeout.\n\n"
    "2. Optimization Parameters -- declare each tunable input and its "
    "range or choices.\n\n"
    "3. Objectives -- name the metrics to maximize or minimize.\n\n"
    "4. Algorithm & Stop Policy -- choose random_search or nsga2, an "
    "optional seed, and a trial limit.\n\n"
    "5. Preview JSON to check the generated configuration, then "
    "\"Do the Thing\" to launch the run. Stop cancels a run in "
    "progress.\n\n"
    "The window opens pre-filled for the Paper Airplane example, which "
    "runs instantly with no extra setup -- just click \"Do the Thing\"."
)

_MR_SMITH_INSTRUCTIONS_TEXT = (
    "Torch is not a dependency of the pipeline itself, so we did not "
    "include it in requirements.txt. HyperLoop treats every worker as "
    "an opaque subprocess -- it never imports or inspects worker code -- "
    "so the core package has no reason to depend on whatever machine "
    "learning library a given worker happens to use.\n\n"
    "iris_worker.py is the one example that trains a real neural "
    "network (a small PyTorch model on the Iris dataset), so it's the "
    "one example that needs an extra install.\n\n"
    "TO TEST THE REAL NN (IRIS) EXAMPLE\n\n"
    "1. One-time setup, in a terminal in the project folder: "
    "python3 -m pip install -r requirements-iris.txt\n"
    "   This installs PyTorch into the same environment/interpreter "
    "you're running HyperLoop from. It's a large package, so this can "
    "take a few minutes and needs an internet connection.\n\n"
    "2. In HyperLoop, set Worker Script to \"iris_worker.py\" from the "
    "dropdown. Leave Metrics Argument as --metrics-out and Timeout at "
    "120.0 -- unlike Paper Airplane, each trial here actually trains a "
    "small network, so it takes a real (if short) amount of time.\n\n"
    "3. Delete the pre-filled parameter rows and add these four:\n"
    "   - learning_rate | float | min 0.0001 | max 0.1\n"
    "   - hidden_size | integer | min 4 | max 128\n"
    "   - epochs | integer | min 5 | max 100\n"
    "   - batch_size | categorical | min field: 8, 16, 32\n\n"
    "4. Delete the pre-filled objective rows and add these three:\n"
    "   - validation_accuracy | maximize\n"
    "   - validation_loss | minimize\n"
    "   - training_time_seconds | minimize\n\n"
    "5. Algorithm & Stop Policy: random_search, Max Trials 20 is a "
    "reasonable default (leave Seed blank for a random one).\n\n"
    "6. Click \"Preview JSON\" to double check the values, then "
    "\"Do the Thing\" to run.\n\n"
    "If PyTorch isn't installed, or step 1 is skipped, this is safe to "
    "try anyway -- each trial just fails individually (recorded as a "
    "failed trial) rather than crashing the whole run."
)


def _show_text_popup(root, app_icon, title, heading, body_text):
    window = tk.Toplevel(root, background=_PAGE)
    window.title(title)
    window.iconphoto(False, app_icon)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=heading,
        font=(_HEADING_FONT, 14, "bold"),
    ).pack(anchor="w", pady=(0, 15))

    text_frame = ttk.Frame(frame)
    text_frame.pack(fill="both", expand=True)

    text_scroll = ttk.Scrollbar(text_frame, orient="vertical")
    text_area = tk.Text(
        text_frame,
        width=58,
        height=24,
        wrap="word",
        background=_SURFACE,
        foreground=_PRIMARY_INK,
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=8,
        pady=8,
        yscrollcommand=text_scroll.set,
    )
    text_scroll.config(command=text_area.yview)

    text_area.insert(tk.END, body_text)
    text_area.config(state="disabled")

    text_area.pack(side="left", fill="both", expand=True)
    text_scroll.pack(side="right", fill="y")

    ttk.Button(
        frame,
        text="Close",
        command=window.destroy,
    ).pack(anchor="e", pady=(15, 0))


def show_instructions(root, app_icon):
    _show_text_popup(
        root, app_icon, "Instructions", "How to Use HyperLoop", _INSTRUCTIONS_TEXT
    )


def show_mr_smith_instructions(root, app_icon):
    _show_text_popup(
        root,
        app_icon,
        "Special Mr. Smith Instructions",
        "Special Mr. Smith Instructions",
        _MR_SMITH_INSTRUCTIONS_TEXT,
    )
