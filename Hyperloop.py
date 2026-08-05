# =============================================================================
# HyperLoop
# =============================================================================
#
# Graphical Configuration Tool for the Black Box Optimizer
#
# HyperLoop provides a Tkinter-based interface for creating, validating, and
# launching optimization runs without manually editing JSON configuration files.
# The application builds a valid optimizer configuration, saves it to disk,
# and launches the optimizer using its public Python API.
#
# Features
# --------
# - Configure worker commands and execution settings.
# - Define optimization parameters (float, integer, categorical).
# - Configure single or multi-objective optimization problems.
# - Select optimization algorithm and stopping criteria.
# - Preview generated JSON configuration.
# - Save configuration files for future use.
# - Launch optimization runs directly from the GUI.
#
# Architecture
# ------------
# HyperLoop is intentionally a thin front-end over the optimizer library.
# All optimization logic remains inside the black_box_optimizer package.
# This application is responsible only for:
#
#   1. Collecting user input
#   2. Building a valid configuration
#   3. Saving the configuration
#   4. Launching the optimizer
#   5. Displaying run status and results
#
# Future Enhancements
# -------------------
# - Integrated results viewer
# - Improved JSON preview with save options
# - Progress reporting during optimization
# - Run history management
# - Standalone executable distribution
#
# =============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import subprocess
import sys
import threading
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from black_box_optimizer.application import initialize_application


def _open_folder(path):
    """Open a folder in the platform file manager without blocking Tkinter."""
    folder = str(path)

    try:
        folder = str(Path(path).resolve())
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder], shell=False)
        else:
            subprocess.Popen(["xdg-open", folder], shell=False)
    except OSError as error:
        messagebox.showerror(
            "Could Not Open Folder",
            f"{error}\n\nResults are still saved at:\n{folder}",
        )


def get_asset_path(relative_path):
    """Get absolute path to resource relative to this script file"""
    base_path = os.path.dirname(os.path.realpath(__file__))

    return os.path.normpath(os.path.join(base_path, relative_path))


def set_icon(relative_img_path):
    try:
        icon_img_path = get_asset_path(relative_img_path)
        icon_img = Image.open(icon_img_path)
        icon = ImageTk.PhotoImage(icon_img)
        return icon
    except Exception as e:
        print(f"Warning: Icon file could not be loaded!!")
        return None


class ConfigApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HyperLoop")
        self.root.geometry("620x775")
        self.app_icon = set_icon(os.path.join("assets", "img", "HyperLoop.png"))
        if self.app_icon:
            self.root.iconphoto(False, self.app_icon)
        self.stop_requested = threading.Event()
        self.last_result = None

        # Main Scrollable Canvas
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, padding=10)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.parameters = []
        self.objectives = []

        self.available_workers = self.discover_workers()
        self.build_worker_section()
        self.build_parameters_section()
        self.build_objectives_section()
        self.build_algorithm_section()
        self.build_footer()

#===========================================================================
# Build Functions for each frame of the GUI
# called at the initization of ConfigApp
#===========================================================================

    def build_worker_section(self):
        frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="Worker Configuration",
            padding=10,
        )
        frame.pack(fill="x", pady=5)

        ttk.Label(frame, text="Worker Script:").grid(row=0, column=0, sticky="w")

        default_worker = (
            Path(__file__).parent / "examples" / "paper_airplane" / "iris_worker.py"
        )

        self.worker_path = tk.StringVar(value=str(default_worker))

        self.worker_choice = tk.StringVar()

        self.worker_combo = ttk.Combobox(
            frame,
            textvariable=self.worker_choice,
            values=list(self.available_workers.keys()) + ["Browse..."],
            state="readonly",
            width=35,
        )

        self.worker_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        if self.available_workers:
            first = sorted(self.available_workers.keys())[0]
            self.worker_choice.set(first)
            self.worker_path.set(str(self.available_workers[first]))

        self.worker_combo.bind("<<ComboboxSelected>>", self.on_worker_selected)

        ttk.Button(
            frame,
            text="Browse...",
            command=self.browse_worker,
        ).grid(row=0, column=2, padx=5)

        self.worker_status = ttk.Label(frame)
        self.worker_status.grid(row=1, column=1, sticky="w", pady=(0, 8))

        self.validate_worker()

        ttk.Label(frame, text="Metrics Argument:").grid(
            row=2,
            column=0,
            sticky="w",
        )

        self.metrics_entry = ttk.Entry(frame, width=40)
        self.metrics_entry.insert(0, "--metrics-out")
        self.metrics_entry.grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )

        ttk.Label(frame, text="Timeout (seconds):").grid(
            row=3,
            column=0,
            sticky="w",
        )

        self.timeout_entry = ttk.Entry(frame, width=40)
        self.timeout_entry.insert(0, "120.0")
        self.timeout_entry.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )

        frame.columnconfigure(1, weight=1)

    def build_parameters_section(self):
        self.param_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Optimization Parameters", padding=10
        )
        self.param_frame.pack(fill="x", pady=5)

        add_btn = ttk.Button(
            self.param_frame, text="+ Add Parameter", command=self.add_parameter_row
        )
        add_btn.pack(anchor="w", pady=5)

        # Headers
        self.header_frame = ttk.Frame(self.param_frame)
        self.header_frame.pack(fill="x")
        ttk.Label(self.header_frame, text="Name", width=16).grid(row=0, column=0)
        ttk.Label(self.header_frame, text="Type", width=14).grid(row=0, column=1)
        ttk.Label(self.header_frame, text="Min", width=19).grid(row=0, column=2)
        ttk.Label(self.header_frame, text="Max", width=10).grid(row=0, column=3)

        self.add_parameter_row("learning_rate", "float", "0.0001", "0.1")
        self.add_parameter_row("hidden_size", "integer", "4", "128")
        self.add_parameter_row("epochs", "integer", 5, 100)
        self.add_parameter_row("batch_size", "categorical", "8, 16, 32")


    def build_objectives_section(self):
        self.obj_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Objectives", padding=10
        )
        self.obj_frame.pack(fill="x", pady=5)

        add_btn = ttk.Button(
            self.obj_frame, text="+ Add Objective", command=self.add_objective_row
        )
        add_btn.pack(anchor="w", pady=5)

        self.add_objective_row("validation_accuracy", "maximize")
        self.add_objective_row("training_time_seconds", "minimize")

    def build_algorithm_section(self):
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Algorithm & Stop Policy", padding=10
        )
        frame.pack(fill="x", pady=5)

        ttk.Label(frame, text="Algorithm Name:").grid(row=0, column=0, sticky="w")
        algorithms = ["random_search", "nsga2"]
        self.algo_entry = ttk.Combobox(frame, values=algorithms, width=20)
        self.algo_entry.current(0)
        self.algo_entry.grid(row=0, column=1, pady=2, sticky="w")

        ttk.Label(frame, text="Seed:").grid(row=1, column=0, sticky="w")
        self.seed_entry = ttk.Entry(frame, width=20)
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=1, column=1, pady=2, sticky="w")

        ttk.Label(frame, text="Max Trials:").grid(row=2, column=0, sticky="w")
        self.max_trials_entry = ttk.Entry(frame, width=20)
        self.max_trials_entry.insert(0, "20")
        self.max_trials_entry.grid(row=2, column=1, pady=2, sticky="w")

    def build_footer(self):
        btn_frame = ttk.Frame(self.scrollable_frame, padding=10)
        btn_frame.pack(fill="x", pady=10)

        # File naming entry layout
        filename_frame = ttk.Frame(btn_frame)
        filename_frame.pack(fill="x", pady=5)
        ttk.Label(filename_frame, text="Save Filename:").pack(side="left")
        self.filename_entry = ttk.Entry(filename_frame, width=30)
        self.filename_entry.insert(0, "config.json")
        self.filename_entry.pack(side="left", padx=5)

        # Actions arrangement
        actions_frame = ttk.Frame(btn_frame)
        actions_frame.pack(fill="x", pady=5)

        self.status_label = ttk.Label(btn_frame, text="Ready")
        self.status_label.pack(anchor="w")

        self.trials_label = ttk.Label(btn_frame, text="")
        self.trials_label.pack(anchor="w")

        preview_btn = ttk.Button(
            actions_frame, text="Preview JSON", command=self.generate_json
        )
        preview_btn.pack(side="left", expand=True, fill="x", padx=2)

        self.stop_btn = ttk.Button(
            actions_frame, text="Stop", command=self.stop_optimizer, state="disabled"
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=2)

        self.go_btn = ttk.Button(
            actions_frame, text="Do the thing", command=self.run_loop
        )
        self.go_btn.pack(side="left", expand=True, fill="x", padx=2)

#===========================================================================
# Functions to add and remove user configurations to the parameters and 
# objectives before creating the JSON 
#===========================================================================

    def add_parameter_row(self, name="", kind="float", p_min="", p_max=""):
        row_frame = ttk.Frame(self.param_frame)
        row_frame.pack(fill="x", pady=2)

        name_ent = ttk.Entry(row_frame, width=15)
        name_ent.insert(0, name)
        name_ent.grid(row=0, column=0, padx=2)

        kind_combo = ttk.Combobox(
            row_frame,
            values=["float", "integer", "categorical"],
            width=10,
            state="readonly",
        )
        kind_combo.set(kind)
        kind_combo.grid(row=0, column=1, padx=2)

        min_ent = ttk.Entry(row_frame, width=18)
        min_ent.insert(0, p_min)
        min_ent.grid(row=0, column=2, padx=2)

        max_ent = ttk.Entry(row_frame, width=10)
        max_ent.insert(0, p_max)
        max_ent.grid(row=0, column=3, padx=2)

        def on_type_change(event):
            if kind_combo.get() == "categorical":
                min_ent.delete(0, tk.END)
                min_ent.insert(0, "8, 16, 32")
                max_ent.configure(state="disabled")
            else:
                max_ent.configure(state="normal")

        kind_combo.bind("<<ComboboxSelected>>", on_type_change)
        if kind == "categorical":
            max_ent.configure(state="disabled")

        del_btn = ttk.Button(
            row_frame,
            text="X",
            width=3,
            command=lambda: self.remove_row(row_frame, self.parameters, row_data),
        )
        del_btn.grid(row=0, column=4, padx=2)

        row_data = {
            "name": name_ent,
            "kind": kind_combo,
            "min": min_ent,
            "max": max_ent,
            "frame": row_frame,
        }
        self.parameters.append(row_data)

    def add_objective_row(self, name="", direction="maximize"):
        row_frame = ttk.Frame(self.obj_frame)
        row_frame.pack(fill="x", pady=2)

        name_ent = ttk.Entry(row_frame, width=25)
        name_ent.insert(0, name)
        name_ent.pack(side="left", padx=2)

        dir_combo = ttk.Combobox(
            row_frame, values=["maximize", "minimize"], width=12, state="readonly"
        )
        dir_combo.set(direction)
        dir_combo.pack(side="left", padx=2)

        del_btn = ttk.Button(
            row_frame,
            text="X",
            width=3,
            command=lambda: self.remove_row(row_frame, self.objectives, row_data),
        )
        del_btn.pack(side="left", padx=2)

        row_data = {"name": name_ent, "direction": dir_combo, "frame": row_frame}
        self.objectives.append(row_data)

    def remove_row(self, frame, item_list, row_data):
        frame.destroy()
        item_list.remove(row_data)

#======================================================================
# Functions to handle the user's selected worker 
#======================================================================

    def browse_worker(self):
        filename = filedialog.askopenfilename(
            title="Select Worker Script",
            filetypes=[("Python Files", "*.py")],
            initialdir=Path(__file__).parent / "examples",
        )

        if filename:
            path = Path(filename)

            self.worker_path.set(str(path))

            # Add it if it wasn't discovered
            if path.name not in self.available_workers:
                self.available_workers[path.name] = path
                self.worker_combo["values"] = list(self.available_workers.keys()) + [
                    "Browse..."
                ]

            self.worker_choice.set(path.name)
            self.validate_worker()

    def validate_worker(self):
        path = Path(self.worker_path.get())

        if path.is_file():
            self.worker_status.config(text="✓ Worker Found", foreground="green")
            return True

        self.worker_status.config(text="✗ Worker Not Found", foreground="red")
        return False

    def on_worker_selected(self, event=None):
        selection = self.worker_choice.get()

        if selection == "Browse...":
            self.browse_worker()
            return

        self.worker_path.set(str(self.available_workers[selection]))
        self.validate_worker()

    def discover_workers(self):
        """Find all worker scripts under the examples directory."""

        examples_dir = Path(__file__).parent / "examples"

        workers = {}

        if examples_dir.exists():
            for worker in examples_dir.rglob("*_worker.py"):
                workers[worker.name] = worker

        return workers
    
    def update_trial_counter(self):
        if not hasattr(self, "_history_file"):
            matches = list(self.last_output_dir.rglob("history.csv"))
            if matches:
                self._history_file = matches[0]

        if hasattr(self, "_history_file"):
            try:
                df = pd.read_csv(self._history_file)

                completed = len(df)
                max_trials = int(self.max_trials_entry.get())

                self.trials_label.config(
                    text=f"Trials completed: {completed} / {max_trials}"
                )
            except Exception:
                pass

        if self.worker_thread.is_alive():
            self.root.after(1000, self.update_trial_counter)

#=================================================================
# Builds the JSON file from the user's inputs 
# generate_json is used to produce the preview window 
# save_json_to_folder will save a json file for future use.
#
# Save_json_to_floder is not currently used in the application
# it will be used in future development to re-run optimizer 
#=================================================================


    def build_config_dict(self):
        """Build the optimizer configuration dictionary."""

        cmd = [sys.executable, self.worker_path.get()]

        timeout = float(self.timeout_entry.get())

        params_out = []

        for p in self.parameters:
            p_type = p["kind"].get()
            p_name = p["name"].get().strip()

            if not p_name:
                continue

            parameter = {
                "name": p_name,
                "kind": p_type,
            }

            if p_type == "categorical":
                choices = []

                for value in p["min"].get().split(","):
                    value = value.strip()

                    if value == "":
                        continue

                    try:
                        if "." in value:
                            choices.append(float(value))
                        else:
                            choices.append(int(value))
                    except ValueError:
                        choices.append(value)

                parameter["choices"] = choices

            elif p_type == "float":
                parameter["minimum"] = float(p["min"].get())
                parameter["maximum"] = float(p["max"].get())

            else:  # integer
                parameter["minimum"] = int(p["min"].get())
                parameter["maximum"] = int(p["max"].get())

            params_out.append(parameter)

        objs_out = []

        for o in self.objectives:
            name = o["name"].get().strip()

            if not name:
                continue

            objs_out.append(
                {
                    "metric_name": name,
                    "direction": o["direction"].get(),
                }
            )

        return {
            "worker": {
                "command": cmd,
                "metrics_argument": self.metrics_entry.get().strip(),
                "timeout_seconds": timeout,
            },
            "optimization": {
                "parameters": params_out,
                "objectives": objs_out,
            },
            "algorithm": {
                "name": self.algo_entry.get().strip(),
                "seed": int(self.seed_entry.get()),
            },
            "stop_policy": {
                "max_trials": int(self.max_trials_entry.get()),
            },
        }

    def generate_json(self):
        try:
            config = self.build_config_dict()
            preview_win = tk.Toplevel(self.root)
            preview_win.title("JSON Preview")
            preview_win.iconphoto(False, self.app_icon)
            text_area = tk.Text(preview_win, width=60, height=25)
            text_area.insert(tk.END, json.dumps(config, indent=2))
            text_area.pack(padx=10, pady=10)
        except ValueError as e:
            messagebox.showerror(
                "Parsing Error", f"Please check your inputs.\nError: {e}"
            )

    def save_json_to_folder(self):
        try:
            config = self.build_config_dict()

            try:
                # If running frozen via PyInstaller
                base_dir = sys._MEIPASS
                current_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
            except AttributeError:
                current_directory = os.path.dirname(os.path.abspath(__file__))

            target_folder = os.path.join(current_directory, "saved_configs")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            filename = self.filename_entry.get().strip()
            if not filename.endswith(".json"):
                filename += ".json"

            full_save_path = os.path.join(target_folder, filename)

            with open(full_save_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo(
                "Success", f"File saved successfully to:\n{full_save_path}"
            )

        except ValueError as e:
            messagebox.showerror(
                "Parsing Error", f"Please check your inputs.\nError: {e}"
            )
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to save file.\nError: {e}")

#=============================================================================
# run_loop is called when "Do the Thing" button is pressed
# JSON file will be created and passed to the optimizer 
#=============================================================================

    def run_loop(self):
        """Save the configuration and launch the optimizer."""

        if not self.validate_worker():
            messagebox.showerror(
                "Invalid Worker", "Please select a valid worker script."
            )
            return

        if hasattr(self, "_history_file"):
            del self._history_file

        self.stop_requested.clear()

        try:
            config = self.build_config_dict()

            if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
                messagebox.showwarning(
                    "Already Running", "An optimization is already running."
                )
                return

            # Base directory (works both normally and with PyInstaller)
            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent

            # Ensure folders exist
            config_dir = base_dir
            output_root = base_dir / "optimizer_runs"

            config_dir.mkdir(exist_ok=True)
            output_root.mkdir(exist_ok=True)

            # Build filenames
            filename = self.filename_entry.get().strip()

            if not filename.endswith(".json"):
                filename += ".json"

            config_path = config_dir / filename

            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_root / f"run_{run_name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            # Disable btn once config is built
            self.go_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # Starts the progress bar
            self.status_label.config(text="Running optimization...")

            # Launch optimizer
            self.worker_thread = threading.Thread(
                target=self.run_optimizer,
                args=(config_path, output_dir),
                daemon=True,
            )

            self.worker_thread.start()

            self.last_output_dir = output_dir
            self.update_trial_counter()

        except Exception as e:
            messagebox.showerror("Error", str(e))

#======================================================
# Starts and Stops the optimizer 
#======================================================

    def run_optimizer(self, config_path, output_dir):
        try:
            app = initialize_application(
                configuration_path=config_path,
                output_directory=output_dir,
            )

            app.controller.cancel_event = self.stop_requested

            result = app.run()
            self.last_result = result

            self.root.after(0, lambda: self.optimizer_finished(output_dir))

            return result

        except KeyboardInterrupt:
            self.root.after(
                0,
                lambda: self.optimizer_cancelled(output_dir),
            )

        except Exception as e:
            self.root.after(0, lambda: self.optimizer_failed(e))

    # Triggers a KeyboardInterrupt inside the optimizer 
    def stop_optimizer(self):
        self.stop_requested.set()

#===================================================================
# optimizer_finished/cancelled/failed handle the different end
# states after the optimizer stops running
#===================================================================

    def optimizer_finished(self, output_dir):
        """Display a summary when the optimizer finishes."""
        self.status_label.config(text="Optimization complete.")
        self.trials_label.config(text="")

        self.go_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.last_output_dir = Path(output_dir)

        window = tk.Toplevel(self.root)
        window.title("Optimization Complete")
        window.iconphoto(False, self.app_icon)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="✓ Optimization Complete",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 15))

        ttk.Label(
            frame,
            text="The optimization finished successfully.",
        ).pack(anchor="w")

        ttk.Label(
            frame,
            text=f"\nResults Directory:\n{output_dir}",
            justify="left",
        ).pack(anchor="w", pady=(10, 20))

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x")

        ttk.Button(
            button_frame,
            text="Open Results Folder",
            command=lambda: _open_folder(output_dir),
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Close",
            command=window.destroy,
        ).pack(side="right", padx=5)

    def optimizer_cancelled(self, output_dir):
        self.go_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.status_label.config(text="Optimization stopped.")
        self.trials_label.config(text="")

        window = tk.Toplevel(self.root)
        window.title("Optimization Stopped")
        window.iconphoto(False, self.app_icon)

        frame = ttk.Frame(window, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="⏹ Optimization Stopped",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 15))

        ttk.Label(
            frame,
            text="The optimization was stopped by the user.",
        ).pack(anchor="w")

        ttk.Button(
            frame,
            text="Open Results Folder",
            command=lambda: _open_folder(output_dir),
        ).pack(side="left", padx=5)

        ttk.Button(
            frame,
            text="Close",
            command=window.destroy,
        ).pack(side="right", padx=5)

    def optimizer_failed(self, error):
        self.go_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.status_label.config(text="Optimization failed.")
        self.trials_label.config(text="")

        messagebox.showerror(
            "Optimizer Error",
            str(error),
        )

#================================================
# Creates the main Tkinter window
# Initializes the Hyperloop application
# Loops until the user closes the application 
# ===============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigApp(root)
    root.mainloop()
