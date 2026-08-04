"""Focused tests for the optional GUI's platform folder launcher."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import Hyperloop
except ModuleNotFoundError as error:
    if error.name in {"PIL", "tkinter", "_tkinter"}:
        raise unittest.SkipTest(
            "optional GUI dependencies are not installed"
        ) from error
    raise


class OpenFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path("results folder")
        self.resolved_folder = str(self.folder.resolve())

    def test_windows_uses_startfile(self) -> None:
        with (
            patch.object(Hyperloop.sys, "platform", "win32"),
            patch.object(
                Hyperloop.os,
                "startfile",
                create=True,
            ) as startfile,
        ):
            Hyperloop._open_folder(self.folder)

        startfile.assert_called_once_with(self.resolved_folder)

    def test_macos_uses_nonblocking_open_command(self) -> None:
        with (
            patch.object(Hyperloop.sys, "platform", "darwin"),
            patch.object(Hyperloop.subprocess, "Popen") as launch,
        ):
            Hyperloop._open_folder(self.folder)

        launch.assert_called_once_with(
            ["open", self.resolved_folder],
            shell=False,
        )

    def test_linux_uses_nonblocking_xdg_open_command(self) -> None:
        with (
            patch.object(Hyperloop.sys, "platform", "linux"),
            patch.object(Hyperloop.subprocess, "Popen") as launch,
        ):
            Hyperloop._open_folder(self.folder)

        launch.assert_called_once_with(
            ["xdg-open", self.resolved_folder],
            shell=False,
        )

    def test_launcher_error_preserves_results_location(self) -> None:
        with (
            patch.object(Hyperloop.sys, "platform", "darwin"),
            patch.object(
                Hyperloop.subprocess,
                "Popen",
                side_effect=OSError("launcher unavailable"),
            ),
            patch.object(Hyperloop.messagebox, "showerror") as showerror,
        ):
            Hyperloop._open_folder(self.folder)

        showerror.assert_called_once()
        title, message = showerror.call_args.args
        self.assertEqual(title, "Could Not Open Folder")
        self.assertIn("launcher unavailable", message)
        self.assertIn(self.resolved_folder, message)


if __name__ == "__main__":
    unittest.main()
