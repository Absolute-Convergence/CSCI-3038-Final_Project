"""Focused checks for the distributable Hyperloop package boundary."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


_REPOSITORY = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (_REPOSITORY / "pyproject.toml").open("rb") as stream:
            cls.document = tomllib.load(stream)

    def test_distribution_identity_and_installed_command(self) -> None:
        project = self.document["project"]

        self.assertEqual(project["name"], "hyperloop-optimizer")
        self.assertEqual(project["version"], "0.1.1")
        self.assertEqual(
            project["scripts"]["hyperloop-optimizer"],
            "black_box_optimizer.cli:installed_main",
        )
        self.assertEqual(
            project["scripts"]["hyperloop-synthetic-worker"],
            "hyperloop_workers.synthetic_worker:main",
        )

    def test_core_dependencies_exclude_worker_and_gui_frameworks(self) -> None:
        dependencies = self.document["project"]["dependencies"]
        normalized = "\n".join(dependencies).lower()

        self.assertIn("numpy", normalized)
        self.assertIn("matplotlib", normalized)
        self.assertNotIn("torch", normalized)
        self.assertNotIn("pillow", normalized)

    def test_distribution_includes_core_and_synthetic_worker_only(self) -> None:
        discovery = self.document["tool"]["setuptools"]["packages"]["find"]

        self.assertEqual(
            discovery["include"],
            ["black_box_optimizer*", "hyperloop_workers*"],
        )
        self.assertIn("examples*", discovery["exclude"])


class RequirementsSeparationTests(unittest.TestCase):
    def test_requirements_files_keep_component_dependencies_separate(
        self,
    ) -> None:
        core = (_REPOSITORY / "requirements.txt").read_text(encoding="utf-8")
        iris = (_REPOSITORY / "requirements-iris.txt").read_text(
            encoding="utf-8"
        )
        gui = (_REPOSITORY / "requirements-gui.txt").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("torch", core.lower())
        self.assertNotIn("pillow", core.lower())
        self.assertIn("torch", iris.lower())
        self.assertNotIn("pillow", iris.lower())
        self.assertIn("pillow", gui.lower())
        self.assertNotIn("torch", gui.lower())


if __name__ == "__main__":
    unittest.main()
