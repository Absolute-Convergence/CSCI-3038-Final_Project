"""Focused tests for the search algorithm registry."""

from __future__ import annotations

import unittest

from black_box_optimizer.models import AlgorithmSpec
from black_box_optimizer.search.random_search import RandomSearch
from black_box_optimizer.search.registry import (
    ALGORITHM_REGISTRY,
    UnknownAlgorithmError,
    create_algorithm,
)


class AlgorithmRegistryTests(unittest.TestCase):
    """Verify registry lookup and unknown-name rejection."""

    def test_random_search_is_registered(self) -> None:
        self.assertIn("random_search", ALGORITHM_REGISTRY)

    def test_create_algorithm_builds_random_search(self) -> None:
        spec = AlgorithmSpec(name="random_search", seed=1)
        algorithm = create_algorithm(spec)
        self.assertIsInstance(algorithm, RandomSearch)

    def test_unknown_algorithm_name_rejected(self) -> None:
        with self.assertRaises(UnknownAlgorithmError):
            create_algorithm(AlgorithmSpec(name="genetic_algorithm", seed=1))

    def test_invalid_seed_is_rejected_before_algorithm_construction(
        self,
    ) -> None:
        # AlgorithmSpec itself rejects a negative seed (see models.py),
        # so create_algorithm() never gets a chance to see one -- the
        # rejection happens at construction, the earliest possible point,
        # not just deep inside RandomSearch's/NSGA2's own defensive checks.
        with self.assertRaisesRegex(ValueError, "negative"):
            AlgorithmSpec(name="random_search", seed=-1)


if __name__ == "__main__":
    unittest.main()
