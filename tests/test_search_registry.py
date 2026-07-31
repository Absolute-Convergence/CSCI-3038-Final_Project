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

    def test_invalid_seed_fails_during_creation(self) -> None:
        # AlgorithmSpec itself only checks that seed is an int, not that
        # it's non-negative, so this proves the rejection actually
        # propagates through create_algorithm() rather than only being
        # tested against RandomSearch directly.
        with self.assertRaisesRegex(ValueError, "negative"):
            create_algorithm(AlgorithmSpec(name="random_search", seed=-1))


if __name__ == "__main__":
    unittest.main()
