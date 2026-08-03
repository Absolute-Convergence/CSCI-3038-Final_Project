"""
Built in registry of the search algorithms this project knows about.

NOTE!!!! The interface summary from section 11.1 shows a shortened version
of this, so I followed from section 7.1 instead because it gives the full
interface! If anybody need to change this later, quick check the
AGENTS.md file for the reasoning. :)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from black_box_optimizer.models import AlgorithmSpec
from black_box_optimizer.search.base import SearchAlgorithm
from black_box_optimizer.search.nsga2 import NSGA2
from black_box_optimizer.search.random_search import RandomSearch

AlgorithmFactory = Callable[[AlgorithmSpec], SearchAlgorithm]


class UnknownAlgorithmError(ValueError):
    """The requested algorithm name is not registered."""


def build_random_search(spec: AlgorithmSpec) -> SearchAlgorithm:
    """Build a RandomSearch from the supplied algorithm settings."""
    return RandomSearch(seed=spec.seed)


def build_nsga2(spec: AlgorithmSpec) -> SearchAlgorithm:
    """Build an NSGA-II search algorithm from the supplied settings.

    An additional, opt-in algorithm -- nothing about registering it here
    changes what build_random_search()/RandomSearch do above.
    """
    return NSGA2(seed=spec.seed)


# Just a plain built in mapping for there is no plugin system here
ALGORITHM_REGISTRY: Mapping[str, AlgorithmFactory] = {
    "random_search": build_random_search,
    "nsga2": build_nsga2,
}


def create_algorithm(spec: AlgorithmSpec) -> SearchAlgorithm:
    """
    Build the search algorithm named in the AlgorithmSpec.

    Unknown names fail here instead of making it farther into the
    optimizer before something breaks.
    """
    factory = ALGORITHM_REGISTRY.get(spec.name)

    if factory is None:
        valid = sorted(ALGORITHM_REGISTRY)
        raise UnknownAlgorithmError(
            f"unknown algorithm {spec.name!r}; must be one of {valid}"
        )

    return factory(spec)