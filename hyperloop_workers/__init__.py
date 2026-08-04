"""Dependency-free opaque workers distributed alongside Hyperloop.

Workers remain outside :mod:`black_box_optimizer` so the optimizer never
imports or inspects the implementation it launches.
"""
