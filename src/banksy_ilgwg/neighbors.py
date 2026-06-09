"""Vectorized spatial k-nearest-neighbour construction for BANKSY.

The original pipeline fits one ``NearestNeighbors(algorithm="ball_tree")`` object
and issues a separate ``kneighbors`` query per azimuthal order ``m`` (requesting
``k_geom * (m + 1)`` neighbours each time). We mirror that exactly: fit once,
query per order.

Per-order querying is deliberate. On gridded data (e.g. ``bin20``) many
neighbours are equidistant, so the k-NN set is tie-ambiguous. Issuing a single
large query and slicing its head would pick different tie-winners than separate
per-order queries, changing the neighbour-mean block.

Backend: ``scipy.spatial.cKDTree`` with ``leafsize=16`` and parallel queries
(``workers=-1``). For 2-D coordinates this is ~9× faster than sklearn's
``ball_tree`` at the cost of a different (equally valid) tie-breaking convention
for equidistant points. On bin20 data most cells sit on a regular grid with many
equidistant neighbours, so tie-breaking was already arbitrary; this backend makes
that explicit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_LEAFSIZE = 16


class SpatialNeighborhood:
    """A fitted spatial k-NN model that issues distance-sorted, self-excluded queries."""

    def __init__(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("coords must have shape (n_obs, n_dims>=2)")
        self.coords = coords
        self.n_obs = int(coords.shape[0])
        self._tree = cKDTree(coords, leafsize=_LEAFSIZE)

    def query(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(indices, distances, deltas)`` for the ``k`` nearest neighbours.

        ``deltas`` are ``coord[nbr] - coord[self]`` over the first two spatial
        dimensions, shape ``(n_obs, k, 2)`` (for azimuthal angles). The cell
        itself is excluded, matching the original.
        """
        if k < 1:
            raise ValueError("k must be positive")
        if k > self.n_obs - 1:
            raise ValueError(
                f"k={k} exceeds available neighbours (n_obs - 1 = {self.n_obs - 1})"
            )
        # Request k+1 neighbours: cKDTree always includes the query point itself
        # (distance 0) as the first result when querying against the training set.
        distances, indices = self._tree.query(self.coords, k=k + 1, workers=-1)
        distances = distances[:, 1:]   # drop self (distance 0, column 0)
        indices = indices[:, 1:]
        deltas = self.coords[indices, :2] - self.coords[:, None, :2]
        return indices, distances, deltas
