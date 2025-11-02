"""Utility functions for aerial robotics mission planning."""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def clip_angle(angle: float) -> float:
    """Wrap an angle to the ``[-pi, pi]`` interval."""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Compute the Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def manhattan_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Compute the Manhattan distance between two points."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return float(np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]))


def check_collinearity(
    point_a: Sequence[int], point_b: Sequence[int], point_c: Sequence[int]
) -> bool:
    """
    Return True when three points are collinear.

    The check relies on the determinant of the 2x2 matrix formed by the vectors
    ``AB`` and ``BC`` being zero.
    """
    return (
        (point_b[1] - point_a[1]) * (point_c[0] - point_b[0])
        - (point_b[0] - point_a[0]) * (point_c[1] - point_b[1])
    ) == 0


def streamline_route(points: np.ndarray) -> np.ndarray:
    """Prune collinear intermediate points to simplify a route."""
    if len(points) <= 2:
        return points

    streamlined_route = [points[0]]
    for index in range(1, len(points) - 1):
        if not check_collinearity(points[index - 1], points[index], points[index + 1]):
            streamlined_route.append(points[index])
    streamlined_route.append(points[-1])
    return np.asarray(streamlined_route)


def build_route_from_positions(
    positions: np.ndarray, current_position: np.ndarray
) -> np.ndarray:
    """
    Build a greedy route that visits every position starting from the closest one.
    """
    if positions.size == 0:
        return positions

    augmented_positions = np.vstack([positions, current_position])
    start_index = len(augmented_positions) - 1

    remaining: set[int] = set(range(len(positions)))
    route_indices: List[int] = []
    current_index = start_index

    while remaining:
        candidates = np.array(sorted(remaining))
        distances = np.linalg.norm(
            augmented_positions[candidates] - augmented_positions[current_index], axis=1
        )
        next_index = int(candidates[np.argmin(distances)])
        route_indices.append(next_index)
        remaining.remove(next_index)
        current_index = next_index

    return augmented_positions[route_indices]


def check_cell_validity(matrix: np.ndarray, position: Tuple[int, int]) -> bool:
    """Return True when a grid cell is within bounds and marked as free."""
    row, col = position
    rows, cols = matrix.shape
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    return matrix[row, col] > 0


def get_neighbors(
    matrix: np.ndarray, cell: Tuple[int, int], mode: str = "euclidean"
) -> List[Tuple[int, int]]:
    """Return valid neighbor cells for the supplied grid cell."""
    if mode == "manhattan":
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        offsets = tuple(
            (dx, dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )

    neighbors: List[Tuple[int, int]] = []
    for dx, dy in offsets:
        neighbor = (cell[0] + dx, cell[1] + dy)
        if check_cell_validity(matrix, neighbor):
            neighbors.append(neighbor)
    return neighbors


def _reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(
    map_grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    mode: str = "euclidean",
) -> Tuple[List[Tuple[int, int]], Optional[float]]:
    """
    Run A* on the supplied grid.

    The grid is first converted to a binary occupancy representation where
    positive values indicate free space.
    """
    occupancy = np.where(map_grid < -0.5, 0, 1)
    heuristic = euclidean_distance if mode == "euclidean" else manhattan_distance

    open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = _reconstruct_path(came_from, current)
            distance = 0.0
            for node, next_node in zip(path[:-1], path[1:]):
                distance += heuristic(node, next_node)
            return path, round(distance, 2)

        neighbors = get_neighbors(occupancy, current, mode=mode)
        for neighbor in neighbors:
            tentative_score = g_score[current] + 1.0
            if neighbor in g_score and tentative_score >= g_score[neighbor]:
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_score
            f_score = tentative_score + heuristic(neighbor, end)
            heapq.heappush(open_set, (f_score, neighbor))

    return [], None


def path_exists(
    map_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    mode: str = "euclidean",
) -> bool:
    """Return True if there is a feasible path between two cells."""
    path, _ = a_star(map_grid, start, goal, mode=mode)
    return len(path) > 0


def treat_prob_point(
    primary: np.ndarray, fallback: np.ndarray, min_size: int = 5
) -> np.ndarray:
    """
    Prefer returning ``primary`` when it contains enough points,
    otherwise fall back to ``fallback``.
    """
    return primary if primary.size >= min_size else fallback

