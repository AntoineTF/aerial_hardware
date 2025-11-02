"""Crazyflie mission script for the EPFL Aerial Robotics project.

The mission is divided into three stages:

1. Take off from the starting platform and reach the exploration zone.
2. Search for a landing pad while building an occupancy grid with the four range
   sensors and avoiding obstacles using A* waypoints.
3. Take off again and return to the original launch corridor to perform a
   precision landing.

The original implementation relied heavily on global state. This refactored
version groups configuration and runtime data in dedicated data classes, adds
type hints, and centralises the navigation helpers in ``mission_utils.py``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Timer
from typing import List, Optional, Sequence

import numpy as np

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from mission_utils import (
    a_star,
    build_route_from_positions,
    clip_angle,
    euclidean_distance,
    manhattan_distance,
    path_exists,
    treat_prob_point,
)


class FlightPhase(Enum):
    """High-level mission phases."""

    TAKE_OFF = auto()
    FOLLOW = auto()
    SEARCH = auto()
    LAND = auto()
    RETURN = auto()
    DONE = auto()


@dataclass
class StateEstimate:
    """Pose estimate provided by the onboard Kalman filter."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass
class SensorReadings:
    """Range readings (in metres) from the four multiranger sensors."""

    range_front: float = 0.0
    range_back: float = 0.0
    range_left: float = 0.0
    range_right: float = 0.0


@dataclass(frozen=True)
class MissionConfig:
    """Configuration constants for the mission."""

    threshold_local: float = 0.25
    threshold_astar: float = 0.20
    next_waypoint_interval: int = 1
    waypoint_tolerance: float = 0.07
    height_desired: float = 0.45
    scan_threshold: float = 1.0
    scan_rate_deg: float = 40.0
    scan_window_deg: float = 45.0
    scan_reset_tolerance_deg: float = 5.0
    map_min_x: float = 0.0
    map_max_x: float = 5.0
    map_min_y: float = 0.0
    map_max_y: float = 3.0
    map_resolution: float = 0.15
    range_max: float = 2.0
    occupancy_confidence: float = 0.1
    landing_zone_boundary_x: float = 3.5
    return_target_x: float = 0.0
    landing_detection_altitude_delta: float = 0.08
    final_landing_x_threshold: float = 1.5
    start_pose_fallback: Sequence[float] = (1.0, 1.5, 0.01)
    map_recording: bool = False
    x_offset: float = 1.0
    y_offset: float = 1.5


@dataclass
class MissionRuntime:
    """Mutable state that evolves over the course of the mission."""

    config: MissionConfig
    flight_phase: FlightPhase = FlightPhase.TAKE_OFF
    start_pose: Optional[np.ndarray] = None
    first_a_star: bool = True
    last_scan_pos: Optional[np.ndarray] = None
    last_scan_yaw: float = 0.0
    scan_active: bool = False
    scan_left_complete: bool = False
    scan_right_complete: bool = False
    previous_control_command: List[float] = field(init=False)
    previous_search_command: List[float] = field(init=False)
    on_ground: bool = True
    second_takeoff: bool = True
    actual_path: List[np.ndarray] = field(default_factory=list)
    path_idx: int = 0
    landing_coord: Optional[np.ndarray] = None
    found_landing_pad: bool = False
    timing_research: int = 0
    last_altitude_sample: Optional[float] = None
    last_final_altitude_sample: Optional[float] = None
    delta_final_altitude: float = 0.0
    found_final_landing_pad: bool = False
    research_grid: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        hover = [0.0, 0.0, 0.0, self.config.height_desired]
        self.previous_control_command = hover.copy()
        self.previous_search_command = hover.copy()


@dataclass
class OccupancyGrid:
    """Grid-based world representation built from the multiranger sensors."""

    config: MissionConfig
    grid: np.ndarray = field(init=False)
    obstructed_cells: set[tuple[int, int]] = field(default_factory=set)
    step_counter: int = 0

    def __post_init__(self) -> None:
        cells_x = int((self.config.map_max_x - self.config.map_min_x) / self.config.map_resolution)
        cells_y = int((self.config.map_max_y - self.config.map_min_y) / self.config.map_resolution)
        self.grid = np.zeros((cells_x, cells_y))
        self.grid[0, :] = -1
        self.grid[-1, :] = -1
        self.grid[:, 0] = -1
        self.grid[:, -1] = -1

    def update(
        self,
        state: StateEstimate,
        sensors: SensorReadings,
        record_map: bool = False,
    ) -> np.ndarray:
        """Integrate the newest range measurements into the occupancy grid."""
        positions = [sensors.range_front, sensors.range_left, sensors.range_back, sensors.range_right]
        pos_x, pos_y, yaw = state.x, state.y, state.yaw

        for idx, measurement in enumerate(positions):
            yaw_sensor = yaw + idx * (np.pi / 2.0)
            max_steps = int(self.config.range_max / self.config.map_resolution)
            for step in range(max_steps):
                dist = step * self.config.map_resolution
                idx_x = int(
                    np.round(
                        (pos_x - self.config.map_min_x + dist * np.cos(yaw_sensor))
                        / self.config.map_resolution,
                        0,
                    )
                )
                idx_y = int(
                    np.round(
                        (pos_y - self.config.map_min_y + dist * np.sin(yaw_sensor))
                        / self.config.map_resolution,
                        0,
                    )
                )
                if (
                    idx_x <= 0
                    or idx_x >= self.grid.shape[0] - 1
                    or idx_y <= 0
                    or idx_y >= self.grid.shape[1] - 1
                    or dist > self.config.range_max
                ):
                    break

                if measurement <= 0:
                    continue

                if dist < measurement:
                    continue

                self.grid[idx_x, idx_y] -= self.config.occupancy_confidence
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        self.obstructed_cells.add((idx_x + dx, idx_y + dy))
                break

        np.clip(self.grid, -1, 1, out=self.grid)
        self.step_counter += 1
        # Optional map recording hooks can be added here if desired.
        return self.grid


def get_landing_region(config: MissionConfig) -> np.ndarray:
    """Create a lattice of candidate landing points along the exploration wall."""
    resolution = 0.1
    y_positions = np.arange(config.map_min_y, config.map_max_y, resolution)
    constant_x = (config.map_max_x - 2.0 * config.map_resolution) * np.ones_like(y_positions)
    return np.column_stack((constant_x, y_positions))


def convert_to_grid_index(location: Sequence[float], resolution: float) -> np.ndarray:
    """Convert metric coordinates to grid indices."""
    arr = np.asarray(location)
    return np.array(np.round(arr / resolution), dtype=int)


def should_recompute_path(
    config: MissionConfig,
    runtime: MissionRuntime,
    sensors: SensorReadings,
    scanning_active: bool,
) -> bool:
    """Return True when a new path should be generated."""
    if runtime.first_a_star or scanning_active:
        return True
    sensor_values = [
        sensors.range_front,
        sensors.range_back,
        sensors.range_left,
        sensors.range_right,
    ]
    return any(value < config.threshold_astar for value in sensor_values if value > 0)


def get_next_checkpoint(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    state: StateEstimate,
    sensors: SensorReadings,
    destination: np.ndarray,
    scanning_active: bool,
    mode: str = "euclidean",
) -> np.ndarray:
    """Return the next waypoint along the A* path towards the destination."""
    current_pos = np.array([state.x, state.y])
    if destination.ndim == 1:
        destination = destination.reshape(1, -1)

    grid_coords = convert_to_grid_index(current_pos, config.map_resolution)
    grid_targets = np.unique(
        convert_to_grid_index(destination, config.map_resolution),
        axis=0,
    )

    best_path: Optional[np.ndarray] = None
    best_distance = np.inf

    if should_recompute_path(config, runtime, sensors, scanning_active):
        for grid_point in grid_targets:
            path, distance = a_star(
                grid_snapshot,
                tuple(grid_coords),
                tuple(int(x) for x in grid_point),
                mode=mode,
            )
            if distance is not None and distance < best_distance:
                best_path = np.asarray(path)
                best_distance = distance
        runtime.first_a_star = False

    if best_path is not None and len(best_path) > 0:
        metric_path = best_path * config.map_resolution
        candidate_idx = (
            config.next_waypoint_interval
            if config.next_waypoint_interval < len(metric_path) - 1
            else -1
        )
        candidate = metric_path[candidate_idx]
        if euclidean_distance(candidate, current_pos) < config.waypoint_tolerance:
            candidate_idx = min(candidate_idx + 1, len(metric_path) - 1)
            candidate = metric_path[candidate_idx]
        return candidate

    return destination[-1]


def scan(
    config: MissionConfig,
    runtime: MissionRuntime,
    state: StateEstimate,
) -> tuple[bool, List[float]]:
    """Perform a local 90-degree sweep to update the occupancy map."""
    if runtime.start_pose is None:
        return False, [0.0, 0.0, 0.0, config.height_desired]

    if runtime.last_scan_pos is None:
        runtime.last_scan_pos = np.array(runtime.start_pose[:2])

    actual_position = np.array([state.x, state.y])
    if euclidean_distance(actual_position, runtime.last_scan_pos) >= config.scan_threshold:
        runtime.scan_active = True
        runtime.last_scan_pos = actual_position
        runtime.last_scan_yaw = state.yaw

    if not runtime.scan_active:
        runtime.scan_left_complete = False
        runtime.scan_right_complete = False
        return False, [0.0, 0.0, 0.0, config.height_desired]

    delta_angle = np.degrees(clip_angle(state.yaw - runtime.last_scan_yaw))

    if not runtime.scan_left_complete:
        command = [0.0, 0.0, config.scan_rate_deg, config.height_desired]
        if delta_angle >= config.scan_window_deg / 2.0:
            runtime.scan_left_complete = True
            command = [0.0, 0.0, 0.0, config.height_desired]
    elif not runtime.scan_right_complete:
        command = [0.0, 0.0, -config.scan_rate_deg, config.height_desired]
        if delta_angle <= -(config.scan_window_deg / 2.0):
            runtime.scan_right_complete = True
            command = [0.0, 0.0, 0.0, config.height_desired]
    else:
        command = [0.0, 0.0, 0.0, config.height_desired]
        if abs(delta_angle) <= config.scan_reset_tolerance_deg:
            runtime.scan_active = False
            runtime.scan_left_complete = False
            runtime.scan_right_complete = False

    return runtime.scan_active, command


def goto_destination_with_scan(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    state: StateEstimate,
    sensors: SensorReadings,
    destination: np.ndarray,
    mode: str = "euclidean",
) -> List[float]:
    """Combine scanning behaviour and waypoint following."""
    is_scanning, scan_command = scan(config, runtime, state)
    if is_scanning:
        return scan_command

    next_destination = get_next_checkpoint(
        config,
        runtime,
        grid_snapshot,
        state,
        sensors,
        destination,
        runtime.scan_active,
        mode=mode,
    )
    relative_position = next_destination - np.array([state.x, state.y])
    relative_angle = np.degrees(
        clip_angle(np.arctan2(relative_position[1], relative_position[0]) - state.yaw)
    )
    abs_angle = abs(relative_angle)

    if runtime.flight_phase in (FlightPhase.FOLLOW, FlightPhase.RETURN):
        if abs_angle < 10:
            command = [0.35, 0.0, 0.0, config.height_desired]
        elif 55 < abs_angle < 100:
            command = [0.0, 0.30 * np.sign(relative_angle), 0.0, config.height_desired]
        elif abs_angle > 170:
            command = [-0.35, 0.0, 0.0, config.height_desired]
        else:
            command = [0.0, 0.0, -50.0 * np.sign(relative_angle), config.height_desired]
    elif runtime.flight_phase == FlightPhase.SEARCH:
        if abs_angle < 10:
            command = [0.30, 0.0, 0.0, config.height_desired]
        elif 55 < abs_angle < 100:
            command = [0.0, 0.30 * np.sign(relative_angle), 0.0, config.height_desired]
        elif abs_angle > 170:
            command = [-0.30, 0.0, 0.0, config.height_desired]
        else:
            command = [0.0, 0.0, -50.0 * np.sign(relative_angle), config.height_desired]
    else:
        command = [0.0, 0.0, 0.0, config.height_desired]

    return command


def initialize_research_grid(occupancy: OccupancyGrid, config: MissionConfig) -> np.ndarray:
    """Allocate the set of cells used to explore the landing zone."""
    start_x_idx = int((config.map_max_x - config.landing_zone_boundary_x) / config.map_resolution) + 2
    grid = np.mgrid[
        start_x_idx : occupancy.grid.shape[0] - 1,
        1 : occupancy.grid.shape[1] - 1,
    ].reshape(2, -1).T
    return np.unique(grid, axis=0)


def research(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    occupancy: OccupancyGrid,
    state: StateEstimate,
    sensors: SensorReadings,
) -> tuple[List[float], bool]:
    """Explore the landing zone using a greedy coverage strategy."""
    if runtime.research_grid is None or len(runtime.research_grid) == 0:
        return runtime.previous_search_command, runtime.found_landing_pad

    actual_position = convert_to_grid_index(
        np.array([state.x, state.y]),
        config.map_resolution,
    )
    right_neighbor = np.array([actual_position[0] + 1, actual_position[1]])
    mask = ~(
        (runtime.research_grid == actual_position).all(axis=1)
        | (runtime.research_grid == right_neighbor).all(axis=1)
    )
    runtime.research_grid = runtime.research_grid[mask]

    path: Optional[np.ndarray] = None
    if occupancy.obstructed_cells and runtime.research_grid.size > 0:
        x_threshold = int(
            (config.map_max_x - config.landing_zone_boundary_x) / config.map_resolution
        )
        remove_set = {
            cell for cell in occupancy.obstructed_cells if cell[0] >= x_threshold
        }
        filtered_points = [
            point
            for point in runtime.research_grid
            if tuple(point) not in remove_set
        ]
        runtime.research_grid = (
            np.array(filtered_points, dtype=int).reshape(-1, 2)
            if filtered_points
            else np.empty((0, 2), dtype=int)
        )

        valid_points: List[np.ndarray] = []
        for point in runtime.research_grid:
            if path_exists(grid_snapshot, tuple(actual_position), tuple(point)):
                valid_points.append(point)

        if valid_points:
            runtime.research_grid = (
                np.array(valid_points, dtype=int).reshape(-1, 2)
            )

        if runtime.research_grid.size > 0:
            probable = runtime.research_grid[
                (runtime.research_grid[:, 0] > x_threshold + 2)
                & (runtime.research_grid[:, 0] < occupancy.grid.shape[0] - 2)
                & (runtime.research_grid[:, 1] > 1)
                & (runtime.research_grid[:, 1] < occupancy.grid.shape[1] - 2)
            ]
        else:
            probable = np.empty((0, 2), dtype=int)

        candidate_points = treat_prob_point(probable, runtime.research_grid)
        if candidate_points.size > 0:
            path = build_route_from_positions(candidate_points, actual_position)

    if path is None:
        if runtime.actual_path:
            target = runtime.actual_path[runtime.path_idx]
            if manhattan_distance(actual_position, target) <= 1:
                runtime.path_idx = (runtime.path_idx + 1) % len(runtime.actual_path)
    else:
        runtime.actual_path = [np.asarray(cell) for cell in path]
        runtime.path_idx = 0

    if runtime.actual_path:
        target_point = runtime.actual_path[runtime.path_idx] * config.map_resolution
        runtime.flight_phase = FlightPhase.SEARCH
        control_command = goto_destination_with_scan(
            config,
            runtime,
            grid_snapshot,
            state,
            sensors,
            np.asarray(target_point),
            mode="manhattan",
        )
        runtime.previous_search_command = control_command
    else:
        control_command = runtime.previous_search_command

    runtime.timing_research += 1

    if runtime.last_altitude_sample is None:
        runtime.last_altitude_sample = state.z
    else:
        delta_alt = state.z - config.height_desired
        if abs(delta_alt) >= config.landing_detection_altitude_delta:
            runtime.found_landing_pad = True
            control_command = [0.0, 0.0, 0.0, config.height_desired]

    return control_command, runtime.found_landing_pad


def perform_takeoff(cf: Crazyflie, config: MissionConfig) -> None:
    """Execute a smooth vertical takeoff."""
    for step in range(20):
        altitude = (step + 1) * config.height_desired / 20.0
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, altitude)
        time.sleep(0.05)
    for _ in range(20):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, config.height_desired)
        time.sleep(0.1)


def perform_landing(cf: Crazyflie, config: MissionConfig) -> None:
    """Descend slowly and cut the motors."""
    for step in range(30):
        altitude = max((10 - step / 3.0) / 25.0, 0.0)
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, altitude)
        time.sleep(0.1)
    cf.commander.send_stop_setpoint()
    time.sleep(2.0)


def perform_secondary_takeoff(
    cf: Crazyflie,
    config: MissionConfig,
    runtime: MissionRuntime,
) -> None:
    """Bring the drone back to cruising altitude after landing on the checkpoint."""
    for step in range(20):
        altitude = (step + 1) / 25.0
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, altitude)
        time.sleep(0.05)
    for _ in range(20):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, config.height_desired)
        time.sleep(0.1)
    runtime.second_takeoff = False


def handle_follow_phase(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    state: StateEstimate,
    sensors: SensorReadings,
    destination: np.ndarray,
) -> List[float]:
    """Navigate towards the exploration wall while avoiding close obstacles."""
    runtime.flight_phase = FlightPhase.FOLLOW
    if sensors.range_front < config.threshold_local:
        command = [-0.1, 0.1, 0.0, config.height_desired]
    elif sensors.range_left < config.threshold_local:
        command = [0.1, -0.1, 0.0, config.height_desired]
    elif sensors.range_right < config.threshold_local:
        command = [0.1, 0.1, 0.0, config.height_desired]
    elif sensors.range_back < config.threshold_local:
        command = [0.1, 0.1, 0.0, config.height_desired]
    else:
        command = goto_destination_with_scan(
            config,
            runtime,
            grid_snapshot,
            state,
            sensors,
            destination,
            mode="euclidean",
        )
    runtime.previous_control_command = command
    return command


def handle_search_phase(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    occupancy: OccupancyGrid,
    state: StateEstimate,
    sensors: SensorReadings,
) -> tuple[List[float], bool]:
    """Search for a landing pad in the exploration area."""
    command, found = research(
        config,
        runtime,
        grid_snapshot,
        occupancy,
        state,
        sensors,
    )
    runtime.previous_control_command = command
    return command, found


def handle_return_phase(
    config: MissionConfig,
    runtime: MissionRuntime,
    grid_snapshot: np.ndarray,
    state: StateEstimate,
    sensors: SensorReadings,
) -> tuple[List[float], bool]:
    """Return to the starting corridor and look for the initial platform."""
    runtime.flight_phase = FlightPhase.RETURN
    command = runtime.previous_control_command
    if not runtime.found_final_landing_pad:
        if sensors.range_front < config.threshold_local:
            command = [-0.1, 0.1, 0.0, config.height_desired]
        elif sensors.range_left < config.threshold_local:
            command = [0.1, -0.1, 0.0, config.height_desired]
        elif sensors.range_right < config.threshold_local:
            command = [0.1, 0.1, 0.0, config.height_desired]
        elif sensors.range_back < config.threshold_local:
            command = [0.1, 0.1, 0.0, config.height_desired]
        else:
            target_y = runtime.landing_coord[1] if runtime.landing_coord is not None else config.start_pose_fallback[1]
            target = np.array([config.return_target_x, target_y])
            command = goto_destination_with_scan(
                config,
                runtime,
                grid_snapshot,
                state,
                sensors,
                target,
                mode="euclidean",
            )

        if runtime.last_final_altitude_sample is None:
            runtime.last_final_altitude_sample = state.z
        else:
            runtime.delta_final_altitude = state.z - config.height_desired
            if abs(runtime.delta_final_altitude) >= config.landing_detection_altitude_delta:
                runtime.found_final_landing_pad = True
    else:
        command = [0.0, 0.0, 0.0, config.height_desired]

    runtime.previous_control_command = command
    return command, runtime.found_final_landing_pad


class LoggingExample:
    """Telemetry helper that keeps the state estimate and sensor data updated."""

    def __init__(
        self,
        link_uri: str,
        config: MissionConfig,
        state: StateEstimate,
        sensors: SensorReadings,
    ) -> None:
        self.config = config
        self.state = state
        self.sensors = sensors
        self._cf = Crazyflie(rw_cache="./cache")

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cf.open_link(link_uri)
        self.is_connected = True

    @property
    def cf(self) -> Crazyflie:
        return self._cf

    def _connected(self, link_uri: str) -> None:
        print(f"Connected to {link_uri}")
        log_conf = LogConfig(name="Stabilizer", period_in_ms=50)
        log_conf.add_variable("stateEstimate.x", "float")
        log_conf.add_variable("stateEstimate.y", "float")
        log_conf.add_variable("stateEstimate.z", "float")
        log_conf.add_variable("stabilizer.yaw", "float")
        log_conf.add_variable("range.front")
        log_conf.add_variable("range.back")
        log_conf.add_variable("range.left")
        log_conf.add_variable("range.right")

        try:
            self._cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._stab_log_data)
            log_conf.error_cb.add_callback(self._stab_log_error)
            log_conf.start()
        except KeyError as exc:
            print(f"Could not start log configuration: {exc}")
        except AttributeError:
            print("Could not add Stabilizer log config, bad configuration.")

        Timer(300, self._cf.close_link).start()

    def _stab_log_error(self, logconf: LogConfig, msg: str) -> None:
        print(f"Error when logging {logconf.name}: {msg}")

    def _stab_log_data(self, timestamp: int, data: dict, logconf: LogConfig) -> None:
        del timestamp, logconf  # Unused in this context
        self.state.x = data["stateEstimate.x"] + self.config.x_offset
        self.state.y = data["stateEstimate.y"] + self.config.y_offset
        self.state.z = data["stateEstimate.z"]
        self.state.yaw = np.deg2rad(data["stabilizer.yaw"])

        self.sensors.range_front = data["range.front"] / 1000.0
        self.sensors.range_back = data["range.back"] / 1000.0
        self.sensors.range_left = data["range.left"] / 1000.0
        self.sensors.range_right = data["range.right"] / 1000.0

    def _connection_failed(self, link_uri: str, msg: str) -> None:
        print(f"Connection to {link_uri} failed: {msg}")
        self.is_connected = False

    def _connection_lost(self, link_uri: str, msg: str) -> None:
        print(f"Connection to {link_uri} lost: {msg}")

    def _disconnected(self, link_uri: str) -> None:
        print(f"Disconnected from {link_uri}")
        self.is_connected = False


def main() -> None:
    logging.basicConfig(level=logging.ERROR)

    config = MissionConfig()
    state_estimate = StateEstimate()
    sensor_data = SensorReadings()
    runtime = MissionRuntime(config=config)
    occupancy_grid = OccupancyGrid(config)

    landing_region = get_landing_region(config)
    uri = uri_helper.uri_from_env(default="radio://0/20/2M/E7E7E7E702")

    cflib.crtp.init_drivers()
    telemetry = LoggingExample(uri, config, state_estimate, sensor_data)
    cf = telemetry.cf

    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)

    runtime.research_grid = initialize_research_grid(occupancy_grid, config)

    while telemetry.is_connected:
        grid_snapshot = occupancy_grid.update(
            state_estimate,
            sensor_data,
            record_map=config.map_recording,
        )

        if runtime.start_pose is None:
            runtime.start_pose = np.array(config.start_pose_fallback)

        if runtime.on_ground:
            perform_takeoff(cf, config)
            runtime.on_ground = False
            runtime.flight_phase = FlightPhase.FOLLOW
            runtime.previous_control_command = [0.0, 0.0, 0.0, config.height_desired]
            runtime.last_altitude_sample = state_estimate.z
            print("Takeoff complete, heading to exploration zone.")
            continue

        command_executed = False

        if (
            runtime.flight_phase == FlightPhase.FOLLOW
            and state_estimate.x <= config.landing_zone_boundary_x
        ):
            command = handle_follow_phase(
                config,
                runtime,
                grid_snapshot,
                state_estimate,
                sensor_data,
                landing_region,
            )
            command_executed = True
        elif runtime.flight_phase != FlightPhase.RETURN and state_estimate.x > config.landing_zone_boundary_x:
            command, found_landing = handle_search_phase(
                config,
                runtime,
                grid_snapshot,
                occupancy_grid,
                state_estimate,
                sensor_data,
            )
            command_executed = True
            if found_landing:
                runtime.landing_coord = np.array([state_estimate.x, state_estimate.y])
                print("Landing pad detected, initiating descent.")
                perform_landing(cf, config)
                runtime.flight_phase = FlightPhase.RETURN
                runtime.second_takeoff = True
                runtime.found_landing_pad = False
                runtime.last_final_altitude_sample = None
                runtime.found_final_landing_pad = False
                runtime.previous_control_command = [0.0, 0.0, 0.0, config.height_desired]
                print("Checkpoint reached. Preparing for return flight.")
                continue
        elif runtime.flight_phase == FlightPhase.RETURN:
            if runtime.second_takeoff:
                perform_secondary_takeoff(cf, config, runtime)
                runtime.previous_control_command = [0.0, 0.0, 0.0, config.height_desired]
                print("Second takeoff complete, returning to home platform.")
                continue
            command, detected_start_pad = handle_return_phase(
                config,
                runtime,
                grid_snapshot,
                state_estimate,
                sensor_data,
            )
            command_executed = True
            if (
                detected_start_pad
                and state_estimate.x < config.final_landing_x_threshold
            ):
                print("Home platform detected, commencing final landing.")
                perform_landing(cf, config)
                runtime.flight_phase = FlightPhase.DONE
                runtime.previous_control_command = [0.0, 0.0, 0.0, 0.0]
                print("Mission completed. Motors disarmed.")
                continue

        if not command_executed:
            command = runtime.previous_control_command

        cf.commander.send_hover_setpoint(*command)
        runtime.previous_control_command = command
        time.sleep(0.1)


if __name__ == "__main__":
    main()
