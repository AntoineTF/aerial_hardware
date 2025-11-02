# Aerial Robotics Landing Mission

Implementation of the final project for the EPFL *Aerial Robotics* course (MICRO-502, Prof. Dario Floreano).  
The objective is to autonomously steer a Crazyflie nano‑drone from its take-off pad to a remote landing platform while mapping the environment, avoid obstacles, perform a precision landing, then take off again to return and land on the original platform.

## Mission Highlights
- **Structured flight pipeline** – take-off, exploration, search, checkpoint landing, return flight, and final landing are handled via a state machine.
- **Occupancy-grid mapping** – builds a live grid from the four multiranger sensors and feeds the A* path planner.
- **Local obstacle avoidance** – combines reactive avoidance with path replanning and periodic yaw scans to refine the map.
- **Search strategy** – greedy coverage pattern over the exploration half of the arena to discover the landing pad.

## Repository Layout

```
hardware_if_research.py  # Main mission script
mission_utils.py         # Shared utilities (A*, distance helpers, geometry tools)
final_run_vid.mp4        # Example flight showcasing the full mission
Aerial_presentation.pptx # Presentation slides prepared for the course
```

## Requirements

Hardware:
- Crazyflie 2.x with Flow Deck v2 and Multi-Ranger Deck
- Crazyradio PA (or equivalent) for 2.4 GHz communication

Software:
- Python 3.9+
- [`cflib`](https://github.com/bitcraze/crazyflie-lib-python) (Crazyflie radio interface)
- `numpy`

Optional:
- `matplotlib` if you enable map recording

Install dependencies with:

```bash
pip install cflib numpy matplotlib
```

## Running the Mission

1. Connect the Crazyradio PA and ensure the Crazyflie is powered with the decks installed.
2. Adjust the URI in `hardware_if_research.py` if you are not using the default `radio://0/20/2M/E7E7E7E702`.
3. (Recommended) Calibrate the Crazyflie sensors and verify the Flow deck height readings in the Crazyflie client.
4. Launch the mission:

   ```bash
   python hardware_if_research.py
   ```

5. The script will:
   - Reset the Kalman filter,
   - Ascend to the target altitude,
   - Navigate to the exploration wall,
   - Search and land on the intermediate pad,
   - Take off again and return to the start.

6. Upon completion the motors are disarmed and the drone remains on the final platform.

If you want to observe the generated occupancy grid, set `map_recording=True` in `MissionConfig`. The current implementation keeps the hook ready but does not write files by default.

## Media

- `final_run_vid.mp4` – A short clip of the real run covering map generation, landing-pad detection, and the return flight.

You can add additional clips to this section for better visibility when sharing the repository.

## Extending the Project

- Plug a SLAM back-end or external motion capture data to improve localisation.
- Log and visualise the occupancy map over time.
- Replace the greedy search with frontier-based exploration to speed up pad discovery.
- Add safety checks (battery, timeout) before initiating the return flight.

## Acknowledgements

This work was produced as part of the Aerial Robotics class at EPFL.  
Many thanks to the course staff and teammates for feedback, testing time, and arena setup support.

