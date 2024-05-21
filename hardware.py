import logging
import time
from threading import Timer
import numpy as np
import heapq 
import keyboard
import matplotlib.pyplot as plt

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
import cv2

# Variables to store state estimates
state_estimate = {
    'x': 0.0,
    'y': 0.0,
    'z': 0.0,
    'yaw': 0.0
}

sensor_data = {
    'range_front': 0.0,
    'range_back': 0.0,
    'range_left': 0.0,
    'range_right': 0.0
}
# BOOLEAN VARIABLES + ARRAYS + DICTIONARIES + CONSTANTS
startpose = None#
threshold_local = 0.15
SCAN = False
plot_map = True
first_a_star = True
landing_bool = False
next_waypoint_interval = 1 
last_delta_alt = None
last_scan_pos = None
waypoint_tolerance = 0.07
height_desired = 0.3
L_SCAN = False
R_SCAN = False
SCAN_THRESHOLD = 1.0
SCAN_RATE = 0.7*100
SCAN_WINDOW = 20 #degrees
final_landing = False
previous_alt = None
t = 0
go_back = False
path_idx = 0
actual_path = []
mean_alt = []
cross_path = []
path_todo = []
last_alt = None
found_landpad = False


x_irl = 3.27
y_irl = 1.5

uri = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E702')

# Occupanxy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2 # meter, maximum range of distance sensor
res_pos = 0.2 # meter
conf = 0.1 # certainty given by each measurement
t = 0 # only for plotting

map_grid = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied
# set the border to occupied we dont want to leave the grid
map_grid[0,:] = -1
map_grid[-1,:] = -1
map_grid[:,0] = -1
map_grid[:,-1] = -1

obstructed_cells = set()

def occupancy_map(sensor_data):
    print("sensor_data",sensor_data)
    global map_grid,state_estimate,t
    pos_x = state_estimate['x']
    pos_y = state_estimate['y']
    yaw = state_estimate['yaw']
    print("state_estimate",state_estimate)
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map, +1 and -1 are added to avoid modifying the border
            if idx_x < 0+1 or idx_x >= map_grid.shape[0]-1 or idx_y < 0+1 or idx_y >= map_grid.shape[1]-1 or dist > range_max:
                break

            # update the map
            if dist < measurement:
                #map_grid[idx_x, idx_y] += conf #<- The A* algorithm is not working with this line of code 
                pass
            else:
                print("exp")
                map_grid[idx_x, idx_y] -= conf
                for x in range(-1,2):
                   for y in range(-1,2):
                       if (idx_x+x, idx_y+y) not in obstructed_cells:
                           obstructed_cells.add((idx_x+x, idx_y+y))
                break
    
    map_grid = np.clip(map_grid, -1, 1) # certainty can never be more than 100%
    #print("map_grid",map_grid)  
    #only plot every Nth time step (comment out if not needed)

    if t % 1 == 0:
        flipped_map = np.flip(map_grid, 1)  # This flips the map horizontally.
        plt.imshow(flipped_map, vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
        print("Image has been saved.")
        plt.close()
    t +=1
    
    return map_grid

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(50, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        global state_estimate, sensor_data
        state_estimate['x'] = data['stateEstimate.x']+x_irl
        state_estimate['y'] = data['stateEstimate.y']+y_irl
        state_estimate['z'] = data['stateEstimate.z']
        state_estimate['yaw'] = np.deg2rad(data['stabilizer.yaw'])
        
        sensor_data['range_front'] = data['range.front']/1000
        sensor_data['range_back'] = data['range.back']/1000
        sensor_data['range_left'] = data['range.left']/1000
        sensor_data['range_right'] = data['range.right']/1000
        
        """Callback from a the log API when data arrives"""
        #print(f'[{timestamp}][{logconf.name}]: ', end='')
        #for name, value in data.items():
        #    print(f'{name}: {value:3.3f} ', end='')
        #print()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False
        
def get_corner():
    resolution = 0.1
    y_positions = np.arange(0,3.0,resolution)
    constant_x = (5.0 - 2.0 * res_pos) * np.ones_like(y_positions)
    corner_coordinates = np.column_stack((constant_x, y_positions))
    return corner_coordinates

def get_next_checkpoint(obj,a_star_mode = "euclidean"):
    global state_estimate, sensor_data
    current_pos = np.array([state_estimate['x'],state_estimate['y']])
    #print("Current position in simulation",current_pos)
    if obj.ndim == 1:
        obj = obj.reshape(1,-1)
    grid_coords = convert_to_grid_index(current_pos) 
    #print("simulation position in grid format",grid_coords)
    grid = np.unique(convert_to_grid_index(obj), axis=0) 
    best_find_path, d_small = [], np.inf
    occupancy_map(sensor_data)
    #if first_a_star or ((sensor_data["range_front"] < 90 or sensor_data["range_left"] < 90 or sensor_data["range_right"] < 90 or sensor_data["range_back"] < 90) and (sensor_data["range_front"]-sensor_data["range_right"] > 90 or sensor_data["range_front"]-sensor_data["range_left"] > 90)):
    if first_a_star or ((sensor_data["range_front"] < 250 or sensor_data["range_left"] < 250 or sensor_data["range_right"] < 250 or sensor_data["range_back"] < 250)):
        #print("#####################################################REDOING ASTAR#############################################")
        for point in grid: #Condition pour pas le run constamment, condition pour que meme quand il est pas sure il run 
            int_point = tuple(int(x) for x in point)
            if a_star_mode == "euclidean":
                path, dist = a_star(map_grid, tuple(grid_coords),int_point, mode='euclidean')
            else:
                path, dist = a_star(map_grid, tuple(grid_coords),int_point, mode='manhattan')
            if dist is not None and dist < d_small:
                #print("int_point",int_point)
                best_find_path, d_small = np.array(path), dist
                
    if len(best_find_path) > 0:      
        if plot_map:
            cross_path.append(grid_coords)
            path_todo = best_find_path
            
        best_find_path = best_find_path*res_pos
        d_small = d_small*res_pos
        next_checkpoint_idx = next_waypoint_interval if next_waypoint_interval < len(best_find_path) -1 else -1
        if euclidial_distance(best_find_path[next_checkpoint_idx],current_pos) < waypoint_tolerance:
            next_checkpoint_idx = next_checkpoint_idx + 1 if next_checkpoint_idx + 1 < len(best_find_path) - 1 else -1
            
        #print("best_find_path[next_checkpoint_idx]",best_find_path[next_checkpoint_idx])
        return best_find_path[next_checkpoint_idx]
    else:
        return obj
    
def convert_to_grid_index(location):
    # Convert a location to grid coordinates, considering resolution
    if isinstance(location, (int, float)):
        return int(np.round(location / res_pos))
    # Handle the conversion for vector-like locations
    return np.array(np.round(location / res_pos), dtype=int)

def a_star(mapGrid, startNode, endNode, mode='euclidean'):
    
    

    if mode == 'manhattan':
        heuristicFunction = manhattan_dist
        isValidNeighbor = lambda x,y: (x == 0 and y == 0) or (x != 0 and y != 0)
    elif mode == 'euclidean':
        heuristicFunction = euclidean_dist
        isValidNeighbor = lambda x,y: x == 0 and y == 0

    priorityQueue = [(0, startNode)]
    previousNodeMap = {}
    g_scores = {startNode: 0}
    f_scores = {startNode: heuristicFunction(startNode, endNode)}

    while priorityQueue:
        currentNode = heapq.heappop(priorityQueue)[1]
        if currentNode == endNode:
            finalPath = [currentNode]
            while currentNode in previousNodeMap:
                currentNode = previousNodeMap[currentNode]
                finalPath.append(currentNode)
            finalPath.reverse()
            totalDistance = 0
            for currentPathNode, nextPathNode in zip(finalPath[:-1], finalPath[1:]):
                totalDistance += heuristicFunction(currentPathNode, nextPathNode)
            return finalPath, np.round(totalDistance, 2)
        
        for neighbor in get_neighbors(mapGrid, currentNode, isValidNeighbor):
            tentative_g_score = g_scores[currentNode] + 1
            if neighbor in g_scores and tentative_g_score >= g_scores[neighbor]:
                continue
            previousNodeMap[neighbor] = currentNode
            g_scores[neighbor] = tentative_g_score
            f_scores[neighbor] = tentative_g_score + heuristicFunction(neighbor, endNode)
            heapq.heappush(priorityQueue, (f_scores[neighbor], neighbor))
    return None, None

def goto_destination_with_scan(destination, sensor_data, state = "follow", a_star_mode = "euclidean"):
    global height_desired

    is_scanning, control_commands = scan(sensor_data)

    if is_scanning:
        return control_commands
    else:
        if a_star_mode == "euclidean":
            next_destination = get_next_checkpoint(destination,a_star_mode="euclidean")
            #print("next_destination",next_destination)
        else:
            next_destination = get_next_checkpoint(destination,a_star_mode="manhattan")
            #print("next_destination",next_destination)
        relative_position = next_destination - np.array([state_estimate['x'], state_estimate['y']])
        relative_position = relative_position.ravel()
        #relative_angle = np.rad2deg(clip_angle(np.arctan2(relative_position[1], relative_position[0]) - state_estimate['yaw']))
        relative_angle = np.rad2deg(clip_angle(np.arctan2(relative_position[1], relative_position[0]) - state_estimate['yaw']))
        #print("relative_angle",relative_angle)
        abs_relative_angle = np.abs(relative_angle)  
        if state == "follow":    
            if abs_relative_angle < 10:
                #print("front")
                control_command = [0.35, 0.0, 0.0, height_desired]
            elif abs_relative_angle > 55 and abs_relative_angle < 100:
                #print("cote")
                control_command = [0.0, 0.5*np.sign(relative_angle) , 0.0, height_desired]
            elif abs_relative_angle > 170:
                #print("back")
                control_command = [-0.35, 0.0, 0.0, height_desired]
            else:
                #print("turning en legende")
                control_command = [0.0, 0.0, -0.70*100*np.sign(relative_angle),height_desired]
        elif state == "search":
            if abs_relative_angle < 10:
                control_command = [0.35, 0.0, 0.0,height_desired]
            elif abs_relative_angle > 55 and abs_relative_angle < 100:
                control_command = [0.0, 0.4*np.sign(relative_angle)*10 , 0.0,height_desired]
            elif abs_relative_angle > 170:
                control_command = [-0.35, 0.0, 0.0, height_desired]
            else:
                control_command = [0.0, 0.0, -0.6*np.sign(relative_angle)*10    ,height_desired]
        else:
            pass
        
    return control_command

def scan(sensor_data):
    global last_scan_pos, SCAN_THRESHOLD, SCAN, last_scan_yaw, L_SCAN, R_SCAN, SCAN_RATE, SCAN_WINDOW,height_desired
    # Initialize the scan
    if last_scan_pos is None:
        last_scan_pos = [startpose[0], startpose[1]]	
    actual_position = np.array([state_estimate['x'], state_estimate['y']])
    #print("last_scan_pos",last_scan_pos,"actual_position",actual_position,"euclidial_distance",euclidial_distance(actual_position, last_scan_pos))
    if euclidial_distance(actual_position, last_scan_pos) >= SCAN_THRESHOLD:
        SCAN = True
        last_scan_pos = actual_position
        last_scan_yaw = state_estimate['yaw']
    
    if not SCAN:
        return SCAN, [0.0, 0.0, 0.0,height_desired]
    
    # Scanning
    if SCAN:
        delta_angle = np.rad2deg(clip_angle(state_estimate['yaw'] - last_scan_yaw))
        if not L_SCAN:
            control_command = [0.0, 0.0, SCAN_RATE,height_desired]
            if delta_angle >= (SCAN_WINDOW/2):
                L_SCAN = True
                control_command = [0.0, 0.0, 0.0,height_desired]
        elif not R_SCAN:
            control_command = [0.0, 0.0, -SCAN_RATE,height_desired]
            if delta_angle <= -(SCAN_WINDOW/2):
                R_SCAN = True
                control_command = [0.0, 0.0, 0.0,height_desired]
        else:
            control_command = [0.0, 0.0, SCAN_RATE,height_desired]
        
        if delta_angle <= 5 and L_SCAN == True and R_SCAN == True:
            SCAN = False
            L_SCAN = False
            R_SCAN = False
            control_command = [0.0, 0.0, 0.0,height_desired]
            
        return SCAN, control_command
    
research_grid = np.unique(np.mgrid[int((5 - 1.5)/res_pos)+2:map_grid.shape[0]-1, 1:map_grid.shape[1]-1].reshape(2,-1).T, axis = 0)
    
def research(sensor_data):
    global research_grid, obstructed_cells, found_landpad, actual_path, path_idx, actual_position, last_alt
#The obj of this function is to define a path to the landing pad, we basically need to explore all the landing region
    actual_position = convert_to_grid_index(np.array([state_estimate["x"], state_estimate["y"]]))
#If the landing pad is not where we are right know, we can remove this point from the list of research points
    right_neighbor = [actual_position[0] + 1, actual_position[1]]
    to_remove = np.where(
        (research_grid == actual_position).all(axis=1) |
        (research_grid == right_neighbor).all(axis=1) 
    )[0]
    research_grid = np.delete(research_grid, to_remove, axis=0)

    path = None
# While we are searching for the pad and we are still discovering obstacles, we need to update the map consequently
    if len(obstructed_cells) > 0:   
# Loading and treating new points - updating the research points
        # Calculate the threshold x-coordinate
        x_threshold = int((5 - 1.5) / res_pos)
        # Filter out obstructed_cells that should be removed based on the x-coordinate
        to_remove = {cell for cell in obstructed_cells if cell[0] >= x_threshold}
        # Remove these points from research_grid
        exploring_tuples = list(map(tuple, research_grid))
        # Create a set of tuples from exploring_points_to_remove for fast lookup
        remove_set = set(map(tuple, to_remove))
        # Filter exploring_points using list comprehension and set membership
        research_grid = np.array([point for point in exploring_tuples if point not in remove_set])
        # Reset new_occupied_cells
        obstructed_cells = set()

# Preprocessing the point where the landing pad cannot be 
        for point in research_grid:
            if not path_exists(actual_position, point):
                point_array = np.array(point)
                mask = ~(research_grid == point_array).all(axis=1)
                research_grid = research_grid[mask]
# Now that every point in the grid is accessible, we can start the exploration by defining the path
        probable_point = research_grid[(research_grid[:,0] > int((5 - 1.5)/res_pos) + 2) & (research_grid[:,0] < map_grid.shape[0]-2) & (research_grid[:,1] > 1) & (research_grid[:,1] < map_grid.shape[1]-2)]
        path = np.array(build_route_from_positions(treat_prob_point(probable_point,research_grid),actual_position))
        print("path research",path)

        
    if path is None:
        print("Path is None")
        if path_idx == len(actual_path) - 1:
            path_idx = 0
        elif euclidean_dist(actual_position*res_pos, actual_path[path_idx]*res_pos) < 0.1:
            path_idx += 1
    else :
        actual_path = path
        path_idx = 0
                    
    control_com = goto_destination_with_scan(actual_path[path_idx]*res_pos,sensor_data,a_star_mode="search")
    
# Stopping condition if we find the landing pad
    # if last_alt is None:
    #     last_alt = state_estimate['z']
    # else:
    #     delta_alt = state_estimate['z'] - last_alt
    #     last_alt = state_estimate['z']
    #     if abs(delta_alt) >= 10: 
    #         found_landpad = True
    #         control_com = [0,0,height_desired,0]   
    found_landpad = False
        
    return control_com, found_landpad

###############################################################################TOOLS
def path_exists(actual_position, point):
    #Check precisely why manhattan here and not euclidean
    path, _ = a_star(map_grid, tuple(actual_position),tuple(point), mode='euclidean')
    return path is not None

def streamline_route(points):
    # Begin with the initial point in the route
    streamlined_route = [points[0]]

    # Iterate over each point, excluding the first and last
    for index in range(1, len(points) - 1):
        # Determine if a point can be skipped based on collinearity
        if not check_collinearity(points[index - 1], points[index], points[index + 1]):
            streamlined_route.append(points[index])  # Include this point if it's crucial for path shape

    # Always include the final point to complete the route
    streamlined_route.append(points[-1])

    return streamlined_route

def build_route_from_positions(positions, current_position):
    route = []
    positions = np.append(positions, np.array([current_position]), axis=0)
    nearest_position = np.argmin(np.linalg.norm(positions - current_position, axis=1))
    route_sequence = [nearest_position]

    # Continuously search for the closest next position
    while len(route_sequence) < len(positions):
        recent_position = route_sequence[-1]
        position_distances = np.linalg.norm(positions[recent_position] - positions, axis=1)
        position_distances[route_sequence] = np.inf  # Ignore already included positions
        next_closest = np.argmin(position_distances)
        route_sequence.append(next_closest)

    route_sequence.remove(nearest_position)  # Remove the initial added current position

    route = [positions[index] for index in route_sequence]
    return route

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle

def euclidial_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors(grid, cell, isValidNeighbor):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if isValidNeighbor(i,j):
                continue
            neighbor = (cell[0] + i, cell[1] + j)
            if check_cell_validity(grid, neighbor):
                neighbors.append(neighbor)
    return neighbors

def check_cell_validity(matrix, position):
    # Check if the position coordinates are within the matrix boundaries
    row, col = position
    rows, cols = matrix.shape
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    # Ensure the matrix value at the position is non-negative
    if matrix[row, col] < 0:
        return False
    return True

def check_collinearity(pointA, pointB, pointC):
    # Calculate the determinant of a matrix formed by three points
    # to check if the points lie on the same line
    determinant = (pointB[1] - pointA[1]) * (pointC[0] - pointB[0]) - (pointB[0] - pointA[0]) * (pointC[1] - pointB[1])
    return determinant == 0



def convert_to_grid_index(location):
    # Convert a location to grid coordinates, considering resolution
    if isinstance(location, (int, float)):
        return int(np.round(location / res_pos))
    # Handle the conversion for vector-like locations
    return np.array(np.round(location / res_pos), dtype=int)

def treat_prob_point(set_point_1, set_point_2):
    if set_point_1.size >=5:
        return set_point_1
    else:
        return set_point_2
    
##############################################################################MAIN

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    le = LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    while le.is_connected:
        # Taking off
        occupancy_map(sensor_data)
        if startpose is None:
            zone_of_interest = get_corner()
            startpose = np.array([state_estimate['x'],state_estimate['y'],state_estimate['z']])
            #print("BEGINNING- STARTPOSE",startpose, "last_scan_pos",last_scan_pos)
            get_next_checkpoint(zone_of_interest)
            first_a_star = False
        time.sleep(0.01)
        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 18) #(vx,vy,yaw, range_down)
            time.sleep(0.1)
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.3)
            time.sleep(0.1)
        
       # Go to Landing Region
        while not landing_bool and state_estimate["x"] <= 5-1.5:
            for _ in range(1):
                if sensor_data["range_front"] < threshold_local:
                    #print("local front", sensor_data["range_front"])
                    control_command = [-0.1, 0, 0, 0.3]
                elif sensor_data["range_left"] < threshold_local:
                    #print("local left", sensor_data["range_left"])
                    control_command = [0, -0.1, 0, 0.3]
                elif sensor_data["range_right"] < threshold_local:
                    #print("local right", sensor_data["range_right"])
                    control_command = [0, 0.1, 0, 0.3]  
                elif sensor_data["range_back"] < threshold_local:
                    #print("local back", sensor_data["range_back"])
                    control_command = [0.1, 0, 0, 0.3]
                else:
                    control_command = goto_destination_with_scan(zone_of_interest, sensor_data,a_star_mode="euclidean")
                    #print("normal")
                #print("control_command",control_command)    
                cf.commander.send_hover_setpoint(control_command[0], control_command[1], control_command[2], control_command[3])

                time.sleep(0.1)
            if last_delta_alt is None:
                last_delta_alt = state_estimate['z']
            else:   
                mean_alt.append(state_estimate['z']-last_delta_alt)
                last_delta_alt = state_estimate['z']
                        
        # Arrived at landing region and looking for the pad
        while not landing_bool and not go_back:
            print("SEARCHING GRID")
            control_command, landing_bool = research(sensor_data)
            
            
            
        for y in range(30):
            print("LANDING")
            cf.commander.send_hover_setpoint(0, 0, 0, (10 - y/3) / 25)
            time.sleep(0.1)
        cf.commander.send_stop_setpoint()
        break
    
    