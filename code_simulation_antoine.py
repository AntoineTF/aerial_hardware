# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import heapq

# Global variables
on_ground = True
height_desired = 0.5
timer = None
startpos = None
timer_done = None
scan_map = False
last_scan_pos = None
SCAN = False
SCAN_THRESHOLD = 1.0
L_SCAN = False
R_SCAN = False
SCAN_RATE = 0.7
SCAN_WINDOW = 20 #degrees
PREVIOUS_COMMAND = np.array([0.8, 0.0, height_desired, 0.0])
pad_length = 0.4
plot_map = True
cross_path = []
path_todo = []
next_waypoint_interval = 1
waypoint_tolerance = 0.07
DEBUGGING = False
first_a_star = True
found_landpad = False
path_idx = 0
actual_path = []
last_alt = None
mean_alt = []
count = 0
last_delta_alt = None
landing_bool = False
go_back = False
previous_alt = None
final_landing = False

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, PREVIOUS_COMMAND, last_delta_alt, landing_bool, go_back, previous_alt, final_landing

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(19
    control_command = [0.0, 0.0, height_desired, 0.0]
    zone_of_interest = get_corner()
    # Take off
    if startpos is None:
        startpos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']])
        get_next_checkpoint(sensor_data, zone_of_interest)
        first_a_star = False
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False
    
  
    # Go to Landing Region
    if not landing_bool and sensor_data["x_global"] <= 5-1.5:
        control_command = goto_destination_with_scan(zone_of_interest, sensor_data)
        if last_delta_alt is None:
            last_delta_alt = sensor_data['range_down']
        else:   
            mean_alt.append(sensor_data['range_down']-last_delta_alt)
            last_delta_alt = sensor_data['range_down'] 
          
    #Search the Landing Pad
    if not landing_bool and not go_back and sensor_data["x_global"] >= 5-1.5:
        control_command, landing_bool = research(sensor_data)
        
    if landing_bool and sensor_data['range_down'] > 0.01:
        control_command = [0,0,PREVIOUS_COMMAND[2]-0.007,0]
        if previous_alt is None:
            previous_alt = sensor_data['range_down']
        else:
            last_delta_alt = sensor_data['range_down'] - previous_alt
            previous_alt = sensor_data['range_down']
    if last_delta_alt < 0.001 and sensor_data["range_down"] <= 0.02 and sensor_data["x_global"] >= 5-1.5 and not go_back:
        landing_bool = False
        go_back = True    
        previous_alt = None
        over = True
        
    if go_back:
        if sensor_data["x_global"] > 1.8:
            control_command = goto_destination_with_scan(np.array([startpos[0]-0.05,startpos[1]]), sensor_data, "follow")
        else :
            control_command = goto_destination_with_scan(np.array([startpos[0]-0.05,startpos[1]]), sensor_data, "search", a_star_mode="manhattan")
        if previous_alt is None:
            previous_alt = sensor_data['range_down']
        else:
            last_delta_alt = abs(sensor_data['range_down'] - previous_alt)
            previous_alt = sensor_data['range_down']
        if abs(last_delta_alt) >= 0.07 and sensor_data["x_global"] <= 5-1.5:
            go_back = False  
            final_landing = True
            control_command = [0,0,PREVIOUS_COMMAND[2]-0.007,0]
    
    if final_landing:
        control_command = [0,0,PREVIOUS_COMMAND[2]-0.007,0]
        if previous_alt is None:
            previous_alt = sensor_data['range_down']
            last_delta_alt = 1
        else:
            last_delta_alt = sensor_data['range_down'] - previous_alt
            previous_alt = sensor_data['range_down']
    if last_delta_alt < 0.001 and sensor_data["range_down"] <= 0.02 and final_landing:
        control_command = [0,0,0,0]
    
    map = occupancy_map(sensor_data)
    PREVIOUS_COMMAND = control_command
    return control_command # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.15 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

############################################################################################################## PREPARATION OF THE OCCUPANCY MAP + FOLLOWING

map_grid = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied
# set the border to occupied we dont want to leave the grid
map_grid[0,:] = -1
map_grid[-1,:] = -1
map_grid[:,0] = -1
map_grid[:,-1] = -1

obstructed_cells = set()
def occupancy_map(sensor_data):
    global map_grid, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
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
                #map_grid[idx_x, idx_y] += conf <- The A* algorithm is not working with this line of code 
                pass
            else:
                map_grid[idx_x, idx_y] -= conf
                for x in range(-1,2):
                    for y in range(-1,2):
                        if (idx_x+x, idx_y+y) not in obstructed_cells:
                            obstructed_cells.add((idx_x+x, idx_y+y))
                break
    
    map_grid = np.clip(map_grid, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    #if t % 50 == 0:
        #flipped_map = np.flip(map_grid, 1)  # This flips the map horizontally.
        #plt.imshow(flipped_map, vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        #plt.savefig("map.png")
        #plt.close()
    #t +=1
    
    return map_grid

def goto_destination_with_scan(destination, sensor_data, state = "follow", a_star_mode = "euclidean"):
    global on_ground, height_desired, startpos, last_scan_pos, SCAN_THRESHOLD, SCAN, last_scan_yaw, L_SCAN, R_SCAN, SCAN_RATE, SCAN_WINDOW, PREVIOUS_COMMAND
   
    is_scanning, control_commands = scan(sensor_data)

    if is_scanning:
        return control_commands
    else:
        if a_star_mode == "euclidean":
            next_destination = get_next_checkpoint(sensor_data, destination,a_star_mode="euclidean")
        else:
            next_destination = get_next_checkpoint(sensor_data, destination,a_star_mode="manhattan")
        relative_position = next_destination - np.array([sensor_data['x_global'], sensor_data['y_global']])
        relative_position = relative_position.ravel()
        relative_angle = np.rad2deg(clip_angle(np.arctan2(relative_position[1], relative_position[0]) - sensor_data['yaw']))
        if state == "follow":    
            if np.abs(relative_angle) < 10:
                control_command = [0.8, 0.0, height_desired, 0.0]
            elif np.abs(relative_angle) > 80 and np.abs(relative_angle) < 100:
                control_command = [0.0, 0.7*np.sign(relative_angle) , height_desired, 0.0]
            elif np.abs(relative_angle) > 170:
                control_command = [-0.8, 0.0, height_desired, 0.0]
            else:
                control_command = [0.0, 0.0, height_desired, 0.85*np.sign(relative_angle)]
        elif state == "search":
            if np.abs(relative_angle) < 10:
                control_command = [0.35, 0.0, height_desired, 0.0]
            elif np.abs(relative_angle) > 80 and np.abs(relative_angle) < 100:
                control_command = [0.0, 0.4*np.sign(relative_angle) , height_desired, 0.0]
            elif np.abs(relative_angle) > 170:
                control_command = [-0.35, 0.0, height_desired, 0.0]
            else:
                control_command = [0.0, 0.0, height_desired, 0.6*np.sign(relative_angle)]
        else:
            pass
        
    return control_command

def scan(sensor_data):
    global last_scan_pos, SCAN_THRESHOLD, SCAN, last_scan_yaw, L_SCAN, R_SCAN, SCAN_RATE, SCAN_WINDOW
    # Initialize the scan
    if last_scan_pos is None:
        last_scan_pos = [startpos[0], startpos[1]]	
    actual_position = np.array([sensor_data['x_global'], sensor_data['y_global']])
    if euclidial_distance(actual_position, last_scan_pos) >= SCAN_THRESHOLD:
        SCAN = True
        last_scan_pos = actual_position
        last_scan_yaw = sensor_data['yaw']
    
    if not SCAN:
        return SCAN, [0.0, 0.0, height_desired, 0.0]
    
    # Scanning
    if SCAN:
        delta_angle = np.rad2deg(clip_angle(sensor_data['yaw'] - last_scan_yaw))
        if not L_SCAN:
            control_command = [0.0, 0.0, height_desired, SCAN_RATE]
            if delta_angle >= (SCAN_WINDOW/2):
                L_SCAN = True
                control_command = [0.0, 0.0, height_desired, 0.0]
        elif not R_SCAN:
            control_command = [0.0, 0.0, height_desired, -SCAN_RATE]
            if delta_angle <= -(SCAN_WINDOW/2):
                R_SCAN = True
                control_command = [0.0, 0.0, height_desired, 0.0]
        else:
            control_command = [0.0, 0.0, height_desired, SCAN_RATE]
        
        if delta_angle <= 5 and L_SCAN == True and R_SCAN == True:
            SCAN = False
            L_SCAN = False
            R_SCAN = False
            control_command = [0.0, 0.0, height_desired, 0.0]
            
        return SCAN, control_command
            

def get_corner():
    resolution = 0.1  # Increment for y-axis coordinates
    y_positions = np.arange(0, 3.0, resolution)
    constant_x = (5.0 - 2.0 * res_pos) * np.ones_like(y_positions)
    corner_coordinates = np.column_stack((constant_x, y_positions))
    return corner_coordinates

def get_next_checkpoint(sensor_data, obj, a_star_mode = "euclidean"):
    current_pos = np.array([sensor_data['x_global'], sensor_data['y_global']])
    start_point = convert_to_grid_index(current_pos)
    if obj.ndim == 1:
        obj = obj.reshape(1,-1) 
    grid = np.unique(convert_to_grid_index(obj), axis=0) 
    best_find_path, d_small = [], np.inf
    
    map = occupancy_map(sensor_data)
    if first_a_star or ((sensor_data["range_front"] < 0.6 or sensor_data["range_left"] < 0.6 or sensor_data["range_right"] < 0.6 or sensor_data["range_back"] < 0.6) and (sensor_data["range_front"]-sensor_data["range_right"] > 0.6 or sensor_data["range_front"]-sensor_data["range_left"] > 0.6)):
        
        for point in grid: #Condition pour pas le run constamment, condition pour que meme quand il est pas sure il run 
            int_point = tuple(int(x) for x in point)
            if a_star_mode == "euclidean":
                path, dist = a_star(map, tuple(start_point),int_point, mode='euclidean')
            else:
                path, dist = a_star(map, tuple(start_point),int_point, mode='manhattan')
            if dist is not None and dist < d_small:
                best_find_path, d_small = np.array(path), dist
                
    if len(best_find_path) > 0:      
        if plot_map:
            cross_path.append(start_point)
            path_todo = best_find_path
            
        best_find_path = best_find_path*res_pos
        next_checkpoint_idx = next_waypoint_interval if next_waypoint_interval < len(best_find_path) -1 else -1
        d_small = d_small*res_pos
        
        if euclidial_distance(best_find_path[next_checkpoint_idx],current_pos) < waypoint_tolerance:
            next_waypoint_idx = next_waypoint_idx + 1 if next_waypoint_idx + 1 < len(best_find_path) - 1 else -1
        return best_find_path[next_checkpoint_idx]
    else:
        return obj
    
############################################################################################################## RESEARCH
    
research_grid = np.unique(np.mgrid[int((5 - 1.5)/res_pos)+2:map_grid.shape[0]-1, 1:map_grid.shape[1]-1].reshape(2,-1).T, axis = 0)

def research(sensor_data):
    global research_grid, obstructed_cells, found_landpad, actual_path, path_idx, actual_position, last_alt,mean_alt
#The obj of this function is to define a path to the landing pad, we basically need to explore all the landing region
    actual_position = convert_to_grid_index(np.array([sensor_data['x_global'], sensor_data['y_global']]))
#If the landing pad is not where we are right know, we can remove this point from the list of research points
    right_neighbor = [actual_position[0] + 1, actual_position[1]]
    to_remove = np.where(
        (research_grid == actual_position).all(axis=1) |
        (research_grid == right_neighbor).all(axis=1) 
    )[0]
    research_grid = np.delete(research_grid, to_remove, axis=0)

    path = None
# While we are searching for the pad and we are still discovering obstacles, we need to update the map consequently
    print("obstructed_cells",obstructed_cells)
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
        print("path",path)
        
    if path is None:
        if path_idx == len(actual_path) - 1:
            path_idx = 0
        elif euclidean_dist(actual_position*res_pos, actual_path[path_idx]*res_pos) < 0.1:
            path_idx += 1
    else :
        actual_path = path
        path_idx = 0
                    
    control_com = goto_destination_with_scan(actual_path[path_idx]*res_pos,sensor_data,"search")
    
# Stopping condition if we find the landing pad
    if last_alt is None:
        last_alt = sensor_data['range_down']
    else:
        delta_alt = sensor_data['range_down'] - last_alt
        last_alt = sensor_data['range_down']
        if abs(delta_alt) >= 0.08: 
            found_landpad = True
            control_com = [0,0,height_desired,0]   
    print("obstructed_cells",obstructed_cells)    
    return control_com, found_landpad

############################################################################################################## TOOLS    

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


############################################################################################################## SMALL CALCULATIONS AND TOOLS

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