import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import cv2
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from shapely.geometry import Polygon, LineString

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary



DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True  # Enable obstacles by default
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 200
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False



class NadirCoverageTracker:
    def __init__(self, area_size=15, resolution=0.1, 
                 focal_length=4e-3, sensor_width=6.17e-3, sensor_height=4.55e-3):
        """
        Args:
            focal_length (m): Lens focal length (e.g., 4mm for DJI cameras)
            sensor_width/height (m): Physical sensor dimensions
        """
        self.resolution = resolution
        self.grid_size = int(area_size / resolution)
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.area_offset = area_size / 2
        
        # Camera intrinsic parameters
        self.f = focal_length
        self.sensor_w = sensor_width
        self.sensor_h = sensor_height
        
        # Pre-compute FOVs
        self.h_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        self.v_fov = 2 * np.arctan(sensor_height / (2 * focal_length))

    def calculate_footprint(self, altitude):
        """Calculate ground footprint rectangle for nadir camera"""
        ground_w = 2 * altitude * np.tan(self.h_fov / 2)
        ground_h = 2 * altitude * np.tan(self.v_fov / 2)
        return (ground_w, ground_h)

    def update(self, drone_positions, altitude):
        ground_w, ground_h = self.calculate_footprint(altitude)
        pixel_w = int(ground_w / self.resolution)
        pixel_h = int(ground_h / self.resolution)
        
        temp_map = np.copy(self.coverage_map)
        
        for pos in drone_positions:
            x_center = int((pos[0] + self.area_offset) / self.resolution)
            y_center = int((pos[1] + self.area_offset) / self.resolution)
            
            # Create rectangular footprint
            x_min = max(0, x_center - pixel_w//2)
            x_max = min(self.grid_size-1, x_center + pixel_w//2)
            y_min = max(0, y_center - pixel_h//2)
            y_max = min(self.grid_size-1, y_center + pixel_h//2)
            
            temp_map[y_min:y_max, x_min:x_max] = 255
            
        self.coverage_map = temp_map

        
    def get_coverage_percentage(self):
        covered = np.sum(self.coverage_map > 0)
        return (covered / self.coverage_map.size) * 100
        
    def visualize(self, window_name="Coverage Map", scale=1, positions=None, altitude=None):
        if positions and altitude:
            w, h = self.calculate_footprint(altitude)
            for pos in positions:
                x = int((pos[0] + self.area_offset) / self.resolution)
                y = int((pos[1] + self.area_offset) / self.resolution)
                cv2.rectangle(display_img, 
                            (x-w//2, y-h//2), (x+w//2, y+h//2),
                            (0,255,0), 1)  # Draw green rectangle
                
        rotated_map = cv2.rotate(self.coverage_map, cv2.ROTATE_180)
        flip_map = cv2.flip(rotated_map, 1)  # Flip vertically
        display_img = cv2.resize(flip_map, 
                               (self.grid_size*scale, self.grid_size*scale),
                               interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)



import numpy as np
from shapely.geometry import Polygon, LineString
import pybullet as p

class ManualBoundaryDrawer:
    def __init__(self, client):
        self.client = client
        self.points = []
        self.line_ids = []
        self.drawing_complete = False
        self.prev_mouse_state = [0] * 10  # Track previous mouse states
        
    def start_drawing(self):
        print("Left-click to add boundary points")
        print("Press 'E' to finish, 'D' to delete last point")
        p.addUserDebugText("Draw boundary (Left-click points, E to finish)", 
                         [0,0,1.5], [1,1,0], 1.5, 0, physicsClientId=self.client)

    def add_point(self, position):
        if self.drawing_complete:
            return
            
        self.points.append(position[:2])  # Store only x,y
        point_height = 0.1  # Slightly above ground
        point_pos = [position[0], position[1], point_height]
        
        # Visual feedback
        p.addUserDebugPoints([point_pos], [[1,0,0]], 10, physicsClientId=self.client)
        
        # Draw line between points
        if len(self.points) > 1:
            line_start = [*self.points[-2], point_height]
            line_end = [*self.points[-1], point_height]
            line_id = p.addUserDebugLine(line_start, line_end, [1,0,0], 2, 
                                       physicsClientId=self.client)
            self.line_ids.append(line_id)
        
    def finish_drawing(self):
        if len(self.points) < 3:
            print("Need at least 3 points to define an area")
            return False
            
        # Close the polygon
        self.points.append(self.points[0])
        point_height = 0.1
        line_start = [*self.points[-2], point_height]
        line_end = [*self.points[-1], point_height]
        line_id = p.addUserDebugLine(line_start, line_end, [1,0,0], 2, 
                                   physicsClientId=self.client)
        self.line_ids.append(line_id)
        
        self.drawing_complete = True
        return True
        
    def clear_last(self):
        if not self.points:
            return
            
        self.points.pop()
        if self.line_ids:
            p.removeUserDebugItem(self.line_ids.pop(), physicsClientId=self.client)
            
    def get_boundary(self):
        if not self.drawing_complete or len(self.points) < 4:
            return None
        return np.array(self.points[:-1])  # Exclude closing point

def generate_boustrophedon_path(boundary_points, spacing=1.5, altitude=1.0):
    """Generate coverage path within boundary"""
    if boundary_points is None or len(boundary_points) < 3:
        return None
        
    polygon = Polygon(boundary_points)
    min_x, min_y, max_x, max_y = polygon.bounds
    
    waypoints = []
    current_y = min_y
    direction = 1  # 1 = right, -1 = left
    
    while current_y <= max_y:
        if direction == 1:
            line = LineString([(min_x, current_y), (max_x, current_y)])
        else:
            line = LineString([(max_x, current_y), (min_x, current_y)])
        
        intersection = polygon.intersection(line)
        
        if not intersection.is_empty:
            if intersection.geom_type == 'LineString':
                for point in list(intersection.coords):
                    waypoints.append([point[0], point[1], altitude])
            elif intersection.geom_type == 'MultiLineString':
                for segment in intersection.geoms:
                    for point in list(segment.coords):
                        waypoints.append([point[0], point[1], altitude])
        
        current_y += spacing
        direction *= -1
    
    return np.array(waypoints)

"""# Modify your waypoint generation in the run() function:
def run(**args):
    # ... existing setup code ...
    
    #### Generate waypoints using Boustrophedon Decomposition ####
    planner = BoustrophedonCoveragePlanner(
        building_size=15,
        altitude=2.5,
        camera_fov=75,
        overlap=0.2
    )
    WAYPOINTS = [planner.generate_path()]
    
    # ... rest of your existing code ..."""


def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    

    

    #### Initialize the simulation #############################
    """INIT_XYZS = np.array([
                          [ 0, 0, .1],
                          [.3, 0, .1],
                          [.6, 0, .1],
                          [0.9, 0, .1]
                          ])"""
    


    NUM_DRONES = 1  # Change this to number of drones you want to simulate
#Change this to number of drones you want to simulate

    INIT_XYZS = np.array([
        #[-5, 5, 0.1],   # Drone 0: Top-left quadrant
        #[5, 5, 0.1],    # Drone 1: Top-right quadrant
        #[-5, -5, 0.1],  # Drone 2: Bottom-left quadrant
        [5, -5, 0.1]    # Drone 3: Bottom-right quadrant
    ])
    """INIT_RPYS = np.array([
                          [0, 0, 0],
                          [0, 0, np.pi/3],
                          [0, 0, np.pi/4],
                          [0, 0, np.pi/2]
                          ])"""
    PHY = Physics.PYB

    #### Load the URDF file for the building layout ###########
    """if obstacles:
        URDF_PATH = "path/to/your/maze_layout.urdf"  # Update this path
        building_id = p.loadURDF(URDF_PATH, useFixedBase=True)"""

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=NUM_DRONES,
                         initial_xyzs=INIT_XYZS,
                         #initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,  # Pass the obstacles flag
                         user_debug_gui=user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    #### Set up top-down view camera ###########################
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=0,
            cameraPitch=-89.99,
            cameraTargetPosition=[0,0,0],
            physicsClientId=env.getPyBulletClient()
        )
        # Hide unnecessary UI elements for cleaner view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    
    #### Define coverage areas for each drone #################
    #AREA_SIZE = 10  # Total area is 20x20 (-10 to 10)
    #SUB_AREA_SIZE = AREA_SIZE / 2  # Each drone covers 10x10 area
    #SPACING = 1.0   # Distance between parallel paths
    #ALTITUDE = 2  # Flying altitude

    # Define boundaries for each drone's area
    AREAS = [
        # Drone 0: Top-left (-10 to 0 in x, 0 to 10 in y)
        (-10, 0, 0, 10),
        # Drone 1: Top-right (0 to 10 in x, 0 to 10 in y)
        (0, 10, 0, 10),
        # Drone 2: Bottom-left (-10 to 0 in x, -10 to 0 in y)
        (-10, 0, -10, 0),
        # Drone 3: Bottom-right (0 to 10 in x, -10 to 0 in y)
        (0, 10, -10, 0)
    ]
    
    
    #### Initialize Boundary Drawing ##########################
    boundary_drawer = ManualBoundaryDrawer(PYB_CLIENT)
    boundary_drawer.start_drawing()
    
    #### Wait for Boundary Drawing ###########################
    print("Drawing coverage area boundary...")
    while not boundary_drawer.drawing_complete:
        # Handle mouse events
        mouse_events = p.getMouseEvents(physicsClientId=PYB_CLIENT)
        
        for event in mouse_events:
            # Event structure: (mouse button, button state, x, y, z)
            if event[0] == 2 and event[1] == 0:  # Left button released
                # Raycast to ground
                ray_from = [0, 0, 10]
                ray_to = [event[2], event[3], 0]
                ray_result = p.rayTest(ray_from, ray_to, physicsClientId=PYB_CLIENT)
                
                if ray_result[0][0] != -1:  # Hit something
                    hit_pos = ray_result[0][3]
                    boundary_drawer.add_point(hit_pos)
        
        # Handle keyboard
        keys = p.getKeyboardEvents(PYB_CLIENT)
        if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
            boundary_drawer.finish_drawing()
        elif ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
            boundary_drawer.clear_last()
        
        p.stepSimulation(physicsClientId=PYB_CLIENT)
        time.sleep(1./240.)
    
    #### Generate Coverage Path ##############################
    boundary = boundary_drawer.get_boundary()
    if boundary is None:
        print("Invalid boundary, using default area")
        boundary = np.array([[-5,-5], [5,-5], [5,5], [-5,5]])
    
    WAYPOINTS = [generate_boustrophedon_path(boundary)]
    
    if WAYPOINTS[0] is not None:
        # Visualize path
        for i in range(len(WAYPOINTS[0])-1):
            p.addUserDebugLine(WAYPOINTS[0][i], WAYPOINTS[0][i+1],
                             [0,1,0], 2, lifeTime=0,
                             physicsClientId=PYB_CLIENT)



    #### Generate waypoints for each drone #####################
    #WAYPOINTS = [generate_square_spiral_path(BUILDING_SIZE, SPACING, ALTITUDE)]

    #WAYPOINTS = [planner.generate_path()]

    #### Initialize waypoint counters and current waypoints ####
    wp_counters = [0 for _ in range(NUM_DRONES)]
    current_waypoints = [WAYPOINTS[i][0] for i in range(NUM_DRONES)]

    #### Define sensor parameters ##############################
    NUM_SENSORS = 4  # Number of sensors per drone
    SENSOR_ANGLES = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)  # Sensor directions (radians)
    SENSOR_RANGE = 15.0  # Maximum range of each sensor

    
    #### Visualize area boundaries ############################
    if gui:
        for i in range(len(WAYPOINTS[0])-1):
            p.addUserDebugLine(
                WAYPOINTS[0][i], 
                WAYPOINTS[0][i+1], 
                lineColorRGB=[0,1,0], 
                lineWidth=2,
                lifeTime=0,
                physicsClientId=PYB_CLIENT
            )
        
    coverage = NadirCoverageTracker(
    focal_length=4.3e-3,       # 4.3mm lens
    sensor_width=6.17e-3,      # 1/2" CMOS sensor
    sensor_height=4.55e-3,
    resolution=0.1            # 10cm grid
    )

    # At 10m altitude:
    print(coverage.calculate_footprint(10))  # Returns (14.2m, 10.5m) coverage

    # Initialize coverage tracker (20x20m area, 0.1m resolution)
    #coverage = NadirCoverageTracker(area_size=15, resolution=0.1, camera_fov_deg=60)
    
    try:
        #### Run the simulation ####################################
        action = np.zeros((NUM_DRONES,4))
        START = time.time()
        for i in range(0, int(duration_sec*env.CTRL_FREQ)):

            #### Step the simulation ###################################
            obs, reward, terminated, truncated, info = env.step(action)
            positions = [obs[j][0:3] for j in range(NUM_DRONES)]
            altitude = positions[0][2]

            timesimulation = time.time() - START

            print(f"Time: {timesimulation:.2f}s")

            if i % 5 == 0:
                coverage.update(positions, altitude)
                if i % 50 == 0:  # Update visualization less frequently
                    coverage.visualize()
                    print(f"Coverage: {coverage.get_coverage_percentage():.2f}%")

            #### Compute control for each drone #######################
            for j in range(NUM_DRONES):
                current_pos = obs[j][0:3]
                target_pos = WAYPOINTS[j][wp_counters[j]]
                direction = target_pos - current_pos
                distance = np.linalg.norm(direction)
                if distance < 0.1:  # If close to the waypoint, move to the next one
                    wp_counters[j] = (wp_counters[j] + 1) % len(WAYPOINTS[j])
                    target_pos = WAYPOINTS[j][wp_counters[j]]
                    direction = target_pos - current_pos
                velocity = (direction / np.linalg.norm(direction) * 0.5)  # Normalize and scale velocity
                action[j, 0:3] = velocity
                action[j, 3] = 0.99  # Keep the throttle high

                # Draw drone's path (optional)
                if gui and i % 10 == 0:
                    p.addUserDebugLine(
                        current_pos, 
                        target_pos, 
                        lineColorRGB=[1,0,0], 
                        lifeTime=1,
                        physicsClientId=env.getPyBulletClient()
                    )

                #### Simulate distance sensors ##########################
                sensor_measurements = []
                for angle in SENSOR_ANGLES:
                    # Calculate sensor direction vector
                    sensor_dir = np.array([np.cos(angle), np.sin(angle), 0])
                    sensor_end = current_pos + sensor_dir * SENSOR_RANGE

                    # Perform raycasting
                    ray_result = p.rayTest(current_pos, sensor_end, physicsClientId=PYB_CLIENT)
                    hit_object_id = ray_result[0][0]
                    hit_fraction = ray_result[0][2]

                    if hit_object_id != -1:  # If the ray hits an obstacle
                        distance_to_obstacle = hit_fraction * SENSOR_RANGE
                    else:  # If no obstacle is detected
                        distance_to_obstacle = SENSOR_RANGE

                    sensor_measurements.append(distance_to_obstacle)

                #### Display sensor measurements on the PyBullet screen ###
                text_position = current_pos + np.array([0, 0, 0.5])  # Position above the drone
                text = f"Drone {j} Sensors: {[f'{d:.2f}' for d in sensor_measurements]}, Time: {timesimulation:.2f}s"
                p.addUserDebugText(text, text_position, textColorRGB=[1, 0, 0], textSize=1, lifeTime=0.1, physicsClientId=PYB_CLIENT)

            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
                # Keep camera fixed on top view (in case user tries to rotate)
                if i % 100 == 0:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=15,
                        cameraYaw=0,
                        cameraPitch=-89.99,
                        cameraTargetPosition=[0,0,0],
                        physicsClientId=env.getPyBulletClient()
                    )
            
            #### Log the simulation ####################################
            for j in range(NUM_DRONES):
                logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j],
                        control=np.hstack([action[j, 0:3], np.zeros(9)])
                        )

            #### Printout ##############################################
            env.render()

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

    finally:
        #### Close the environment #################################
        env.close()
        cv2.destroyAllWindows()
        print(f"Final Coverage: {coverage.get_coverage_percentage():.2f}%")

    #### Plot the simulation results ###########################
    logger.save_as_csv("coverage_path") # Optional CSV save
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Coverage Path Planning using VelocityAviary')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))