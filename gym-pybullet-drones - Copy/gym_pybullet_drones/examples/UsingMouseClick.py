import os
import time
import argparse
import numpy as np
import cv2
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 200
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

class PyBulletWaypointGenerator:
    def __init__(self, client, altitude=1.0):
        self.waypoints = []
        self.client = client
        self.altitude = altitude
        self.last_mouse_click_time = 0
        self.line_id = None
        self.marker_ids = []
        
        # Enable mouse events and disable some GUI elements for better clicking
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        
    def process_mouse_clicks(self):
        """Process mouse clicks and add waypoints"""
        current_time = time.time()
        mouse_events = p.getMouseEvents(physicsClientId=self.client)
        
        for event in mouse_events:
            # Check for left mouse button click (eventType 2 is mouse button, event[3] is button index)
            if len(event) >= 4 and event[0] == 2 and event[3] == 0:  # Left mouse button
                # Get mouse coordinates
                mouse_x = event[1]
                mouse_y = event[2]
                
                # Get ray from mouse click
                width, height, viewMat, projMat, _, _, _, _, _, _, _, _ = p.getDebugVisualizerCamera(physicsClientId=self.client)
                ray_from, ray_to, _ = p.rayTestFromMouse(mouse_x, mouse_y, physicsClientId=self.client)
                
                # Calculate intersection with ground plane (z=0)
                ray_dir = np.array(ray_to) - np.array(ray_from)
                if ray_dir[2] != 0:  # Avoid division by zero
                    t = -ray_from[2] / ray_dir[2]
                    if t > 0:  # Only if pointing downward
                        hit_pos = np.array(ray_from) + t * ray_dir
                        new_waypoint = [hit_pos[0], hit_pos[1], self.altitude]
                        
                        # Add new waypoint if not too close to last one
                        if len(self.waypoints) == 0 or \
                           np.linalg.norm(np.array(new_waypoint[:2]) - np.array(self.waypoints[-1][:2])) > 0.5:
                            self.waypoints.append(new_waypoint)
                            print(f"Added waypoint at: {new_waypoint}")
                            self.update_visualization()
    
    def update_visualization(self):
        """Update the path visualization in PyBullet"""
        # Remove old markers and lines
        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id, physicsClientId=self.client)
        for marker_id in self.marker_ids:
            p.removeUserDebugItem(marker_id, physicsClientId=self.client)
        
        self.marker_ids = []
        
        # Draw new path if we have at least 2 waypoints
        if len(self.waypoints) > 1:
            self.line_id = p.addUserDebugLine(
                self.waypoints[-2], 
                self.waypoints[-1], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=2,
                lifeTime=0,
                physicsClientId=self.client
            )
        
        # Draw all waypoint markers
        for i, wp in enumerate(self.waypoints):
            color = [0, 1, 0] if i == 0 else [1, 0, 0]  # Green for first, red for others
            self.marker_ids.append(
                p.addUserDebugPoints(
                    [wp], 
                    pointColorsRGB=[color], 
                    pointSize=10,
                    lifeTime=0,
                    physicsClientId=self.client
                )
            )
    
    def get_waypoints(self):
        """Return the current list of waypoints"""
        return self.waypoints
    
class NadirCoverageTracker:
    def __init__(self, area_size=15, resolution=0.1, 
                 focal_length=4e-3, sensor_width=6.17e-3, sensor_height=4.55e-3):
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
        
    def visualize(self, window_name="Coverage Map", scale=1):
        rotated_map = cv2.rotate(self.coverage_map, cv2.ROTATE_180)
        flip_map = cv2.flip(rotated_map, 1)  # Flip vertically
        display_img = cv2.resize(flip_map, 
                               (self.grid_size*scale, self.grid_size*scale),
                               interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)

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
    
    NUM_DRONES = 1
    INIT_XYZS = np.array([[0, 0, 0.1]])  # Start at origin
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=NUM_DRONES,
                         initial_xyzs=INIT_XYZS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
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
            physicsClientId=PYB_CLIENT
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize coverage tracker ###########################
    coverage = NadirCoverageTracker(
        focal_length=4.3e-3,
        sensor_width=6.17e-3,
        sensor_height=4.55e-3,
        resolution=0.1
    )

    # Initialize the generator
    waypoint_generator = PyBulletWaypointGenerator(PYB_CLIENT, altitude=1.0)

    # In your main loop:
    waypoint_generator.process_mouse_clicks()
    WAYPOINTS = waypoint_generator.get_waypoints()

    #### Main simulation loop #################################
    action = np.zeros((NUM_DRONES,4))
    START = time.time()
    wp_counter = 0
    
    print("Click in the PyBullet window to add waypoints...")
    print("The drone will follow the waypoints in order.")

    try:
        for i in range(0, int(duration_sec*env.CTRL_FREQ)):
            #### Check for new waypoints ###########################
            waypoint_generator.update()
            WAYPOINTS = waypoint_generator.waypoints
            
            #### Step the simulation ###############################
            obs, reward, terminated, truncated, info = env.step(action)
            positions = [obs[j][0:3] for j in range(NUM_DRONES)]
            altitude = positions[0][2]

            timesimulation = time.time() - START
            if i % 10 == 0:
                print(f"Time: {timesimulation:.2f}s | Waypoint {wp_counter+1}/{len(WAYPOINTS)}")

            #### Update coverage map ###############################
            if i % 5 == 0:
                coverage.update(positions, altitude)
                if i % 50 == 0:
                    coverage.visualize()
                    print(f"Coverage: {coverage.get_coverage_percentage():.2f}%")

            #### Drone control logic ##############################
            if len(WAYPOINTS) > 1:  # Need at least 2 waypoints to move
                current_pos = obs[0][0:3]
                target_pos = WAYPOINTS[wp_counter]
                direction = target_pos - current_pos
                distance = np.linalg.norm(direction)
                
                # If close to the waypoint, move to the next one
                if distance < 0.2:
                    wp_counter = (wp_counter + 1) % len(WAYPOINTS)
                    target_pos = WAYPOINTS[wp_counter]
                    direction = target_pos - current_pos
                    print(f"Moving to waypoint {wp_counter+1}/{len(WAYPOINTS)}: {target_pos}")
                
                # Calculate velocity
                velocity = (direction / np.linalg.norm(direction)) * 0.5  # Normalize and scale velocity
                action[0, 0:3] = velocity
                action[0, 3] = 0.99  # Keep the throttle high

                # Draw line to current target
                p.addUserDebugLine(current_pos, target_pos, [0, 1, 1], 1, lifeTime=0.1, physicsClientId=PYB_CLIENT)

            #### Log the simulation ###############################
            logger.log(drone=0,
                     timestamp=i/env.CTRL_FREQ,
                     state=obs[0],
                     control=np.hstack([action[0, 0:3], np.zeros(9)]))
            
            #### Sync the simulation ##############################
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
                
                # Keep camera fixed on top view
                if i % 100 == 0:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=15,
                        cameraYaw=0,
                        cameraPitch=-89.99,
                        cameraTargetPosition=[0,0,0],
                        physicsClientId=PYB_CLIENT
                    )

            #### Printout #########################################
            env.render()

    finally:
        #### Close the environment ##############################
        env.close()
        cv2.destroyAllWindows()
        print(f"Final Coverage: {coverage.get_coverage_percentage():.2f}%")

    #### Plot the simulation results ##########################
    logger.save_as_csv("waypoint_path")
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone Waypoint Following with PyBullet Mouse Clicks')
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool, help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool, help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, help='Duration of the simulation in seconds (default: 200)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))