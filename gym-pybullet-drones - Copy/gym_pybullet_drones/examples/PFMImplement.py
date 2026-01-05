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
from PIL import Image

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

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

def create_obstacles(pyb_client, waypoints, num_obstacles=10, size=0.3):
    """
    Create small obstacles randomly placed near the waypoint path
    Args:
        pyb_client: PyBullet physics client ID
        waypoints: List of waypoints the drone will follow
        num_obstacles: Number of obstacles to create
        size: Size of each obstacle (radius for spheres)
    Returns:
        List of obstacle IDs
    """
    obstacle_ids = []
    
    for _ in range(num_obstacles):
        seg_idx = random.randint(0, len(waypoints)-2)
        start = waypoints[seg_idx]
        end = waypoints[seg_idx+1]
        
        t = random.uniform(0.2, 0.8)
        pos = start + t * (end - start)
        
        pos[0] += random.uniform(-1, 1)
        pos[1] += random.uniform(-1, 1)
        pos[2] = 3.1  # Place on ground
        
        obstacle_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 3],
            physicsClientId=pyb_client
        )
        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 3],
            rgbaColor=[0.8, 0.3, 0.3, 1],
            physicsClientId=pyb_client
        )
        obstacle_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_id,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=pos,
            physicsClientId=pyb_client
        )
        obstacle_ids.append(obstacle_body)
    
    return obstacle_ids

class PotentialFieldController:
    def __init__(self, drone_pos, waypoint, obstacles, max_speed=2.0, max_force=1.0):
        self.max_speed = max_speed
        self.max_force = max_force
        self.attraction_gain = 1.0
        self.repulsion_gain = 100.0
        self.repulsion_range = 0.5
        self.waypoint_threshold = 1.0
        self.slowdown_radius = 2.0
        
    def compute_force(self, drone_pos, waypoint, obstacles, altitude):
        to_waypoint = waypoint - drone_pos
        dist_to_waypoint = np.linalg.norm(to_waypoint)
        
        waypoint_dir = to_waypoint / dist_to_waypoint if dist_to_waypoint > 0 else np.zeros(3)
        
        desired_speed = self.max_speed
        if dist_to_waypoint < self.slowdown_radius:
            desired_speed *= (dist_to_waypoint / self.slowdown_radius)
        
        desired_velocity = waypoint_dir * desired_speed
        steer_attract = desired_velocity * self.attraction_gain
        
        
        steer_repulse = np.zeros(3)
        for obstacle in obstacles:
            obstacle_pos = np.array(obstacle[0:3])
            diff = drone_pos - obstacle_pos
            dist = np.linalg.norm(diff)
            
            if dist < self.repulsion_range and dist > 0:
                repulse_mag = self.repulsion_gain / (dist * dist)
                repulse_mag *= min(1.0, dist_to_waypoint / self.slowdown_radius)
                steer_repulse += (diff / dist) * repulse_mag
        
        total_force = steer_attract + steer_repulse
        if drone_pos[2] > altitude:
            total_force[2] = 0
        
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force and force_mag > 0:
            total_force = total_force / force_mag * self.max_force
        
        return total_force, dist_to_waypoint

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

def generate_square_spiral_path(building_size, spacing, altitude):
    waypoints = []
    half_size = building_size / 2
    current_min = -half_size + spacing
    current_max = half_size - spacing
    
    x, y = current_min, current_min
    waypoints.append([x, y, altitude])
    
    while current_max - current_min > spacing:
        x = current_max
        waypoints.append([x, y, altitude])
        
        y = current_max
        waypoints.append([x, y, altitude])
        
        x = current_min
        waypoints.append([x, y, altitude])
        
        y = current_min + spacing
        waypoints.append([x, y, altitude])
        
        current_min += spacing
        current_max -= spacing
        
        x = current_min
        waypoints.append([x, y, altitude])
    
    waypoints.append([0, 0, altitude])
    
    return np.array(waypoints)

def get_nadir_camera_image(positions, drone_quat, fov=60, resolution=(320, 240), near=0.1, far=100):
    width, height = resolution
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=positions,
        distance=0.001,   # close to drone
        yaw=0,
        pitch=-90,       # look directly downward
        roll=0,
        upAxisIndex=2
    )
    
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )

    img_arr = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    rgb_array = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    return rgb_array

def detect_crack(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    _, tresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Crack", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

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
    
    NUM_DRONES = 2
    INIT_XYZS = np.array([[5, -5, 0.1],
                          [-5, 5, 0.1]])
    PHY = Physics.PYB

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

    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=0,
            cameraPitch=-89.99,
            cameraTargetPosition=[0,0,0],
            physicsClientId=PYB_CLIENT
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=2,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    BUILDING_SIZE = 15
    SPACING = 1.5
    ALTITUDE = 1
    
    WAYPOINTS = [generate_square_spiral_path(BUILDING_SIZE, SPACING, ALTITUDE) for i in range(NUM_DRONES)]

    # Create obstacles along the path
    obstacle_ids = []
    if obstacles:
        obstacle_ids = create_obstacles(PYB_CLIENT, WAYPOINTS[0], 
                                      num_obstacles=20,
                                      size=0.3)

    wp_counters = [0 for _ in range(NUM_DRONES)]
    current_waypoints = [WAYPOINTS[i][0] for i in range(NUM_DRONES)]

    NUM_SENSORS = 4
    SENSOR_ANGLES = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)
    SENSOR_RANGE = 15.0
    
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
        focal_length=4.3e-3,
        sensor_width=6.17e-3,
        sensor_height=4.55e-3,
        resolution=0.1
    )



    #print(coverage.calculate_footprint(10))

    pf_controllers = [
        PotentialFieldController(
            INIT_XYZS[i],
            current_waypoints[i],
            [],
            max_speed=1.5,
            max_force=0.5
        ) for i in range(NUM_DRONES)
    ]
    
    try:
        action = np.zeros((NUM_DRONES,4))
        START = time.time()
        for i in range(0, int(duration_sec*env.CTRL_FREQ)):

            obs, reward, terminated, truncated, info = env.step(action)
            positions = [obs[j][0:3] for j in range(NUM_DRONES)]
            altitude = positions[0][2]

            for d in range(NUM_DRONES):
                cam_img = get_nadir_camera_image(positions[d], None)
                img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
                output_img = detect_crack(img_bgr)
                cv2.imshow(f"Drone {d} Camera", output_img)
                cv2.waitKey(1)



            timesimulation = time.time() - START
            #print(f"Time: {timesimulation:.2f}s")

            if i % 5 == 0:
                coverage.update(positions, altitude)
                if i % 50 == 0:
                    coverage.visualize()
                    #print(f"Coverage: {coverage.get_coverage_percentage():.2f}%")

            for j in range(NUM_DRONES):
                current_pos = obs[j][0:3]
                target_pos = WAYPOINTS[j][wp_counters[j]]
                
                nearby_obstacles = []
                for angle in SENSOR_ANGLES:
                    sensor_dir = np.array([np.cos(angle), np.sin(angle), 0])
                    sensor_end = current_pos + sensor_dir * SENSOR_RANGE
                    ray_result = p.rayTest(current_pos, sensor_end, physicsClientId=PYB_CLIENT)
                    
                    if ray_result[0][0] != -1:
                        hit_pos = np.array(ray_result[0][3])
                        nearby_obstacles.append([hit_pos[0], hit_pos[1], hit_pos[2], SENSOR_RANGE * ray_result[0][2]])
                
                pf_force, dist_to_waypoint = pf_controllers[j].compute_force(
                    current_pos,
                    target_pos,
                    nearby_obstacles,
                    ALTITUDE
                )
                
                if dist_to_waypoint < pf_controllers[j].waypoint_threshold:
                    wp_counters[j] = (wp_counters[j] + 1) % len(WAYPOINTS[j])
                    target_pos = WAYPOINTS[j][wp_counters[j]]
                    pf_controllers[j] = PotentialFieldController(
                        current_pos,
                        target_pos,
                        [],
                        max_speed=1.5,
                        max_force=0.5
                    )

                sensor_measurements = []
                for angle in SENSOR_ANGLES:
                    sensor_dir = np.array([np.cos(angle), np.sin(angle), 0])
                    sensor_end = current_pos + sensor_dir * SENSOR_RANGE
                    ray_result = p.rayTest(current_pos, sensor_end, physicsClientId=PYB_CLIENT)
                    hit_object_id = ray_result[0][0]
                    hit_fraction = ray_result[0][2]

                    if hit_object_id != -1:
                        distance_to_obstacle = hit_fraction * SENSOR_RANGE
                    else:
                        distance_to_obstacle = SENSOR_RANGE

                    sensor_measurements.append(distance_to_obstacle)

                mark = []
                for d in sensor_measurements:
                    if d < 0.5:
                        mark.append(sensor_measurements.index(d))



                if 0 in mark:
                    pf_forces1 = pf_force[1] + 0.2
                else:
                    pf_forces1 = pf_force[1]

                if 1 in mark:
                    pf_forces0 = pf_force[0] + 0.2
                else:
                    pf_forces0 = pf_force[0]
                if 2 in mark:
                    pf_forces1 = pf_force[1] - 0.2
                else:
                    pf_forces1 = pf_force[1]
                if 3 in mark:
                    pf_forces0 = pf_force[0] - 0.2
                else:
                    pf_forces0 = pf_force[0]

                pf_forcess = [pf_forces0, pf_forces1, pf_force[2]]

                
                action[j, 0:3] = pf_forcess
                action[j, 3] = 0.99

                if gui and i % 10 == 0:
                    p.addUserDebugLine(
                        current_pos, 
                        current_pos + pf_force, 
                        lineColorRGB=[1,0,0], 
                        lifeTime=0.1,
                        physicsClientId=PYB_CLIENT
                    )
                    p.addUserDebugLine(
                        current_pos, 
                        target_pos, 
                        lineColorRGB=[0,1,0], 
                        lifeTime=0.1,
                        physicsClientId=PYB_CLIENT
                    )


                #print(f"Drone {j} Sensors: {[f'{d:.2f}' for d in sensor_measurements]}")
                mark = []
                for d in sensor_measurements:
                    if d < 1:
                        mark.append(sensor_measurements.index(d))
                    # dist = d
                    # print(f"sensor {j}:", dist)

                if 1 in mark or 3 in mark:
                    ko = "kanan kiri"
                else:
                    ko = 0
                if 0 in mark or 2 in mark:
                    ko1 = "maju mundur"
                else:
                    ko1 = 0

                print(f"Mark: {mark}" if mark else '')
                text_position = np.array([-10, 10, 0.5])
                text = f"Drone {j} Sensors: {[f'{d:.2f}' for d in sensor_measurements]}, Time: {timesimulation:.2f}s, altitude: {altitude:.2f}m, force: {pf_force}"
                p.addUserDebugText(text, text_position, textColorRGB=[1, 0, 0], textSize=1, lifeTime=0.1, physicsClientId=PYB_CLIENT)

            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
                if i % 100 == 0:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=15,
                        cameraYaw=0,
                        cameraPitch=-89.99,
                        cameraTargetPosition=[0,0,0],
                        physicsClientId=PYB_CLIENT
                    )
            
            for j in range(NUM_DRONES):
                logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j],
                        control=np.hstack([action[j, 0:3], np.zeros(9)])
                        )

            env.render()

            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

    finally:
        env.close()
        cv2.destroyAllWindows()
        print(f"Final Coverage: {coverage.get_coverage_percentage():.2f}%")

    logger.save_as_csv("coverage_path")
    if plot:
        logger.plot()

if __name__ == "__main__":
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