"""
Copyright (c) 2025 Lalu Muhamad Alhadad

This code was developed as part of a Master's thesis under the
Faculty of Mechanical and Aerospace Engineering,
Institut Teknologi Bandung (ITB).

Based on gym-pybullet-drones developed by the University of Cambridge,
Prorok Lab (MIT License). Modifications include control logic adaptation,
flight dynamics limiters, and PID-based trajectory tracking.

Licensed under the MIT License. See LICENSE file for details.
"""


# Modified simulation with easy waypoint and swarm parameter input
# Based on your existing PyBullet simulation code

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
from collections import deque
import cv2
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label, center_of_mass
import pandas as pd
import signal
import sys
import psutil
import gc
from gymnasium import spaces
import threading
from matplotlib.gridspec import GridSpec

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
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_DURATION_SEC = 105
DEFAULT_OUTPUT_FOLDER = 'results/Optimizer'
DEFAULT_COLAB = False

# Add this after DEFAULT_COLAB = False

# ============================================================================
# CAMERA FOLLOW CONFIGURATION
# ============================================================================
ENABLE_FOLLOW_CAMERA = True  # Set to True to enable follow camera
FOLLOW_DRONE_ID = 0          # Which drone to follow (0, 1, 2, etc.)
CAMERA_DISTANCE = 5.0        # Distance from drone
CAMERA_HEIGHT_OFFSET = 2.0   # Height above drone
CAMERA_SMOOTH_FACTOR = 0.1   # Camera smoothing (0.1 = smooth, 1.0 = instant)


class EnhancedVelocityAviary(VelocityAviary):
    """Enhanced VelocityAviary with proper yaw control and realistic speed limits"""
    
    def __init__(self, *args, **kwargs):
        # ===== EXTRACT PARAMETERS BEFORE CALLING SUPER() =====
        if 'num_drones' in kwargs:
            num_drones = kwargs['num_drones']
        elif len(args) >= 2:
            num_drones = args[1]
        else:
            num_drones = 1

        # Initialize yaw control attributes
        self.yaw_rate_limit = np.radians(30)  # REDUCED from 60 to 30 degrees per second
        self.target_yaw = np.zeros(num_drones)
        
        # Call parent initialization
        super().__init__(*args, **kwargs)
        
        # ===== POST-INITIALIZATION SETUP =====
        self.SPEED_LIMIT = 0.3 * self.MAX_SPEED_KMH * (1000/3600)
        
        print(f"[INFO] Enhanced VelocityAviary initialized:")
        print(f"   Number of drones: {self.NUM_DRONES}")
        print(f"   Speed limit: {self.SPEED_LIMIT:.2f} m/s")
        print(f"   Yaw rate limit: {np.degrees(self.yaw_rate_limit):.1f}°/s")
        
        # Resize target_yaw if needed
        if len(self.target_yaw) != self.NUM_DRONES:
            self.target_yaw = np.zeros(self.NUM_DRONES)
    
    def _actionSpace(self):
        """Enhanced action space: [vx, vy, vz, yaw_rate]"""
        act_lower_bound = np.array([
            [-1.0, -1.0, -1.0, -self.yaw_rate_limit] for i in range(self.NUM_DRONES)
        ])
        act_upper_bound = np.array([
            [1.0, 1.0, 1.0, self.yaw_rate_limit] for i in range(self.NUM_DRONES)
        ])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    def _preprocessAction(self, action):
        """SIMPLIFIED preprocessing with proper yaw control"""
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            command = action[k, :]
            
            # Extract components
            velocity_cmd = command[0:3]  # [vx, vy, vz] normalized to [-1, 1]
            yaw_rate = command[3] if len(command) > 3 else 0.0
            
            # ===== SIMPLIFIED YAW CONTROL =====
            current_yaw = state[9]  # Current yaw from state vector
            
            # Update target yaw based on yaw rate command
            if abs(yaw_rate) > 0.01:  # Only update if meaningful yaw rate
                yaw_rate_clamped = np.clip(yaw_rate, -self.yaw_rate_limit, self.yaw_rate_limit)
                self.target_yaw[k] += yaw_rate_clamped * self.CTRL_TIMESTEP
                # Normalize to [-pi, pi]
                self.target_yaw[k] = ((self.target_yaw[k] + np.pi) % (2 * np.pi)) - np.pi
            
            # Convert normalized velocity to actual velocity
            target_velocity = velocity_cmd * self.SPEED_LIMIT
            
            # Use PID control with target yaw
            try:
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=state[0:3],  # Hover at current position
                    target_rpy=np.array([0, 0, self.target_yaw[k]]),  # Use target yaw
                    target_vel=target_velocity
                )
                rpm[k,:] = temp
                
            except Exception as e:
                print(f"[ERROR] PID control failed for drone {k}: {e}")
                rpm[k,:] = np.array([self.HOVER_RPM, self.HOVER_RPM, self.HOVER_RPM, self.HOVER_RPM])
                
        return rpm
    
    def get_speed_limit(self):
        """Get current speed limit for external reference"""
        return self.SPEED_LIMIT
    
    def get_target_yaw(self, drone_id):
        """Get target yaw for specific drone"""
        if 0 <= drone_id < len(self.target_yaw):
            return self.target_yaw[drone_id]
        return 0.0
    
    def set_target_yaw(self, drone_id, yaw_angle):
        """Manually set target yaw for specific drone"""
        if 0 <= drone_id < len(self.target_yaw):
            self.target_yaw[drone_id] = yaw_angle


class DroneFollowCamera:
    def __init__(self, target_drone_id=0, distance=5.0, height_offset=2.0, smooth_factor=0.1):
        self.target_drone_id = target_drone_id
        self.distance = distance
        self.height_offset = height_offset
        self.smooth_factor = smooth_factor
        self.enabled = True  # Add this line
        
        # Camera state
        self.camera_pos = np.array([0.0, 0.0, 5.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        
        # Camera angles
        self.yaw = 0.0
        self.pitch = -30.0  # Look down at angle
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.enabled = not self.enabled
        print(f"📹 Follow camera: {'ON' if self.enabled else 'OFF'}")
        return self.enabled
        
    def update_camera(self, drone_positions, drone_velocities, pyb_client):
        """Update camera to follow target drone"""
        if not self.enabled or self.target_drone_id >= len(drone_positions):
            return
            
        target_pos = np.array(drone_positions[self.target_drone_id])
        target_vel = np.array(drone_velocities[self.target_drone_id])
        
        # Calculate desired camera position (behind and above drone)
        # Predict drone future position for smoother following
        prediction_time = 0.5
        predicted_pos = target_pos + target_vel * prediction_time
        
        # Camera follows from behind based on drone's movement direction
        if np.linalg.norm(target_vel[:2]) > 0.1:  # If drone is moving
            move_direction = target_vel[:2] / np.linalg.norm(target_vel[:2])
            # Position camera behind the movement direction
            camera_offset = -move_direction * self.distance
        else:
            # If drone is stationary, maintain current relative position
            camera_offset = np.array([-self.distance, 0])
        
        desired_camera_pos = np.array([
            predicted_pos[0] + camera_offset[0],
            predicted_pos[1] + camera_offset[1],
            predicted_pos[2] + self.height_offset
        ])
        
        # Smooth camera movement
        self.camera_pos = (self.camera_pos * (1 - self.smooth_factor) + 
                          desired_camera_pos * self.smooth_factor)
        
        # Camera always looks at the drone
        self.camera_target = target_pos
        
        # Calculate yaw to look at drone
        dx = self.camera_target[0] - self.camera_pos[0]
        dy = self.camera_target[1] - self.camera_pos[1]
        
        if abs(dx) > 0.001 or abs(dy) > 0.001:
            self.yaw = np.degrees(np.arctan2(dy, dx))
        
        # Update PyBullet camera
        p.resetDebugVisualizerCamera(
            cameraDistance=0.1,  # Small distance since we set position directly
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.camera_target.tolist(),
            physicsClientId=pyb_client
        )
        
    def switch_target_drone(self, new_drone_id, max_drones):
        """Switch to follow different drone"""
        if 0 <= new_drone_id < max_drones:
            self.target_drone_id = new_drone_id
            print(f"📹 Camera now following Drone {new_drone_id}")
            return True
        return False
        
    def get_camera_info(self):
        """Get current camera information"""
        return {
            'following_drone': self.target_drone_id,
            'camera_pos': self.camera_pos.tolist(),
            'target_pos': self.camera_target.tolist(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'enabled': self.enabled
        }

# ============================================================================
# WAYPOINT CONFIGURATION SECTION
# ============================================================================

def get_waypoint_configuration(external_waypoints=None):

    if external_waypoints is not None:
        print("🎯 Using waypoints from GUI optimization")

        if len(external_waypoints) >= 8:
            get_waypoint_configuration.building_shape = external_waypoints[6]
            get_waypoint_configuration.starting_formation = external_waypoints[7]
        return external_waypoints[:6]
    """
    🎯 EASY WAYPOINT CONFIGURATION
    
    Modify this function to set your waypoints and swarm parameters.
    You can copy waypoints from the tkinter GUI or create custom patterns.
    """
    
    # ===== SWARM PARAMETERS =====
    NUM_DRONES = 4        # Number of drones (1-8)
    BUILDING_SIZE_HEIGHT = 20      # Size of area to cover (meters)
    BUILDING_SIZE_WIDTH = 50        # Size of area to cover (meters)
    ALTITUDE = 2          # Flight altitude (meters)
    
    # ===== WAYPOINT SELECTION =====
    # Choose one of the waypoint sets below, or add your own
    
    waypoint_mode = "FOUR_DRONE"  # Change this to switch waypoint sets
    
    if waypoint_mode == "SINGLE_DRONE":
        # Single drone with optimized waypoints (from tkinter GUI)
        NUM_DRONES = 1
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
# Drone Swarm Waypoints - Fast Copy Format
# Drones: 1 | Grid: 23x7 | Formation: single_point

        WAYPOINTS = [
            # Drone 0 - 139 waypoints
            np.array([
                    [  -24.0 ,   -6.7 ,     2 ],
                    [  -23.6 ,   -8.6 ,     2 ],
                    [  -21.4 ,   -8.6 ,     2 ],
                    [  -19.3 ,   -8.6 ,     2 ],
                    [  -17.1 ,   -8.6 ,     2 ],
                    [  -15.0 ,   -8.6 ,     2 ],
                    [  -12.8 ,   -8.6 ,     2 ],
                    [  -10.7 ,   -8.6 ,     2 ],
                    [   -8.6 ,   -8.6 ,     2 ],
                    [   -6.4 ,   -8.6 ,     2 ],
                    [   -4.3 ,   -8.6 ,     2 ],
                    [   -2.1 ,   -8.6 ,     2 ],
                    [   0.0 ,   -8.6 ,     2 ],
                    [   2.1 ,   -8.6 ,     2 ],
                    [   4.3 ,   -8.6 ,     2 ],
                    [   6.4 ,   -8.6 ,     2 ],
                    [   8.6 ,   -8.6 ,     2 ],
                    [  10.7 ,   -8.6 ,     2 ],
                    [  12.8 ,   -8.6 ,     2 ],
                    [  15.0 ,   -8.6 ,     2 ],
                    [  17.1 ,   -8.6 ,     2 ],
                    [  19.3 ,   -8.6 ,     2 ],
                    [  21.4 ,   -8.6 ,     2 ],
                    [  23.6 ,   -8.6 ,     2 ],
                    [  23.6 ,   -5.1 ,     2 ],
                    [  21.4 ,   -5.1 ,     2 ],
                    [  19.3 ,   -5.1 ,     2 ],
                    [  17.1 ,   -5.1 ,     2 ],
                    [  15.0 ,   -5.1 ,     2 ],
                    [  12.8 ,   -5.1 ,     2 ],
                    [  10.7 ,   -5.1 ,     2 ],
                    [   8.6 ,   -5.1 ,     2 ],
                    [   6.4 ,   -5.1 ,     2 ],
                    [   4.3 ,   -5.1 ,     2 ],
                    [   2.1 ,   -5.1 ,     2 ],
                    [   0.0 ,   -5.1 ,     2 ],
                    [   -2.1 ,   -5.1 ,     2 ],
                    [   -4.3 ,   -5.1 ,     2 ],
                    [   -6.4 ,   -5.1 ,     2 ],
                    [   -8.6 ,   -5.1 ,     2 ],
                    [  -10.7 ,   -5.1 ,     2 ],
                    [  -12.8 ,   -5.1 ,     2 ],
                    [  -15.0 ,   -5.1 ,     2 ],
                    [  -17.1 ,   -5.1 ,     2 ],
                    [  -19.3 ,   -5.1 ,     2 ],
                    [  -21.4 ,   -5.1 ,     2 ],
                    [  -23.6 ,   -5.1 ,     2 ],
                    [  -23.6 ,   -1.7 ,     2 ],
                    [  -21.4 ,   -1.7 ,     2 ],
                    [  -19.3 ,   -1.7 ,     2 ],
                    [  -17.1 ,   -1.7 ,     2 ],
                    [  -15.0 ,   -1.7 ,     2 ],
                    [  -12.8 ,   -1.7 ,     2 ],
                    [  -10.7 ,   -1.7 ,     2 ],
                    [   -8.6 ,   -1.7 ,     2 ],
                    [   -6.4 ,   -1.7 ,     2 ],
                    [   -4.3 ,   -1.7 ,     2 ],
                    [   -2.1 ,   -1.7 ,     2 ],
                    [   0.0 ,   -1.7 ,     2 ],
                    [   2.1 ,   -1.7 ,     2 ],
                    [   4.3 ,   -1.7 ,     2 ],
                    [   6.4 ,   -1.7 ,     2 ],
                    [   8.6 ,   -1.7 ,     2 ],
                    [  10.7 ,   -1.7 ,     2 ],
                    [  12.8 ,   -1.7 ,     2 ],
                    [  15.0 ,   -1.7 ,     2 ],
                    [  17.1 ,   -1.7 ,     2 ],
                    [  19.3 ,   -1.7 ,     2 ],
                    [  21.4 ,   -1.7 ,     2 ],
                    [  23.6 ,   -1.7 ,     2 ],
                    [  23.6 ,   1.7 ,     2 ],
                    [  21.4 ,   1.7 ,     2 ],
                    [  19.3 ,   1.7 ,     2 ],
                    [  17.1 ,   1.7 ,     2 ],
                    [  15.0 ,   1.7 ,     2 ],
                    [  12.8 ,   1.7 ,     2 ],
                    [  10.7 ,   1.7 ,     2 ],
                    [   8.6 ,   1.7 ,     2 ],
                    [   6.4 ,   1.7 ,     2 ],
                    [   4.3 ,   1.7 ,     2 ],
                    [   2.1 ,   1.7 ,     2 ],
                    [   0.0 ,   1.7 ,     2 ],
                    [   -2.1 ,   1.7 ,     2 ],
                    [   -4.3 ,   1.7 ,     2 ],
                    [   -6.4 ,   1.7 ,     2 ],
                    [   -8.6 ,   1.7 ,     2 ],
                    [  -10.7 ,   1.7 ,     2 ],
                    [  -12.8 ,   1.7 ,     2 ],
                    [  -15.0 ,   1.7 ,     2 ],
                    [  -17.1 ,   1.7 ,     2 ],
                    [  -19.3 ,   1.7 ,     2 ],
                    [  -21.4 ,   1.7 ,     2 ],
                    [  -23.6 ,   1.7 ,     2 ],
                    [  -23.6 ,   5.1 ,     2 ],
                    [  -21.4 ,   5.1 ,     2 ],
                    [  -19.3 ,   5.1 ,     2 ],
                    [  -17.1 ,   5.1 ,     2 ],
                    [  -15.0 ,   5.1 ,     2 ],
                    [  -12.8 ,   5.1 ,     2 ],
                    [  -10.7 ,   5.1 ,     2 ],
                    [   -8.6 ,   5.1 ,     2 ],
                    [   -6.4 ,   5.1 ,     2 ],
                    [   -4.3 ,   5.1 ,     2 ],
                    [   -2.1 ,   5.1 ,     2 ],
                    [   0.0 ,   5.1 ,     2 ],
                    [   2.1 ,   5.1 ,     2 ],
                    [   4.3 ,   5.1 ,     2 ],
                    [   6.4 ,   5.1 ,     2 ],
                    [   8.6 ,   5.1 ,     2 ],
                    [  10.7 ,   5.1 ,     2 ],
                    [  12.8 ,   5.1 ,     2 ],
                    [  15.0 ,   5.1 ,     2 ],
                    [  17.1 ,   5.1 ,     2 ],
                    [  19.3 ,   5.1 ,     2 ],
                    [  21.4 ,   5.1 ,     2 ],
                    [  23.6 ,   5.1 ,     2 ],
                    [  23.6 ,   8.6 ,     2 ],
                    [  21.4 ,   8.6 ,     2 ],
                    [  19.3 ,   8.6 ,     2 ],
                    [  17.1 ,   8.6 ,     2 ],
                    [  15.0 ,   8.6 ,     2 ],
                    [  12.8 ,   8.6 ,     2 ],
                    [  10.7 ,   8.6 ,     2 ],
                    [   8.6 ,   8.6 ,     2 ],
                    [   6.4 ,   8.6 ,     2 ],
                    [   4.3 ,   8.6 ,     2 ],
                    [   2.1 ,   8.6 ,     2 ],
                    [   0.0 ,   8.6 ,     2 ],
                    [   -2.1 ,   8.6 ,     2 ],
                    [   -4.3 ,   8.6 ,     2 ],
                    [   -6.4 ,   8.6 ,     2 ],
                    [   -8.6 ,   8.6 ,     2 ],
                    [  -10.7 ,   8.6 ,     2 ],
                    [  -12.8 ,   8.6 ,     2 ],
                    [  -15.0 ,   8.6 ,     2 ],
                    [  -17.1 ,   8.6 ,     2 ],
                    [  -19.3 ,   8.6 ,     2 ],
                    [  -21.4 ,   8.6 ,     2 ],
                    [  -23.6 ,   8.6 ,     2 ]
            ]),
        ]
        
    elif waypoint_mode == "TWO_DRONE":
        # Two drones with coordinated coverage
        NUM_DRONES = 2
        BUILDING_SIZE_WIDTH = 33.5
        BUILDING_SIZE_HEIGHT = 33.5
# Drone Swarm Waypoints - Fast Copy Format
# Drones: 2 | Grid: 15x6 | Formation: single_point

        WAYPOINTS = [
            # Drone 0 - 46 waypoints
            np.array([
                    [  -24.0 ,   -6.7 ,     2 ],
                    [  -23.6 ,   -5.1 ,     2 ],
                    [  -23.6 ,   -8.6 ,     2 ],
                    [  -16.8 ,   -8.6 ,     2 ],
                    [  -20.2 ,   -5.1 ,     2 ],
                    [  -16.8 ,   -1.7 ,     2 ],
                    [  -13.5 ,   -5.1 ,     2 ],
                    [  -10.1 ,   -8.6 ,     2 ],
                    [  -10.1 ,   -1.7 ,     2 ],
                    [  -13.5 ,   1.7 ,     2 ],
                    [  -10.1 ,   5.1 ,     2 ],
                    [   -3.4 ,   5.1 ,     2 ],
                    [   -6.7 ,   1.7 ,     2 ],
                    [   -3.4 ,   -1.7 ,     2 ],
                    [   -6.7 ,   -5.1 ,     2 ],
                    [   -3.4 ,   -8.6 ,     2 ],
                    [   3.4 ,   -5.1 ,     2 ],
                    [   6.7 ,   -8.6 ,     2 ],
                    [   6.7 ,   -1.7 ,     2 ],
                    [  10.1 ,   -5.1 ,     2 ],
                    [  13.5 ,   -8.6 ,     2 ],
                    [  16.8 ,   -5.1 ,     2 ],
                    [  20.2 ,   -8.6 ,     2 ],
                    [  23.6 ,   -5.1 ,     2 ],
                    [  20.2 ,   -1.7 ,     2 ],
                    [  23.6 ,   1.7 ,     2 ],
                    [  23.6 ,   8.6 ,     2 ],
                    [  16.8 ,   8.6 ,     2 ],
                    [  13.5 ,   5.1 ,     2 ],
                    [  20.2 ,   5.1 ,     2 ],
                    [  16.8 ,   1.7 ,     2 ],
                    [  13.5 ,   -1.7 ,     2 ],
                    [  10.1 ,   1.7 ,     2 ],
                    [  10.1 ,   8.6 ,     2 ],
                    [   6.7 ,   5.1 ,     2 ],
                    [   0.0 ,   1.7 ,     2 ],
                    [   0.0 ,   8.6 ,     2 ],
                    [   -6.7 ,   8.6 ,     2 ],
                    [  -13.5 ,   8.6 ,     2 ],
                    [  -16.8 ,   5.1 ,     2 ],
                    [  -20.2 ,   8.6 ,     2 ],
                    [  -23.6 ,   8.6 ,     2 ],
                    [  -23.6 ,   5.1 ,     2 ],
                    [  -23.6 ,   1.7 ,     2 ],
                    [  -20.2 ,   1.7 ,     2 ],
                    [  -23.6 ,   -1.7 ,     2 ]
            ]),

            # Drone 1 - 46 waypoints
            np.array([
                    [  -21.6 ,   -6.7 ,     2 ],
                    [  -20.2 ,   -8.6 ,     2 ],
                    [  -16.8 ,   -5.1 ,     2 ],
                    [  -20.2 ,   -1.7 ,     2 ],
                    [  -16.8 ,   1.7 ,     2 ],
                    [  -20.2 ,   5.1 ,     2 ],
                    [  -16.8 ,   8.6 ,     2 ],
                    [  -13.5 ,   5.1 ,     2 ],
                    [  -10.1 ,   1.7 ,     2 ],
                    [  -13.5 ,   -1.7 ,     2 ],
                    [  -10.1 ,   -5.1 ,     2 ],
                    [  -13.5 ,   -8.6 ,     2 ],
                    [   -6.7 ,   -8.6 ,     2 ],
                    [   -3.4 ,   -5.1 ,     2 ],
                    [   -6.7 ,   -1.7 ,     2 ],
                    [   -3.4 ,   1.7 ,     2 ],
                    [   0.0 ,   5.1 ,     2 ],
                    [   3.4 ,   5.1 ,     2 ],
                    [   6.7 ,   1.7 ,     2 ],
                    [   3.4 ,   1.7 ,     2 ],
                    [   3.4 ,   -1.7 ,     2 ],
                    [   0.0 ,   -1.7 ,     2 ],
                    [   0.0 ,   -5.1 ,     2 ],
                    [   0.0 ,   -8.6 ,     2 ],
                    [   3.4 ,   -8.6 ,     2 ],
                    [   6.7 ,   -5.1 ,     2 ],
                    [  10.1 ,   -1.7 ,     2 ],
                    [  10.1 ,   -8.6 ,     2 ],
                    [  16.8 ,   -8.6 ,     2 ],
                    [  23.6 ,   -8.6 ,     2 ],
                    [  23.6 ,   -1.7 ,     2 ],
                    [  20.2 ,   -5.1 ,     2 ],
                    [  13.5 ,   -5.1 ,     2 ],
                    [  16.8 ,   -1.7 ,     2 ],
                    [  20.2 ,   1.7 ,     2 ],
                    [  23.6 ,   5.1 ,     2 ],
                    [  20.2 ,   8.6 ,     2 ],
                    [  13.5 ,   8.6 ,     2 ],
                    [  16.8 ,   5.1 ,     2 ],
                    [  13.5 ,   1.7 ,     2 ],
                    [  10.1 ,   5.1 ,     2 ],
                    [   6.7 ,   8.6 ,     2 ],
                    [   3.4 ,   8.6 ,     2 ],
                    [   -3.4 ,   8.6 ,     2 ],
                    [   -6.7 ,   5.1 ,     2 ],
                    [  -10.1 ,   8.6 ,     2 ]
            ]),
        ]

    elif waypoint_mode == "FOUR_DRONE":
        # Four drones with quadrant division
        NUM_DRONES = 4
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
# Drone Swarm Waypoints - Fast Copy Format
# Drones: 4 | Grid: 15x8 | Formation: corner

        WAYPOINTS = [
            # Drone 0 - 17 waypoints
            np.array([
                    [  -22.6 ,   -7.6 ,     2 ],
                    [  -23.6 ,   -8.6 ,     2 ],
                    [  -20.2 ,   -8.6 ,     2 ],
                    [  -16.8 ,   -8.6 ,     2 ],
                    [  -13.5 ,   -8.6 ,     2 ],
                    [  -10.1 ,   -8.6 ,     2 ],
                    [   -6.7 ,   -8.6 ,     2 ],
                    [   -3.4 ,   -8.6 ,     2 ],
                    [   0.0 ,   -8.6 ,     2 ],
                    [   -3.4 ,   -5.1 ,     2 ],
                    [   -6.7 ,   -5.1 ,     2 ],
                    [  -10.1 ,   -5.1 ,     2 ],
                    [  -13.5 ,   -5.1 ,     2 ],
                    [  -16.8 ,   -5.1 ,     2 ],
                    [  -20.2 ,   -5.1 ,     2 ],
                    [  -23.6 ,   -5.1 ,     2 ],
                    [  -23.6 ,   -1.7 ,     2 ]
            ]),

            # Drone 1 - 17 waypoints
            np.array([
                    [  22.6 ,   -7.6 ,     2 ],
                    [  23.6 ,   -8.6 ,     2 ],
                    [  20.2 ,   -8.6 ,     2 ],
                    [  16.8 ,   -8.6 ,     2 ],
                    [  13.5 ,   -8.6 ,     2 ],
                    [  10.1 ,   -8.6 ,     2 ],
                    [   6.7 ,   -8.6 ,     2 ],
                    [   3.4 ,   -8.6 ,     2 ],
                    [   3.4 ,   -5.1 ,     2 ],
                    [   0.0 ,   -5.1 ,     2 ],
                    [   0.0 ,   -1.7 ,     2 ],
                    [   -3.4 ,   -1.7 ,     2 ],
                    [   -6.7 ,   -1.7 ,     2 ],
                    [  -10.1 ,   -1.7 ,     2 ],
                    [  -13.5 ,   -1.7 ,     2 ],
                    [  -16.8 ,   -1.7 ,     2 ],
                    [  -20.2 ,   -1.7 ,     2 ]
            ]),

            # Drone 2 - 30 waypoints
            np.array([
                    [  22.6 ,   7.6 ,     2 ],
                    [  23.6 ,   8.6 ,     2 ],
                    [  20.2 ,   8.6 ,     2 ],
                    [  16.8 ,   8.6 ,     2 ],
                    [  13.5 ,   8.6 ,     2 ],
                    [  10.1 ,   8.6 ,     2 ],
                    [   6.7 ,   8.6 ,     2 ],
                    [   3.4 ,   8.6 ,     2 ],
                    [   0.0 ,   8.6 ,     2 ],
                    [   0.0 ,   5.1 ,     2 ],
                    [   3.4 ,   5.1 ,     2 ],
                    [   6.7 ,   5.1 ,     2 ],
                    [  10.1 ,   5.1 ,     2 ],
                    [  13.5 ,   5.1 ,     2 ],
                    [  16.8 ,   5.1 ,     2 ],
                    [  20.2 ,   5.1 ,     2 ],
                    [  23.6 ,   5.1 ,     2 ],
                    [  23.6 ,   1.7 ,     2 ],
                    [  20.2 ,   1.7 ,     2 ],
                    [  16.8 ,   1.7 ,     2 ],
                    [  13.5 ,   1.7 ,     2 ],
                    [  10.1 ,   1.7 ,     2 ],
                    [   6.7 ,   1.7 ,     2 ],
                    [   3.4 ,   1.7 ,     2 ],
                    [   3.4 ,   -1.7 ,     2 ],
                    [   6.7 ,   -1.7 ,     2 ],
                    [  10.1 ,   -1.7 ,     2 ],
                    [  13.5 ,   -1.7 ,     2 ],
                    [  16.8 ,   -1.7 ,     2 ],
                    [  20.2 ,   -1.7 ,     2 ]
            ]),

            # Drone 3 - 30 waypoints
            np.array([
                    [  -22.6 ,   7.6 ,     2 ],
                    [  -23.6 ,   8.6 ,     2 ],
                    [  -20.2 ,   8.6 ,     2 ],
                    [  -16.8 ,   8.6 ,     2 ],
                    [  -13.5 ,   8.6 ,     2 ],
                    [  -10.1 ,   8.6 ,     2 ],
                    [   -6.7 ,   8.6 ,     2 ],
                    [   -3.4 ,   8.6 ,     2 ],
                    [   -3.4 ,   5.1 ,     2 ],
                    [   -6.7 ,   5.1 ,     2 ],
                    [  -10.1 ,   5.1 ,     2 ],
                    [  -13.5 ,   5.1 ,     2 ],
                    [  -16.8 ,   5.1 ,     2 ],
                    [  -20.2 ,   5.1 ,     2 ],
                    [  -23.6 ,   5.1 ,     2 ],
                    [  -23.6 ,   1.7 ,     2 ],
                    [  -20.2 ,   1.7 ,     2 ],
                    [  -16.8 ,   1.7 ,     2 ],
                    [  -13.5 ,   1.7 ,     2 ],
                    [  -10.1 ,   1.7 ,     2 ],
                    [   -6.7 ,   1.7 ,     2 ],
                    [   -3.4 ,   1.7 ,     2 ],
                    [   0.0 ,   1.7 ,     2 ],
                    [   6.7 ,   -5.1 ,     2 ],
                    [  10.1 ,   -5.1 ,     2 ],
                    [  13.5 ,   -5.1 ,     2 ],
                    [  16.8 ,   -5.1 ,     2 ],
                    [  20.2 ,   -5.1 ,     2 ],
                    [  23.6 ,   -5.1 ,     2 ],
                    [  23.6 ,   -1.7 ,     2 ]
            ]),
        ]

    elif waypoint_mode == "FIVE_DRONE":
        # Four drones with quadrant division
        NUM_DRONES = 5
        BUILDING_SIZE_WIDTH = 33.5
        BUILDING_SIZE_HEIGHT = 33.5
# Drone Swarm Waypoints - Fast Copy Format
# Drones: 5 | Grid: 16x15 | Formation: corner

        WAYPOINTS = [
            # Drone 0 - 14 waypoints
            np.array([
                    [  13.8 ,   -1.2 ,     2 ],
                    [  13.0 ,   0.0 ,     2 ],
                    [  10.4 ,   0.0 ,     2 ],
                    [   7.8 ,   0.0 ,     2 ],
                    [   5.2 ,   0.0 ,     2 ],
                    [   2.6 ,   0.0 ,     2 ],
                    [   1.8 ,   1.8 ,     2 ],
                    [   0.0 ,   0.0 ,     2 ],
                    [   -0.0 ,   -5.2 ,     2 ],
                    [   -0.0 ,   -7.8 ,     2 ],
                    [   -0.0 ,  -10.4 ,     2 ],
                    [   2.8 ,  -12.7 ,     2 ],
                    [   -0.0 ,  -13.0 ,     2 ],
                    [   -2.8 ,  -12.7 ,     2 ]
            ]),

            # Drone 1 - 13 waypoints
            np.array([
                    [   9.4 ,   9.4 ,     2 ],
                    [   8.1 ,  10.1 ,     2 ],
                    [  10.1 ,   8.1 ,     2 ],
                    [   7.4 ,   7.4 ,     2 ],
                    [   4.4 ,   6.4 ,     2 ],
                    [   3.7 ,   3.7 ,     2 ],
                    [   6.4 ,   4.4 ,     2 ],
                    [  10.0 ,   2.8 ,     2 ],
                    [  12.7 ,   2.8 ,     2 ],
                    [  12.7 ,   -2.8 ,     2 ],
                    [   9.1 ,   -5.1 ,     2 ],
                    [   6.4 ,   -4.4 ,     2 ],
                    [  10.1 ,   -8.1 ,     2 ]
            ]),

            # Drone 2 - 18 waypoints
            np.array([
                    [   -1.2 ,  13.8 ,     2 ],
                    [   0.0 ,  13.0 ,     2 ],
                    [   0.0 ,  10.4 ,     2 ],
                    [   0.0 ,   7.8 ,     2 ],
                    [   0.0 ,   5.2 ,     2 ],
                    [   0.0 ,   2.6 ,     2 ],
                    [   -1.8 ,   1.8 ,     2 ],
                    [   -3.7 ,   -3.7 ,     2 ],
                    [   -6.4 ,   -4.4 ,     2 ],
                    [   -4.4 ,   -6.4 ,     2 ],
                    [   -5.1 ,   -9.1 ,     2 ],
                    [   -7.4 ,   -7.4 ,     2 ],
                    [   -8.1 ,  -10.1 ,     2 ],
                    [  -10.1 ,   -8.1 ,     2 ],
                    [  -10.0 ,   -2.8 ,     2 ],
                    [  -12.7 ,   -2.8 ,     2 ],
                    [  -12.7 ,   2.8 ,     2 ],
                    [  -10.0 ,   2.8 ,     2 ]
            ]),

            # Drone 3 - 11 waypoints
            np.array([
                    [  -11.9 ,   9.4 ,     2 ],
                    [  -10.1 ,   8.1 ,     2 ],
                    [   -8.1 ,  10.1 ,     2 ],
                    [   -7.4 ,   7.4 ,     2 ],
                    [   -6.4 ,   4.4 ,     2 ],
                    [   -3.7 ,   3.7 ,     2 ],
                    [   -4.4 ,   6.4 ,     2 ],
                    [   -2.8 ,  10.0 ,     2 ],
                    [   -2.8 ,  12.7 ,     2 ],
                    [   2.8 ,  12.7 ,     2 ],
                    [   2.8 ,  10.0 ,     2 ]
            ]),

            # Drone 4 - 14 waypoints
            np.array([
                    [  -13.8 ,   -1.2 ,     2 ],
                    [  -13.0 ,   0.0 ,     2 ],
                    [  -10.4 ,   0.0 ,     2 ],
                    [   -7.8 ,   0.0 ,     2 ],
                    [   -5.2 ,   0.0 ,     2 ],
                    [   -2.6 ,   0.0 ,     2 ],
                    [   -1.8 ,   -1.8 ,     2 ],
                    [   -0.0 ,   -2.6 ,     2 ],
                    [   1.8 ,   -1.8 ,     2 ],
                    [   3.7 ,   -3.7 ,     2 ],
                    [   4.4 ,   -6.4 ,     2 ],
                    [   5.1 ,   -9.1 ,     2 ],
                    [   7.4 ,   -7.4 ,     2 ],
                    [   8.1 ,  -10.1 ,     2 ]
            ]),
        ]
        
    elif waypoint_mode == "HEXAGON_PATTERN":
        # Single drone hexagonal pattern
        NUM_DRONES = 1
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
        WAYPOINTS = [np.array([
            [    0.0,     0.0,   ALTITUDE],  # Center
            [    8.0,     0.0,   ALTITUDE],  # Right
            [    4.0,     6.9,   ALTITUDE],  # Top-right
            [   -4.0,     6.9,   ALTITUDE],  # Top-left
            [   -8.0,     0.0,   ALTITUDE],  # Left
            [   -4.0,    -6.9,   ALTITUDE],  # Bottom-left
            [    4.0,    -6.9,   ALTITUDE],  # Bottom-right
            [    0.0,     0.0,   ALTITUDE],  # Return to center
        ])]
        
    elif waypoint_mode == "CIRCULAR_PATTERN":
        # Single drone circular pattern
        NUM_DRONES = 1
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
        radius = 7
        waypoints = []
        # Generate circular waypoints
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            waypoints.append([x, y, ALTITUDE])
        waypoints.append([0, 0, ALTITUDE])  # Return to center
        WAYPOINTS = [np.array(waypoints)]
        
    elif waypoint_mode == "CUSTOM":
        # 🎯 ADD YOUR CUSTOM WAYPOINTS HERE
        # Copy waypoints from tkinter GUI or create your own pattern
        NUM_DRONES = 1  # Set your drone count
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
        WAYPOINTS = [np.array([
            # Add your waypoints here in format: [x, y, altitude]
            [0.0, 0.0, ALTITUDE],
            # Add more waypoints...
        ])]
        
    else:
        # Default fallback
        NUM_DRONES = 1
        BUILDING_SIZE_WIDTH = 50
        BUILDING_SIZE_HEIGHT = 20
        WAYPOINTS = [np.array([[0.0, 0.0, ALTITUDE]])]
    
    # ===== INITIAL POSITIONS =====
    # Generate initial positions based on first waypoint of each drone
    INIT_XYZS = []
    for i, drone_waypoints in enumerate(WAYPOINTS):
        if len(drone_waypoints) > 0:
            first_wp = drone_waypoints[0]
            INIT_XYZS.append([first_wp[0], first_wp[1], 0.1])  # Start slightly above ground
        else:
            INIT_XYZS.append([0.0, 0.0, 0.1])  # Default position
    
    INIT_XYZS = np.array(INIT_XYZS)
    
    return NUM_DRONES, BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT, WAYPOINTS, INIT_XYZS, ALTITUDE

# ============================================================================
# HELPER FUNCTIONS FOR WAYPOINT PATTERNS
# ============================================================================

def generate_grid_pattern(width, height, spacing_x, spacing_y, altitude):
    """Generate a grid pattern of waypoints"""
    waypoints = []
    
    # Generate grid points
    x_points = np.arange(-width/2, width/2 + spacing_x, spacing_x)
    y_points = np.arange(-height/2, height/2 + spacing_y, spacing_y)
    
    # Alternate direction for efficient coverage (lawnmower pattern)
    for i, y in enumerate(y_points):
        if i % 2 == 0:
            # Left to right
            for x in x_points:
                waypoints.append([x, y, altitude])
        else:
            # Right to left
            for x in reversed(x_points):
                waypoints.append([x, y, altitude])
    
    return waypoints

def create_permanent_waypoint_arrows(waypoint_managers, pyb_client):
    """Create permanent arrow visualization for all waypoints at simulation start"""
    
    arrow_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    arrow_bodies = []
    
    print("🏹 Creating permanent waypoint arrows...")
    
    for j, waypoint_manager in enumerate(waypoint_managers):
        drone_arrows = []
        arrow_color = arrow_colors[j % len(arrow_colors)]
        
        if hasattr(waypoint_manager, 'waypoints') and len(waypoint_manager.waypoints) > 1:
            print(f"   Drone {j}: Creating arrows for {len(waypoint_manager.waypoints)-1} waypoint segments")
            
            # Create arrows for each waypoint segment
            for i in range(len(waypoint_manager.waypoints) - 1):
                start_wp = waypoint_manager.waypoints[i]
                end_wp = waypoint_manager.waypoints[i + 1]
                
                # Calculate arrow direction and position
                direction = np.array([
                    end_wp[0] - start_wp[0],
                    end_wp[1] - start_wp[1],
                    0
                ])
                
                direction_length = np.linalg.norm(direction[:2])
                
                if direction_length > 0.1:  # Only create arrow if significant distance
                    # Normalize direction
                    direction_norm = direction / direction_length
                    
                    # Calculate arrow shaft length (70% of segment length, max 4m)
                    arrow_length = min(4.0, direction_length * 0.7)
                    
                    # Arrow positioning - start from waypoint, point toward next
                    arrow_start = np.array([start_wp[0], start_wp[1], start_wp[2] + 0.5])
                    arrow_direction = direction_norm * arrow_length
                    arrow_end = arrow_start + arrow_direction
                    
                    # Create arrow shaft as permanent line
                    shaft_id = p.addUserDebugLine(
                        arrow_start, arrow_end,
                        lineColorRGB=arrow_color,
                        lineWidth=4,
                        lifeTime=0,  # Permanent (lifeTime=0)
                        physicsClientId=pyb_client
                    )
                    
                    # Create arrowhead (two lines forming arrow tip)
                    if arrow_length > 0.8:
                        # Calculate arrowhead geometry
                        perp_vector = np.array([-direction_norm[1], direction_norm[0], 0]) * arrow_length * 0.2
                        back_vector = -direction_norm * arrow_length * 0.3
                        
                        # Left arrowhead line
                        arrowhead_left = arrow_end + back_vector + perp_vector
                        left_id = p.addUserDebugLine(
                            arrow_end, arrowhead_left,
                            lineColorRGB=arrow_color,
                            lineWidth=3,
                            lifeTime=0,  # Permanent
                            physicsClientId=pyb_client
                        )
                        
                        # Right arrowhead line
                        arrowhead_right = arrow_end + back_vector - perp_vector
                        right_id = p.addUserDebugLine(
                            arrow_end, arrowhead_right,
                            lineColorRGB=arrow_color,
                            lineWidth=3,
                            lifeTime=0,  # Permanent
                            physicsClientId=pyb_client
                        )
                        
                        # Store arrow components
                        arrow_data = {
                            'shaft_id': shaft_id,
                            'left_id': left_id,
                            'right_id': right_id,
                            'start_pos': arrow_start,
                            'end_pos': arrow_end,
                            'waypoint_index': i,
                            'color': arrow_color
                        }
                    else:
                        # Simple arrow without arrowhead for short segments
                        arrow_data = {
                            'shaft_id': shaft_id,
                            'left_id': None,
                            'right_id': None,
                            'start_pos': arrow_start,
                            'end_pos': arrow_end,
                            'waypoint_index': i,
                            'color': arrow_color
                        }
                    
                    drone_arrows.append(arrow_data)
        
        arrow_bodies.append(drone_arrows)
    
    print(f"✅ Created permanent arrows for {len(waypoint_managers)} drones")
    return arrow_bodies

def create_waypoint_labels(waypoint_managers, pyb_client):
    """Create permanent waypoint number labels"""
    
    arrow_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    label_ids = []
    
    print("🏷️ Creating permanent waypoint labels...")
    
    for j, waypoint_manager in enumerate(waypoint_managers):
        drone_labels = []
        arrow_color = arrow_colors[j % len(arrow_colors)]
        
        if hasattr(waypoint_manager, 'waypoints') and len(waypoint_manager.waypoints) > 0:
            # Create labels for each waypoint
            for i, waypoint in enumerate(waypoint_manager.waypoints):
                if i == 0:
                    # Start position
                    label_text = f"D{j}-START"
                    text_color = [c * 0.8 for c in arrow_color]  # Slightly dimmer
                else:
                    # Regular waypoint
                    label_text = f"D{j}-WP{i}"
                    text_color = arrow_color
                
                label_id = p.addUserDebugText(
                    label_text,
                    [waypoint[0], waypoint[1], waypoint[2] + 1.2],
                    textColorRGB=text_color,
                    textSize=0.01,
                    lifeTime=0,  # Permanent
                    physicsClientId=pyb_client
                )
                
                drone_labels.append({
                    'label_id': label_id,
                    'waypoint_index': i,
                    'position': waypoint,
                    'text': label_text
                })
        
        label_ids.append(drone_labels)
    
    print(f"✅ Created permanent labels for {len(waypoint_managers)} drones")
    return label_ids

def create_enhanced_waypoint_visualization(waypoint_managers, pyb_client):
    """Create comprehensive permanent waypoint visualization"""
    
    print("🎯 Initializing comprehensive waypoint visualization...")
    
    # Create permanent arrows
    arrow_bodies = create_permanent_waypoint_arrows(waypoint_managers, pyb_client)
    
    # Create permanent labels
    label_ids = create_waypoint_labels(waypoint_managers, pyb_client)
    
    # Create waypoint markers (small spheres at each waypoint)
    waypoint_markers = create_waypoint_markers(waypoint_managers, pyb_client)
    
    visualization_data = {
        'arrows': arrow_bodies,
        'labels': label_ids,
        'markers': waypoint_markers
    }
    
    print("✅ Comprehensive waypoint visualization initialized")
    return visualization_data

def create_waypoint_markers(waypoint_managers, pyb_client):
    """Create small spherical markers at each waypoint"""
    
    arrow_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    marker_bodies = []
    
    for j, waypoint_manager in enumerate(waypoint_managers):
        drone_markers = []
        arrow_color = arrow_colors[j % len(arrow_colors)]
        
        if hasattr(waypoint_manager, 'waypoints') and len(waypoint_manager.waypoints) > 0:
            for i, waypoint in enumerate(waypoint_manager.waypoints):
                # Different marker for start vs waypoints
                if i == 0:
                    # Start position - larger green sphere
                    marker_color = [0, 1, 0, 0.7]  # Green with transparency
                    marker_radius = 0.3
                else:
                    # Regular waypoint - smaller colored sphere
                    marker_color = [arrow_color[0], arrow_color[1], arrow_color[2], 0.6]
                    marker_radius = 0.2
                
                # Create visual sphere
                marker_visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=marker_radius,
                    rgbaColor=marker_color,
                    physicsClientId=pyb_client
                )
                
                # Create marker body
                marker_body = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=marker_visual,
                    basePosition=[waypoint[0], waypoint[1], waypoint[2] + 0.1],
                    physicsClientId=pyb_client
                )
                
                drone_markers.append({
                    'body_id': marker_body,
                    'waypoint_index': i,
                    'position': waypoint,
                    'radius': marker_radius
                })
        
        marker_bodies.append(drone_markers)
    
    return marker_bodies

def add_drone_markers(pyb_client, drone_positions, colors=None, marker_size=0.4):
    """
    Add visible markers to represent drones in the simulation.
    
    Args:
        pyb_client: PyBullet client ID
        drone_positions: List of drone positions [x, y, z]
        colors: List of RGB colors for each drone
        marker_size: Size of the marker
    
    Returns:
        List of marker IDs
    """
    marker_ids = []
    
    if colors is None:
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    
    for i, pos in enumerate(drone_positions):
        color = colors[i % len(colors)]
        
        # Create a cross marker with perpendicular lines
        x, y, z = pos
        
        # Horizontal line (X-axis)
        line_x = p.addUserDebugLine(
            [x - marker_size/2, y, z],
            [x + marker_size/2, y, z],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0.1,  # Short lifetime as we'll update it frequently
            physicsClientId=pyb_client
        )
        
        # Vertical line (Y-axis)
        line_y = p.addUserDebugLine(
            [x, y - marker_size/2, z],
            [x, y + marker_size/2, z],
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0.1,
            physicsClientId=pyb_client
        )
        
        # Add drone number with larger text for visibility but reduced complexity
        text_id = p.addUserDebugText(
            f"D{i}",
            [x, y, z + marker_size/2],
            textColorRGB=color,
            textSize=1.5,
            lifeTime=0.1,
            physicsClientId=pyb_client
        )
        
        marker_ids.append((line_x, line_y, text_id))
    
    return marker_ids

def generate_perimeter_pattern(width, height, altitude, spacing=2):
    """Generate a perimeter following pattern"""
    waypoints = []
    
    # Bottom edge (left to right)
    for x in np.arange(-width/2, width/2 + spacing, spacing):
        waypoints.append([x, -height/2, altitude])
    
    # Right edge (bottom to top)
    for y in np.arange(-height/2 + spacing, height/2 + spacing, spacing):
        waypoints.append([width/2, y, altitude])
    
    # Top edge (right to left)
    for x in np.arange(width/2 - spacing, -width/2 - spacing, -spacing):
        waypoints.append([x, height/2, altitude])
    
    # Left edge (top to bottom)
    for y in np.arange(height/2 - spacing, -height/2 - spacing, -spacing):
        waypoints.append([-width/2, y, altitude])
    
    return waypoints

# ============================================================================
# YOUR EXISTING CLASSES (Unchanged)
# ============================================================================

def create_obstacles(pyb_client, waypoints, num_obstacles=10, size=0.01):
    """Create obstacles randomly placed near the waypoint path"""
    obstacle_ids = []
    
    # Handle multiple drone waypoints
    all_waypoints = []
    if isinstance(waypoints[0], np.ndarray) and len(waypoints[0].shape) == 2:
        # Multiple drones
        for drone_waypoints in waypoints:
            all_waypoints.extend(drone_waypoints.tolist())
    else:
        # Single drone or already flattened
        all_waypoints = waypoints[0].tolist() if len(waypoints) == 1 else waypoints
    
    if len(all_waypoints) < 2:
        return obstacle_ids
    
    for _ in range(num_obstacles):
        seg_idx = random.randint(0, len(all_waypoints)-2)
        start = np.array(all_waypoints[seg_idx])
        end = np.array(all_waypoints[seg_idx+1])
        
        t = random.uniform(0.2, 0.8)
        pos = start + t * (end - start)
        
        pos[0] += random.uniform(-1, 1)
        pos[1] += random.uniform(-1, 1)
        pos[2] = 3.1  # Place on ground
        
        obstacle_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.01],
            physicsClientId=pyb_client
        )
        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.01],
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

def create_building_boundary_visualization(pyb_client, building_shape, building_width, building_height):
    """Create building boundary visualization in PyBullet"""
    if building_shape == 'circle':
        # Create circular boundary markers
        radius = min(building_width, building_height) / 2
        num_markers = 32
        for i in range(num_markers):
            angle = 2 * math.pi * i / num_markers
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            p.addUserDebugLine([x, y, 0], [x, y, 2], lineColorRGB=[0.5, 0.5, 0.5], 
                             lineWidth=2, lifeTime=0, physicsClientId=pyb_client)
    
    elif building_shape == 'L_shape':
        # Create L-shape boundary
        size = min(building_width, building_height) / 2
        arm_width = size / 2
        # Draw L-shape outline
        l_points = [
            [-size, -size], [0, -size], [0, -size + arm_width],
            [-size + arm_width, -size + arm_width], [-size + arm_width, size],
            [-size, size], [-size, -size]
        ]
        for i in range(len(l_points) - 1):
            p.addUserDebugLine([l_points[i][0], l_points[i][1], 0],
                             [l_points[i+1][0], l_points[i+1][1], 0],
                             lineColorRGB=[0.5, 0.5, 0.5], lineWidth=3, lifeTime=0,
                             physicsClientId=pyb_client)
    
    # Add similar boundary visualizations for U_shape, T_shape, plus_shape
    else:
        # Default rectangular boundary
        corners = [
            [-building_width/2, -building_height/2, 0],
            [building_width/2, -building_height/2, 0],
            [building_width/2, building_height/2, 0],
            [-building_width/2, building_height/2, 0]
        ]
        for i in range(len(corners)):
            start = corners[i]
            end = corners[(i+1) % len(corners)]
            p.addUserDebugLine(start, end, lineColorRGB=[0.5, 0.5, 0.5], 
                             lineWidth=4, lifeTime=0, physicsClientId=pyb_client)

class PIDController:
    def __init__(self, kp=1.5, ki=0.1, kd=0.2, max_output=None, min_output=None):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        
        self.max_output = max_output
        self.min_output = min_output
        
        # PID state variables
        self.previous_error = 0.0
        self.integral = 0.0
        #self.last_time = time.time()
        
    def update(self, setpoint, current_value, dt=None, simulation_timestep=None):
        """Update PID controller and return control output"""
        
        if simulation_timestep is None:
            simulation_timestep = 1/25  # Default to 25Hz
        
        dt = simulation_timestep
        
        # Avoid division by zero
        if dt <= 0.0:
            dt = 1/25  # Fallback to default timestep
            
        # Calculate error
        error = setpoint - current_value
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term (with windup protection)
        self.integral += error * dt
        # Limit integral to prevent windup
        if self.max_output is not None:
            max_integral = abs(self.max_output / self.ki) if self.ki != 0 else float('inf')
            self.integral = max(-max_integral, min(max_integral, self.integral))
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Calculate total output
        output = proportional + integral + derivative
        
        # Apply output limits
        if self.max_output is not None:
            output = min(self.max_output, output)
        if self.min_output is not None:
            output = max(self.min_output, output)
            
        # Update state for next iteration
        self.previous_error = error
        #self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.previous_error = 0.0
        self.integral = 0.0
        #self.last_time = time.time()

class DroneState:
    def __init__(self, target_altitude):
        self.state = "TAKEOFF"  # TAKEOFF, MISSION, RTL, LANDING
        self.target_altitude = target_altitude
        self.takeoff_complete = False
        self.altitude_tolerance = 0.1  # meters
        
    def update_takeoff_status(self, current_altitude):
        if abs(current_altitude - self.target_altitude) < self.altitude_tolerance:
            if not self.takeoff_complete:
                self.takeoff_complete = True
                self.state = "MISSION"
                return True  # Just completed takeoff
        return False

class ImprovedPotentialFieldController:
    def __init__(self, drone_id, drone_pos, waypoint, max_speed=5.0, max_force=1.0):
        self.drone_id = drone_id
        self.max_speed = max_speed
        self.max_force = max_force
        
        # Enhanced parameters
        self.base_attraction_gain = 1.0
        self.base_repulsion_gain = 20.0
        self.dynamic_repulsion_range = 0.3
        self.waypoint_threshold = 0.45
        self.slowdown_radius = 0.5

        self.last_target_speed = 0.0
        
        # ADD THESE PID CONTROLLERS:
        # Speed PID controller for smooth speed transitions
        self.speed_pid = PIDController(
            kp=2.0,      # Proportional gain for speed control
            ki=0.05,      # Integral gain for steady-state accuracy
            kd=0.1,      # Derivative gain for smooth transitions
            max_output=max_speed,
            min_output=0.2  # Minimum speed to prevent stopping
        )
        
        # Distance PID controller for waypoint approach
        self.distance_pid = PIDController(
            kp=0.8,      # Proportional gain for distance control
            ki=0.02,     # Small integral to eliminate steady-state error
            kd=0.1,     # Derivative for smooth approach
            max_output=1.0,
            min_output=0.0
        )
        
        # Obstacle avoidance PID controller
        self.obstacle_pid = PIDController(
            kp=1.5,      # Strong response to obstacles
            ki=0.0,      # No integral for obstacle avoidance
            kd=0.2,      # Derivative for smooth avoidance
            max_output=1.0,
            min_output=0.3
        )
        
        # Velocity smoothing
        self.velocity_smoothing = 0.5
        self.previous_velocity = np.zeros(3)
        
        # Stuck detection - only for walls
        self.stuck_threshold = 0.02
        self.stuck_counter = 0
        self.max_stuck_time = 150
        self.previous_pos = None
        self.position_history = []
        self.history_length = 30
        
        # Wall-specific detection parameters
        self.wall_detection_threshold = 2.0  # Minimum obstacle size to be considered a wall
        self.wall_surround_threshold = 5     # Number of sensors that must detect wall
        self.wall_proximity_threshold = 1.2  # Distance considered "near wall"
        
        # Waypoint management
        self.waypoint_stuck_counter = 0
        self.max_waypoint_stuck_time = 240
        self.target_waypoint = None
        self.waypoint_start_time = None
        
        # Emergency maneuvers
        self.emergency_maneuver_active = False
        self.emergency_timer = 0
        self.emergency_direction = np.zeros(3)
        self.emergency_type = None
        
        # Wall following
        self.wall_follow_active = False
        self.wall_follow_timer = 0
        self.wall_follow_direction = 1
        
    def detect_wall_vs_obstacle(self, drone_pos, sensor_data, pyb_client):
        """
        Distinguish between walls (large static obstacles) and small obstacles
        Returns: (is_near_wall, wall_info)
        """
        wall_info = {
            'is_wall': False,
            'wall_coverage': 0.0,
            'closest_wall_distance': float('inf'),
            'obstacle_type': 'none'
        }
        
        distances = sensor_data.get('distances', [])
        if not distances:
            return False, wall_info
        
        # Analyze sensor readings to detect wall patterns
        close_sensors = []
        wall_distances = []
        
        # Get detailed obstacle information
        for i, distance in enumerate(distances):
            if distance < self.wall_proximity_threshold:
                close_sensors.append(i)
                wall_distances.append(distance)
        
        # Check if multiple consecutive sensors detect obstacles (wall pattern)
        consecutive_wall_sensors = self._find_consecutive_sensors(close_sensors, len(distances))
        
        # Wall criteria:
        # 1. Multiple consecutive sensors detect obstacles
        # 2. Obstacles are at consistent distances (indicating a flat wall)
        # 3. Obstacles cover significant portion of sensor array
        
        wall_coverage = len(close_sensors) / len(distances)
        
        if len(consecutive_wall_sensors) >= self.wall_surround_threshold:
            # Check if distances are consistent (wall-like pattern)
            if wall_distances:
                distance_variance = np.var(wall_distances)
                avg_distance = np.mean(wall_distances)
                
                # Low variance indicates a consistent wall surface
                if distance_variance < 0.1 and avg_distance < self.wall_proximity_threshold:
                    wall_info['is_wall'] = True
                    wall_info['wall_coverage'] = wall_coverage
                    wall_info['closest_wall_distance'] = min(wall_distances)
                    wall_info['obstacle_type'] = 'wall'
                    return True, wall_info
        
        # If not a wall, determine if it's a small obstacle
        if close_sensors:
            wall_info['obstacle_type'] = 'small_obstacle'
            wall_info['closest_wall_distance'] = min(wall_distances) if wall_distances else float('inf')
        
        return False, wall_info
    
    def _find_consecutive_sensors(self, sensor_indices, total_sensors):
        """Find consecutive sensor indices (handles wrap-around)"""
        if not sensor_indices:
            return []
        
        consecutive_groups = []
        current_group = [sensor_indices[0]]
        
        for i in range(1, len(sensor_indices)):
            current_idx = sensor_indices[i]
            prev_idx = sensor_indices[i-1]
            
            # Check if consecutive (including wrap-around)
            if (current_idx == prev_idx + 1) or (prev_idx == total_sensors - 1 and current_idx == 0):
                current_group.append(current_idx)
            else:
                consecutive_groups.append(current_group)
                current_group = [current_idx]
        
        consecutive_groups.append(current_group)
        
        # Return the largest consecutive group
        return max(consecutive_groups, key=len) if consecutive_groups else []
    
    def is_stuck_near_wall(self, drone_pos, sensor_data, pyb_client):
        """Enhanced stuck detection - ONLY for walls, not small obstacles"""
        # Update position history
        self.position_history.append(drone_pos.copy())
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        # Only check stuck if we have enough history
        if len(self.position_history) < self.history_length:
            return False
        
        # First, check if we're actually near a WALL (not just any obstacle)
        is_near_wall, wall_info = self.detect_wall_vs_obstacle(drone_pos, sensor_data, pyb_client)
        
        if not is_near_wall:
            # Reset stuck counter if not near wall
            self.stuck_counter = 0
            return False
        
        # Check if drone is truly stuck (minimal movement over long period)
        positions = np.array(self.position_history)
        
        # Calculate movement variance over the entire history
        position_variance = np.var(positions, axis=0)
        total_variance = np.sum(position_variance)
        
        # Calculate average movement between consecutive positions
        movements = []
        for i in range(1, len(positions)):
            movement = np.linalg.norm(positions[i] - positions[i-1])
            movements.append(movement)
        avg_movement = np.mean(movements) if movements else 1.0
        
        # Stuck criteria - only when near a wall
        is_position_stuck = (total_variance < 0.005 and avg_movement < 0.01)
        
        # Additional wall-specific stuck criteria
        wall_coverage = wall_info['wall_coverage']
        close_to_wall = wall_info['closest_wall_distance'] < 0.8
        
        # Only consider stuck if:
        # 1. Near a detected wall (not small obstacle)
        # 2. High wall coverage (drone is surrounded by wall)
        # 3. Very close to wall
        # 4. Minimal movement
        if is_near_wall and is_position_stuck and wall_coverage > 0.6 and close_to_wall:
            self.stuck_counter += 1
            if self.stuck_counter > 60:  # 2.5 seconds of continuous stuck condition
                print(f"🧱 Drone {self.drone_id} detected as stuck near WALL (not obstacle)")
                print(f"   Wall coverage: {wall_coverage:.2f}")
                print(f"   Distance to wall: {wall_info['closest_wall_distance']:.2f}")
                return True
        else:
            self.stuck_counter = 0
        
        return False
    
    def is_waypoint_unreachable(self, drone_pos, target_waypoint, time_threshold=15.0):
        """Check waypoint reachability - unchanged"""
        if self.target_waypoint is None or not np.array_equal(self.target_waypoint, target_waypoint):
            self.target_waypoint = target_waypoint.copy()
            self.waypoint_start_time = time.time()
            self.waypoint_stuck_counter = 0
            return False
        
        time_spent = time.time() - self.waypoint_start_time
        distance_to_waypoint = np.linalg.norm(drone_pos - target_waypoint)
        
        # Only unreachable if very far and very long time
        if time_spent > time_threshold and distance_to_waypoint > 2.0:
            return True
        
        return False
    
    def compute_force(self, drone_pos, drone_vel, waypoint, obstacles, altitude, other_drones=None, pyb_client=None):
        """Enhanced force computation with wall-specific stuck detection"""
        to_waypoint = waypoint - drone_pos
        dist_to_waypoint = np.linalg.norm(to_waypoint)
        
        # Create sensor data for wall detection
        sensor_data = {'distances': []}
        for obstacle in obstacles:
            if len(obstacle) >= 3:
                obstacle_pos = np.array(obstacle[0:3])
                dist = np.linalg.norm(drone_pos - obstacle_pos)
                sensor_data['distances'].append(dist)
        
        # Check for wall-specific stuck conditions
        stuck_near_wall = self.is_stuck_near_wall(drone_pos, sensor_data, pyb_client) if pyb_client else False
        waypoint_unreachable = self.is_waypoint_unreachable(drone_pos, waypoint)
        
        # Get wall information for debugging
        is_near_wall, wall_info = self.detect_wall_vs_obstacle(drone_pos, sensor_data, pyb_client) if pyb_client else (False, {})
        
        # Adaptive attraction - only reduce for walls, not small obstacles
        min_obstacle_dist = float('inf')
        for obstacle in obstacles:
            if len(obstacle) >= 3:
                obstacle_pos = np.array(obstacle[0:3])
                dist = np.linalg.norm(drone_pos - obstacle_pos)
                min_obstacle_dist = min(min_obstacle_dist, dist)
        
        attraction_gain = self.base_attraction_gain
        
        # Only reduce attraction for walls, not small obstacles
        if is_near_wall and wall_info.get('closest_wall_distance', float('inf')) < 1.0:
            attraction_gain *= max(0.3, wall_info['closest_wall_distance'])
            print(f"🧱 Drone {self.drone_id} near wall - reducing attraction")
        elif min_obstacle_dist < 0.8:  # Small obstacle - less reduction
            attraction_gain *= max(0.7, min_obstacle_dist / 0.8)
        
        if stuck_near_wall:
            attraction_gain *= 0.2
        
        # Compute waypoint direction and desired velocity
        waypoint_dir = to_waypoint / dist_to_waypoint if dist_to_waypoint > 0 else np.zeros(3)
        
        # PID-Enhanced Adaptive Speed System
        current_speed = np.linalg.norm(drone_vel[:2])  # Current horizontal speed

        # 1. Distance-based speed control using PID
        target_approach_speed = self.max_speed
        if dist_to_waypoint < self.slowdown_radius:
            # Use PID to smoothly approach waypoint
            # Setpoint: 0 (want to reach waypoint), Current: distance to waypoint
            distance_control = self.distance_pid.update(0, dist_to_waypoint)
            # Convert PID output to speed multiplier
            distance_multiplier = max(0.2, min(1.0, distance_control))
            target_approach_speed *= distance_multiplier
        else:
            # Reset distance PID when far from waypoint
            self.distance_pid.reset()

        # 2. Obstacle-based speed control using PID
        obstacle_speed_multiplier = 1.0
        if min_obstacle_dist < 1.0:
            # Use PID for smooth obstacle avoidance speed control
            # Setpoint: 1.5 (desired safe distance), Current: actual distance
            obstacle_control = self.obstacle_pid.update(1.0, min_obstacle_dist)
            obstacle_speed_multiplier = max(0.6, min(1.0, obstacle_control))
        else:
            # Reset obstacle PID when safe
            self.obstacle_pid.reset()

        # 3. Wall-specific speed adjustment
        wall_speed_multiplier = 1.0
        if is_near_wall and wall_info.get('closest_wall_distance', float('inf')) < 0.8:
            wall_distance = wall_info['closest_wall_distance']
            wall_speed_multiplier = max(0.5, wall_distance)
            print(f"🧱 Drone {self.drone_id} wall speed reduction: {wall_speed_multiplier:.2f}")

        # 4. Emergency speed adjustment
        emergency_speed_multiplier = 1.0
        if stuck_near_wall:
            emergency_speed_multiplier = 0.5
            print(f"🚨 Drone {self.drone_id} emergency speed reduction")

        openness_factor = min(1.5, min_obstacle_dist / 2.0)

        # 5. Combine all speed factors
        target_speed = (self.max_speed * 
                    min(target_approach_speed/self.max_speed, 1.0) *
                    obstacle_speed_multiplier * 
                    wall_speed_multiplier * 
                    emergency_speed_multiplier *
                    openness_factor)
        
        self.last_target_speed = target_speed

        # 6. Use main speed PID controller for smooth speed transitions
        base_desired_speed = self.speed_pid.update(target_speed, current_speed)

        # Apply speed zone factor if available
        if hasattr(self, 'current_speed_factor'):
            desired_speed = base_desired_speed * self.current_speed_factor
            if self.current_speed_factor < 0.8:  # Only print when significantly reduced
                print(f"🏎️ Drone {self.drone_id} speed zone factor: {self.current_speed_factor:.2f}")
        else:
            desired_speed = base_desired_speed

        # Debug output (optional)
        if self.drone_id == 0 and int(time.time() * 10) % 10 == 0:  # Print once per second for drone 0
            print(f"🎛️ Drone {self.drone_id} Speed PID: Target={target_speed:.2f}, Current={current_speed:.2f}, Output={desired_speed:.2f}")
        
        desired_velocity = waypoint_dir * desired_speed
        steer_attract = desired_velocity * attraction_gain
        
        # Enhanced repulsion
        steer_repulse = self._compute_enhanced_repulsion(drone_pos, drone_vel, obstacles, dist_to_waypoint)
        
        # Inter-drone avoidance
        drone_repulsion = self._compute_drone_repulsion(drone_pos, other_drones or [])
        
        # Wall following only for walls
        wall_follow_force = self._compute_wall_following_force(drone_pos, obstacles, stuck_near_wall)
        
        # Combine forces
        total_force = steer_attract + steer_repulse + drone_repulsion + wall_follow_force
        
        current_altitude = drone_pos[2]
        altitude_error = altitude - current_altitude
        
        # STRONG altitude control with priority over horizontal movement
        altitude_gain = 3.0  # Much stronger than before
        altitude_force = altitude_error * altitude_gain
        
        # Allow strong altitude correction
        max_altitude_force = 1.5  # Much stronger than ±0.2
        altitude_force = np.clip(altitude_force, -max_altitude_force, max_altitude_force)
        
        # PRIORITY SYSTEM: Strong altitude control
        if abs(altitude_error) > 0.15:  # If off altitude
            # REDUCE horizontal forces when altitude is wrong
            horizontal_reduction = min(1.0, abs(altitude_error) * 3.0)
            total_force[0] *= (1.0 - horizontal_reduction * 0.5)  # Reduce X
            total_force[1] *= (1.0 - horizontal_reduction * 0.5)  # Reduce Y
            total_force[2] = altitude_force  # STRONG altitude override
        else:
            # Normal altitude adjustment
            total_force[2] = altitude_force * 0.7 + total_force[2] * 0.3
        
        # Handle stuck condition
        total_force = self._handle_stuck_condition(drone_pos, total_force, stuck_near_wall)
        
        # Velocity smoothing
        total_force = self._apply_velocity_smoothing(total_force)
        
        # Limit force magnitude
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force and force_mag > 0:
            total_force = total_force / force_mag * self.max_force
        
        # Return wall information for waypoint management
        return total_force, dist_to_waypoint, stuck_near_wall, waypoint_unreachable, wall_info
    
    def _compute_wall_following_force(self, drone_pos, obstacles, stuck_near_wall):
        """Wall following only when truly stuck near walls"""
        if not stuck_near_wall and not self.wall_follow_active:
            return np.zeros(3)
        
        if stuck_near_wall and not self.wall_follow_active:
            self.wall_follow_active = True
            self.wall_follow_timer = 120
            print(f"🧱 Drone {self.drone_id} activating wall following (stuck near wall)")
        
        if self.wall_follow_active:
            # Find the closest wall/obstacle
            closest_obstacle = None
            min_dist = float('inf')
            
            for obstacle in obstacles:
                if len(obstacle) >= 3:
                    obstacle_pos = np.array(obstacle[0:3])
                    dist = np.linalg.norm(drone_pos - obstacle_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_obstacle = obstacle_pos
            
            if closest_obstacle is not None:
                to_wall = closest_obstacle - drone_pos
                to_wall_normalized = to_wall / np.linalg.norm(to_wall)
                
                wall_follow_dir = np.array([-to_wall_normalized[1], to_wall_normalized[0], 0]) * self.wall_follow_direction
                
                desired_wall_distance = 0.2
                if min_dist < desired_wall_distance:
                    wall_force = -to_wall_normalized * 0.6
                else:
                    wall_force = wall_follow_dir * 0.8
                
                self.wall_follow_timer -= 1
                if self.wall_follow_timer <= 0:
                    self.wall_follow_active = False
                    print(f"🧱 Drone {self.drone_id} ending wall following")
                
                return wall_force
        
        return np.zeros(3)
    
    def _compute_enhanced_repulsion(self, drone_pos, drone_vel, obstacles, dist_to_waypoint):
        """Normal repulsion for all obstacles"""
        steer_repulse = np.zeros(3)
        prediction_time = 1.5
        
        for obstacle in obstacles:
            if len(obstacle) < 3:
                continue
                
            obstacle_pos = np.array(obstacle[0:3])
            
            diff = drone_pos - obstacle_pos
            current_dist = np.linalg.norm(diff)
            
            future_pos = drone_pos + drone_vel * prediction_time
            future_diff = future_pos - obstacle_pos
            future_dist = np.linalg.norm(future_diff)
            
            critical_dist = min(current_dist, future_dist)
            critical_diff = diff if current_dist <= future_dist else future_diff
            
            adaptive_range = self.dynamic_repulsion_range
            if np.linalg.norm(drone_vel) > 1.0:
                adaptive_range *= 1.5
            if dist_to_waypoint < 2.0:
                adaptive_range *= 1.1
            
            if critical_dist < adaptive_range and critical_dist > 0:
                repulse_strength = self.base_repulsion_gain / (critical_dist * critical_dist + 0.01)
                
                if critical_dist < 0.2:
                    repulse_strength *= 2.0
                
                velocity_factor = max(1.0, np.linalg.norm(drone_vel))
                repulse_strength *= velocity_factor
                
                waypoint_factor = min(1.0, max(0.6, dist_to_waypoint / self.slowdown_radius))
                repulse_strength *= waypoint_factor
                
                repulse_direction = critical_diff / critical_dist
                steer_repulse += repulse_direction * repulse_strength
        
        return steer_repulse
    
    def _compute_drone_repulsion(self, drone_pos, other_drones):
        """Inter-drone repulsion"""
        drone_repulsion = np.zeros(3)
        separation_distance = 1.0
        
        for other_drone in other_drones:
            if len(other_drone) < 3:
                continue
                
            other_pos = np.array(other_drone[0:3])
            diff = drone_pos - other_pos
            dist = np.linalg.norm(diff)
            
            if dist < separation_distance and dist > 0.01:
                repulsion_strength = 60.0 / (dist * dist)
                direction = diff / dist
                drone_repulsion += direction * repulsion_strength
        
        return drone_repulsion
    
    def _handle_stuck_condition(self, drone_pos, total_force, stuck_near_wall):
        """Handle stuck conditions - only for walls"""
        if self.previous_pos is not None:
            movement = np.linalg.norm(drone_pos - self.previous_pos)
            if movement < self.stuck_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 2)
                if not stuck_near_wall:
                    self.emergency_maneuver_active = False
                    self.reset_pid_controllers()
        
        self.previous_pos = drone_pos.copy()
        
        # Only emergency for wall-stuck situations
        if stuck_near_wall and self.stuck_counter > self.max_stuck_time:
            if not self.emergency_maneuver_active:
                print(f"🧱 Drone {self.drone_id} stuck near wall - emergency escape")
                self.emergency_type = 'wall_escape'
                self.emergency_direction = np.array([1.2, 0.0, 0]) * np.random.choice([-1, 1])
                self.emergency_maneuver_active = True
                self.emergency_timer = 60
                self.reset_pid_controllers()
            
            if self.emergency_timer > 0:
                total_force += self.emergency_direction
                self.emergency_timer -= 1
            else:
                self.emergency_maneuver_active = False
                self.stuck_counter = 0
        
        return total_force
    
    def _apply_velocity_smoothing(self, total_force):
        """Apply velocity smoothing"""
        smoothed_force = (self.velocity_smoothing * self.previous_velocity + 
                         (1 - self.velocity_smoothing) * total_force)
        self.previous_velocity = smoothed_force.copy()
        return smoothed_force
    
    def reset_pid_controllers(self):
        """Reset all PID controllers (useful for emergency situations)"""
        self.speed_pid.reset()
        self.distance_pid.reset() 
        self.obstacle_pid.reset()
        print(f"🔄 Drone {self.drone_id} PID controllers reset")

class VelocityAviaryAdapter:
    """Adapter to convert potential field forces to VelocityAviary commands"""
    
    def __init__(self, max_horizontal_speed=2.0, max_vertical_speed=1.5, max_yaw_rate=0.5):  # REDUCED yaw rate
        self.max_horizontal_speed = max_horizontal_speed
        self.max_vertical_speed = max_vertical_speed
        self.max_yaw_rate = max_yaw_rate  # Reduced from 1.0 to 0.5
        
        # PID controllers for smooth control
        self.altitude_pid = PIDController(
            kp=2.0, ki=0.1, kd=0.5,
            max_output=max_vertical_speed,
            min_output=-max_vertical_speed
        )
        
        # SIMPLIFIED yaw PID with gentler parameters
        self.yaw_pid = PIDController(
            kp=0.8, ki=0.02, kd=0.1,  # REDUCED gains
            max_output=max_yaw_rate,
            min_output=-max_yaw_rate
        )
        
        # State tracking
        self.target_yaw = 0.0  # Initialize to 0
        self.yaw_initialized = True  # Start initialized
        
        print(f"✅ VelocityAviaryAdapter initialized:")
        print(f"   Max horizontal speed: {max_horizontal_speed} m/s")
        print(f"   Max vertical speed: {max_vertical_speed} m/s")
        print(f"   Max yaw rate: {max_yaw_rate} rad/s")
    
    def convert_force_to_velocity_command(self, pf_force, current_velocity, target_altitude, 
                                        current_altitude, current_rpy=None, desired_heading=None):
        """Convert potential field force to VelocityAviary-compatible command"""
        
        # 1. HORIZONTAL VELOCITY CONTROL
        horizontal_force = pf_force[:2]
        force_scale = 0.8
        desired_horizontal_velocity = horizontal_force * force_scale
        
        horizontal_speed = np.linalg.norm(desired_horizontal_velocity)
        if horizontal_speed > self.max_horizontal_speed:
            desired_horizontal_velocity = (desired_horizontal_velocity / horizontal_speed) * self.max_horizontal_speed
        
        vx_norm = np.clip(desired_horizontal_velocity[0] / self.max_horizontal_speed, -1.0, 1.0)
        vy_norm = np.clip(desired_horizontal_velocity[1] / self.max_horizontal_speed, -1.0, 1.0)
        
        # 2. VERTICAL VELOCITY CONTROL
        altitude_error = target_altitude - current_altitude
        vz_command = self.altitude_pid.update(0, -altitude_error)
        vz_norm = np.clip(vz_command / self.max_vertical_speed, -1.0, 1.0)
        
        # 3. ===== SIMPLIFIED YAW CONTROL =====
        yaw_rate = 0.0
        
        # Only do yaw control if we have current orientation and are not moving fast
        if current_rpy is not None:
            current_yaw = current_rpy[2]
            horizontal_speed = np.linalg.norm(desired_horizontal_velocity)
            
            # Update target yaw if a specific heading is desired
            if desired_heading is not None:
                self.target_yaw = desired_heading
            
            # Only apply yaw control if moving slowly (to avoid interference)
            if horizontal_speed < 0.3:  # Only yaw when moving slowly
                yaw_error = self._normalize_angle(self.target_yaw - current_yaw)
                
                # Large tolerance to prevent constant corrections
                if abs(yaw_error) > np.radians(20):  # 20 degree tolerance
                    yaw_rate = self.yaw_pid.update(0, yaw_error) * 0.5  # Reduced gain
                    yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate * 0.3, self.max_yaw_rate * 0.3)  # Further limit
                else:
                    # Reset PID when within tolerance
                    self.yaw_pid.reset()
                    yaw_rate = 0.0
            else:
                # Don't yaw when moving - update target to current to prevent snap-back
                self.target_yaw = current_yaw
                yaw_rate = 0.0
        
        # 4. RETURN COMMAND
        command = np.array([vx_norm, vy_norm, vz_norm, yaw_rate])
        
        return command
    
    def set_target_heading(self, heading_rad):
        """Set target heading manually"""
        self.target_yaw = self._normalize_angle(heading_rad)
        print(f"🧭 Target heading set to: {np.degrees(self.target_yaw):.1f}°")
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def get_status(self):
        """Get adapter status for debugging"""
        return {
            'target_yaw_deg': np.degrees(self.target_yaw),
            'yaw_initialized': self.yaw_initialized,
            'max_speeds': {
                'horizontal': self.max_horizontal_speed,
                'vertical': self.max_vertical_speed,
                'yaw_rate': self.max_yaw_rate
            }
        }


def initialize_drone_headings(force_adapters, initial_headings=None):
    """Initialize target headings for all drones"""
    for i, adapter in enumerate(force_adapters):
        if initial_headings is not None and i < len(initial_headings):
            # Set specific heading
            adapter.set_target_heading(np.radians(initial_headings[i]))
        else:
            # Use default (0 degrees = facing positive X)
            adapter.set_target_heading(0.0)
        
        print(f"🧭 Drone {i} heading initialized")

def set_heading_to_movement_direction(force_adapters, drone_id, target_pos, current_pos):
    """Set drone heading to face movement direction"""
    movement_vector = target_pos[:2] - current_pos[:2]
    if np.linalg.norm(movement_vector) > 0.1:
        movement_heading = np.arctan2(movement_vector[1], movement_vector[0])
        force_adapters[drone_id].set_target_heading(movement_heading)
        return True
    return False


# Adjusted waypoint manager - only skip for walls
class WaypointManager:
    def __init__(self, drone_id, waypoints):
        self.drone_id = drone_id
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.skipped_waypoints = []
        self.waypoint_attempt_start_time = None
        self.max_attempt_time = 25.0  # 25 seconds per waypoint
        self.stuck_threshold_time = 15.0  # 15 seconds before considering skip
        
    def get_current_waypoint(self):
        """Get current target waypoint"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    def should_skip_waypoint(self, drone_pos, stuck_near_wall, waypoint_unreachable, wall_info, current_simulation_time):
        """WALL-SPECIFIC waypoint skipping - only skip for walls, not small obstacles"""
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            return False
        
        # Initialize timing for new waypoint
        if self.waypoint_attempt_start_time is None:
            self.waypoint_attempt_start_time = current_simulation_time
            return False
        
        time_spent = current_simulation_time - self.waypoint_attempt_start_time
        distance_to_waypoint = np.linalg.norm(drone_pos - current_waypoint)
        
        # Check if we're dealing with a wall vs small obstacle
        is_wall_situation = wall_info.get('is_wall', False)
        obstacle_type = wall_info.get('obstacle_type', 'none')
        
        # ONLY skip for wall situations, not small obstacles
        skip_conditions = [
            # Only skip if stuck near WALL (not small obstacle) for long time
            stuck_near_wall and is_wall_situation and time_spent > self.stuck_threshold_time,
            # Only skip if waypoint unreachable due to wall blocking for very long time
            waypoint_unreachable and is_wall_situation and time_spent > self.stuck_threshold_time,
            # Maximum time exceeded for wall situations only
            time_spent > self.max_attempt_time and is_wall_situation,
        ]
        
        should_skip = any(skip_conditions)
        


        if should_skip:
            print(f"🧱 Drone {self.drone_id} WALL-based skip analysis:")
            print(f"   Time spent: {time_spent:.1f}s")
            print(f"   Distance to waypoint: {distance_to_waypoint:.2f}m")
            print(f"   Stuck near wall: {stuck_near_wall}")
            print(f"   Is wall situation: {is_wall_situation}")
            print(f"   Obstacle type: {obstacle_type}")
            print(f"   Wall coverage: {wall_info.get('wall_coverage', 0):.2f}")
        elif obstacle_type == 'small_obstacle':
            # Don't skip for small obstacles, just navigate around them
            if time_spent > 8.0:  # Reset timer for small obstacles
                print(f"🔄 Drone {self.drone_id} navigating around small obstacle (not skipping)")
                self.waypoint_attempt_start_time = time.time()  # Reset timer
        
        return should_skip
    
    def skip_to_next_waypoint(self, current_simulation_time):
        """Skip current waypoint"""
        if self.current_waypoint_index < len(self.waypoints):
            skipped_wp = self.waypoints[self.current_waypoint_index]
            self.skipped_waypoints.append((self.current_waypoint_index, skipped_wp))
            print(f"🧱 Drone {self.drone_id} skipping waypoint due to WALL: {skipped_wp}")
            
            self.current_waypoint_index += 1
            self.waypoint_attempt_start_time = None
            
            return True
        return False
    
    def advance_waypoint(self, current_simulation_time):
        """Advance to next waypoint normally"""
        if self.current_waypoint_index < len(self.waypoints): 
            print(f"✅ Drone {self.drone_id} reached waypoint {self.current_waypoint_index}")
            self.current_waypoint_index += 1
            self.waypoint_attempt_start_time = None
            return True
        return False
    
    def is_finished(self):
        """Check if all waypoints are processed"""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def get_skipped_waypoints(self):
        """Get list of skipped waypoints"""
        return self.skipped_waypoints
    
class DroneBatterySimulator:
    def __init__(self, drone_id, initial_battery=100.0, battery_capacity_mah=350):
        self.drone_id = drone_id
        self.initial_battery = initial_battery
        self.current_battery = initial_battery
        self.battery_capacity_mah = battery_capacity_mah
        
        # Battery consumption rates (mAh per second)
        self.base_consumption = 0.292  # Hovering consumption
        self.movement_consumption = 0.292  # Additional consumption when moving
        self.emergency_consumption = 0.584  # Additional consumption during emergency maneuvers
        self.camera_consumption = 0.0058  # Camera processing consumption
        
        # Battery behavior parameters
        self.voltage_min = 2.5  # Minimum safe voltage (3.6V per cell for 3S battery)
        self.voltage_max = 4.2  # Maximum voltage (4.2V per cell for 3S battery)
        self.low_battery_threshold = 20.0  # Low battery warning percentage
        self.critical_battery_threshold = 10.0  # Critical battery percentage
        self.rtl_battery_threshold = 15.0  # Return-to-launch threshold
        
        # Usage tracking
        self.total_flight_time = 0.0
        self.total_distance_traveled = 0.0
        self.emergency_maneuver_time = 0.0
        self.previous_position = None
        #self.last_update_time = time.time()
        
        # Battery status
        self.battery_status = "NORMAL"  # NORMAL, LOW, CRITICAL, EMERGENCY
        self.return_to_launch_triggered = False
        self.landing_required = False
        
    def update_battery(self, current_pos, current_vel, is_emergency_active=False, 
                      is_camera_active=True, dt=None, simulation_timestep=None):
        """
        Update battery based on current drone state
        
        Args:
            current_pos: Current position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            is_emergency_active: Whether emergency maneuvers are active
            is_camera_active: Whether camera is processing
            dt: Time delta (auto-calculated if None)
        """
        #current_time = time.time()

        if simulation_timestep is None:
            simulation_timestep = 1/25  # Default to 25 FPS
        
        dt = simulation_timestep
        
        # Calculate movement-based consumption
        speed = np.linalg.norm(current_vel) if current_vel is not None else 0.0
        
        # Base consumption (always present)
        consumption = self.base_consumption * dt
        
        # Movement consumption (proportional to speed)
        if speed > 0.1:  # Moving
            movement_factor = min(speed / 2.0, 2.0)  # Cap at 2x for very high speeds
            consumption += self.movement_consumption * movement_factor * dt
        
        # Emergency maneuver consumption
        if is_emergency_active:
            consumption += self.emergency_consumption * dt
            self.emergency_maneuver_time += dt
        
        # Camera processing consumption
        if is_camera_active:
            consumption += self.camera_consumption * dt
        
        # Altitude consumption (higher altitude = more power)
        if len(current_pos) > 2:
            altitude_factor = max(1.0, current_pos[2] / 2.0)  # Increase consumption with altitude
            consumption *= altitude_factor
        
        # Convert mAh consumption to percentage
        battery_consumed_percent = (consumption / self.battery_capacity_mah) * 100
        self.current_battery = max(0.0, self.current_battery - battery_consumed_percent)
        
        # Update tracking
        self.total_flight_time += dt
        if self.previous_position is not None:
            distance = np.linalg.norm(np.array(current_pos) - np.array(self.previous_position))
            self.total_distance_traveled += distance
        
        self.previous_position = current_pos.copy()
        #self.last_update_time = current_time
        
        # Update battery status
        self._update_battery_status()
        
        return self.current_battery
    
    def _update_battery_status(self):
        """Update battery status based on current level"""
        if self.current_battery <= 0.0:
            self.battery_status = "EMPTY"
            self.landing_required = True
        elif self.current_battery <= self.critical_battery_threshold:
            self.battery_status = "CRITICAL"
            self.landing_required = True
        elif self.current_battery <= self.rtl_battery_threshold:
            self.battery_status = "RTL"  # Return to Launch
            if not self.return_to_launch_triggered:
                self.return_to_launch_triggered = True
                print(f"🔋 Drone {self.drone_id} triggering Return-to-Launch (Battery: {self.current_battery:.1f}%)")
        elif self.current_battery <= self.low_battery_threshold:
            self.battery_status = "LOW"
        else:
            self.battery_status = "NORMAL"
    
    def get_voltage(self):
        """Calculate current voltage based on battery percentage"""
        voltage_range = self.voltage_max - self.voltage_min
        voltage = self.voltage_min + (self.current_battery / 100.0) * voltage_range
        return round(voltage, 2)
    
    def get_estimated_flight_time_remaining(self):
        """Estimate remaining flight time based on current consumption rate"""
        if self.total_flight_time <= 0:
            return float('inf')
        
        avg_consumption_rate = (self.initial_battery - self.current_battery) / self.total_flight_time
        if avg_consumption_rate <= 0:
            return float('inf')
        
        remaining_time = self.current_battery / avg_consumption_rate
        return max(0.0, remaining_time)
    
    def get_battery_info(self):
        """Get comprehensive battery information"""
        return {
            'percentage': round(self.current_battery, 1),
            'voltage': self.get_voltage(),
            'status': self.battery_status,
            'flight_time': round(self.total_flight_time, 1),
            'distance_traveled': round(self.total_distance_traveled, 1),
            'estimated_remaining_time': round(self.get_estimated_flight_time_remaining(), 1),
            'return_to_launch': self.return_to_launch_triggered,
            'landing_required': self.landing_required,
            'emergency_time': round(self.emergency_maneuver_time, 1)
        }
    
    def should_return_to_launch(self):
        """Check if drone should return to launch due to battery"""
        return self.return_to_launch_triggered
    
    def should_land_immediately(self):
        """Check if drone should land immediately due to critical battery"""
        return self.landing_required
    
    def force_land(self):
        """Force immediate landing due to battery depletion"""
        self.current_battery = 0.0
        self.battery_status = "EMPTY"
        self.landing_required = True
        print(f"⚠️ Drone {self.drone_id} FORCED LANDING - Battery depleted!")


def create_battery_simulators(num_drones, initial_batteries=None, battery_capacities=None):
    """
    Create battery simulators for all drones
    
    Args:
        num_drones: Number of drones
        initial_batteries: List of initial battery percentages (or single value for all)
        battery_capacities: List of battery capacities in mAh (or single value for all)
    """
    simulators = []
    
    for i in range(num_drones):
        # Set initial battery
        if initial_batteries is None:
            initial_battery = 100.0
        elif isinstance(initial_batteries, (int, float)):
            initial_battery = initial_batteries
        else:
            initial_battery = initial_batteries[i] if i < len(initial_batteries) else 100.0
        
        # Set battery capacity
        if battery_capacities is None:
            capacity = 350  # Default 350mAh
        elif isinstance(battery_capacities, (int, float)):
            capacity = battery_capacities
        else:
            capacity = battery_capacities[i] if i < len(battery_capacities) else 350
        
        simulator = DroneBatterySimulator(
            drone_id=i,
            initial_battery=initial_battery,
            battery_capacity_mah=capacity
        )
        simulators.append(simulator)
    
    return simulators


def handle_battery_emergency_actions(drone_id, battery_info, current_pos, waypoint_managers, exploration_targets):
    """
    Handle emergency actions based on battery status
    
    Returns:
        new_target_pos: Modified target position if battery emergency
        mission_status: 'continue', 'rtl', 'land'
    """
    if battery_info['landing_required']:
        # Force landing at current position
        landing_pos = current_pos.copy()
        landing_pos[2] = 0.1  # Land
        print(f"🚨 Drone {drone_id} EMERGENCY LANDING - Battery: {battery_info['percentage']:.1f}%")
        return landing_pos, 'land'
    
    elif battery_info['return_to_launch']:
        # Return to launch position (assuming origin [0, 0, altitude])
        rtl_pos = np.array([0.0, 0.0, current_pos[2]])
        print(f"🔋 Drone {drone_id} Returning to Launch - Battery: {battery_info['percentage']:.1f}%")
        
        # Clear exploration targets to prioritize RTL
        exploration_targets[drone_id] = None
        
        return rtl_pos, 'rtl'
    
    return None, 'continue'

# Visualization function for battery status
def visualize_battery_status(pyb_client, drone_positions, battery_simulators, building_size):
    """
    Add battery status visualization to the GUI
    """
    for j, (pos, battery_sim) in enumerate(zip(drone_positions, battery_simulators)):
        battery_info = battery_sim.get_battery_info()
        
        # Battery status colors
        if battery_info['status'] == 'CRITICAL':
            color = [1, 0, 0]  # Red
        elif battery_info['status'] == 'LOW':
            color = [1, 0.5, 0]  # Orange
        elif battery_info['status'] == 'RTL':
            color = [1, 1, 0]  # Yellow
        else:
            color = [0, 1, 0]  # Green
        
        # Draw battery indicator above drone
        battery_pos = np.array(pos) + np.array([0, 0, 0.8])
        p.addUserDebugLine(
            pos, battery_pos,
            lineColorRGB=color,
            lineWidth=6,
            lifeTime=0.1,
            physicsClientId=pyb_client
        )
        
        # Battery status text
        text_position = np.array([-building_size, building_size/2 - j*5, 2.0])
        battery_text = (f"🔋 Drone {j}: {battery_info['percentage']:.1f}% "
                       f"({battery_info['voltage']:.1f}V) | "
                       f"Status: {battery_info['status']} | "
                       f"Flight: {battery_info['flight_time']:.1f}s | "
                       f"Remaining: {battery_info['estimated_remaining_time']:.1f}s")
        
        p.addUserDebugText(battery_text, text_position, textColorRGB=color, 
                         textSize=1, lifeTime=0.1, physicsClientId=pyb_client)


def get_enhanced_sensor_readings(pyb_client, drone_pos, drone_vel, num_sensors=8, sensor_range=3.0):
    """Enhanced sensor readings with wall detection capabilities"""
    sensor_angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)
    
    # Adaptive range based on velocity
    speed = np.linalg.norm(drone_vel) if drone_vel is not None else 0
    if speed > 1.5:
        current_range = min(5.0, sensor_range * 1.5)
    elif speed < 0.5:
        current_range = max(1.0, sensor_range * 0.7)
    else:
        current_range = sensor_range
    
    sensor_data = {
        'distances': [],
        'obstacles': [],
        'threat_level': 0.0,
        'wall_detected': False,
        'wall_proximity': 0.0
    }
    
    near_range = current_range * 0.5
    wall_detection_threshold = 1.0
    close_obstacles = 0
    
    for i, angle in enumerate(sensor_angles):
        sensor_dir = np.array([np.cos(angle), np.sin(angle), 0])
        
        # Near range detection (high priority)
        near_end = drone_pos + sensor_dir * near_range
        near_result = p.rayTest(drone_pos, near_end, physicsClientId=pyb_client)
        
        # Far range detection
        far_end = drone_pos + sensor_dir * current_range
        far_result = p.rayTest(drone_pos, far_end, physicsClientId=pyb_client)
        
        if near_result[0][0] != -1:
            distance = near_result[0][2] * near_range
            sensor_data['distances'].append(distance)
            
            # High threat level for near obstacles
            threat_level = max(0, (near_range - distance) / near_range)
            sensor_data['threat_level'] = max(sensor_data['threat_level'], threat_level)
            
            # Wall detection
            if distance < wall_detection_threshold:
                close_obstacles += 1
            
            # Add obstacle position
            hit_pos = np.array(near_result[0][3])
            sensor_data['obstacles'].append([hit_pos[0], hit_pos[1], hit_pos[2], distance])
            
        elif far_result[0][0] != -1:
            distance = far_result[0][2] * current_range
            sensor_data['distances'].append(distance)
            
            # Medium threat level for far obstacles
            threat_level = max(0, (current_range - distance) / current_range) * 0.5
            sensor_data['threat_level'] = max(sensor_data['threat_level'], threat_level)
            
            # Add obstacle position
            hit_pos = np.array(far_result[0][3])
            sensor_data['obstacles'].append([hit_pos[0], hit_pos[1], hit_pos[2], distance])
            
        else:
            sensor_data['distances'].append(current_range)
    
    # Wall detection logic
    if close_obstacles >= 2:  # At least 3 sensors detecting close obstacles
        sensor_data['wall_detected'] = True
        sensor_data['wall_proximity'] = close_obstacles / num_sensors
    
    return sensor_data



class NadirCoverageTracker:
    def __init__(self, area_width=50.0, area_height=20.0, resolution=0.2, 
                 focal_length=4e-3, sensor_width=6.17e-3, sensor_height=4.55e-3):
        self.area_width = area_width
        self.area_height = area_height
        self.resolution = resolution
        
        # Calculate grid size
        self.grid_width = int(area_width / resolution)
        self.grid_height = int(area_height / resolution)
        self.coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Camera parameters
        self.f = focal_length
        self.sensor_w = sensor_width
        self.sensor_h = sensor_height
        
        # Pre-compute FOVs
        self.h_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        self.v_fov = 2 * np.arctan(sensor_height / (2 * focal_length))
        
        print(f"✅ Improved Coverage Tracker:")
        print(f"   Area: {area_width}m x {area_height}m")
        print(f"   Grid: {self.grid_width} x {self.grid_height} = {self.grid_width * self.grid_height:,} cells")
        print(f"   Resolution: {resolution}m per cell")
        print(f"   Camera FOV: H={np.degrees(self.h_fov):.1f}°, V={np.degrees(self.v_fov):.1f}°")

    def calculate_footprint(self, altitude):
        ground_w = 2 * altitude * np.tan(self.h_fov / 2)
        ground_h = 2 * altitude * np.tan(self.v_fov / 2)
        return (ground_w, ground_h)

    def update(self, drone_positions, altitude):
        """Update coverage map with drone positions"""
        ground_w, ground_h = self.calculate_footprint(altitude)
        
        # Convert to grid cells
        footprint_cells_w = max(1, int(ground_w / self.resolution))
        footprint_cells_h = max(1, int(ground_h / self.resolution))
        
        for pos in drone_positions:
            # Convert world coordinates to grid coordinates
            grid_x = int((pos[0] + self.area_width/2) / self.resolution)
            grid_y = int((pos[1] + self.area_height/2) / self.resolution)
            
            # Mark coverage area (footprint)
            x_min = max(0, grid_x - footprint_cells_w//2)
            x_max = min(self.grid_width, grid_x + footprint_cells_w//2 + 1)
            y_min = max(0, grid_y - footprint_cells_h//2)
            y_max = min(self.grid_height, grid_y + footprint_cells_h//2 + 1)
            
            self.coverage_map[y_min:y_max, x_min:x_max] = 255
            
    def get_coverage_percentage(self):
        """Get accurate coverage percentage"""
        covered_cells = np.sum(self.coverage_map > 0)
        total_cells = self.coverage_map.size
        coverage_percent = (covered_cells / total_cells) * 100
        
        return {
            'covered_cells': int(covered_cells),
            'total_possible_cells': int(total_cells),
            'coverage_percentage': coverage_percent,
            'area_covered_m2': covered_cells * (self.resolution ** 2),
            'total_area_m2': self.area_width * self.area_height
        }
        
    def visualize_coverage_map(self, window_name="Coverage Map"):
        """Visualize the coverage map"""
        if self.coverage_map.size > 0:
            # Create colored visualization
            coverage_colored = cv2.applyColorMap(self.coverage_map, cv2.COLORMAP_JET)
            
            # Resize for better visualization
            scale_factor = max(1, 800 // max(self.grid_width, self.grid_height))
            display_img = cv2.resize(coverage_colored, 
                                   (self.grid_width * scale_factor, self.grid_height * scale_factor),
                                   interpolation=cv2.INTER_NEAREST)
            
            display_img = cv2.flip(display_img, 0)  # Flip vertically
            
            # Add coverage percentage text
            coverage_info = self.get_coverage_percentage()
            cv2.putText(display_img, f"Coverage: {coverage_info['coverage_percentage']:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_img)
            cv2.waitKey(1)



class SimpleCoverageVisualizer:
    def __init__(self, area_width=50.0, area_height=20.0, resolution=2.0):
        self.area_width = area_width
        self.area_height = area_height
        self.resolution = resolution
        self.coverage_bodies = {}
        
        # Camera parameters
        self.focal_length = 4e-3
        self.sensor_width = 6.17e-3
        self.sensor_height = 4.55e-3
        self.h_fov = 2 * np.arctan(self.sensor_width / (2 * self.focal_length))
        self.v_fov = 2 * np.arctan(self.sensor_height / (2 * self.focal_length))
        
        # Calculate total possible coverage cells for percentage calculation
        self.total_possible_cells = int((area_width / resolution) * (area_height / resolution))
        
        print(f"✅ SimpleCoverageVisualizer initialized:")
        print(f"   Area: {area_width}m x {area_height}m ({area_width*area_height:.0f}m²)")
        print(f"   Resolution: {resolution}m -> {self.total_possible_cells} possible cells")
        
    def calculate_realistic_footprint(self, altitude):
        """Calculate actual camera footprint based on altitude"""
        ground_w = 2 * altitude * np.tan(self.h_fov / 2)
        ground_h = 2 * altitude * np.tan(self.v_fov / 2)
        return (ground_w, ground_h)
        
    def update_coverage_from_positions(self, drone_positions, altitude, pyb_client=None):
        """Create coverage zones using REALISTIC camera footprint"""
        ground_w, ground_h = self.calculate_realistic_footprint(altitude)
        
        for i, pos in enumerate(drone_positions):
            # Check if position is within RECTANGULAR mission area bounds
            if (abs(pos[0]) > self.area_width/2 or abs(pos[1]) > self.area_height/2):
                continue  # Skip if outside rectangular bounds
            
            # Grid snapping within rectangular bounds
            key = (round(pos[0]/self.resolution)*self.resolution, 
                   round(pos[1]/self.resolution)*self.resolution)
            
            if key not in self.coverage_bodies:
                # Create coverage zone with actual camera footprint size
                visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[ground_w/2, ground_h/2, 0.05],
                    rgbaColor=[0.0, 1.0, 0.0, 0.2],
                    physicsClientId=pyb_client
                )
                
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_id,
                    basePosition=[key[0], key[1], 0.1],
                    physicsClientId=pyb_client
                )
                
                self.coverage_bodies[key] = body_id
    
    def get_coverage_percentage(self):
        """Calculate coverage percentage for RECTANGULAR area"""
        covered_cells = len(self.coverage_bodies)
        coverage_percent = (covered_cells / self.total_possible_cells * 100) if self.total_possible_cells > 0 else 0
        
        return {
            'covered_cells': covered_cells,
            'total_possible_cells': self.total_possible_cells,
            'coverage_percentage': coverage_percent,
            'area_covered_m2': covered_cells * (self.resolution ** 2),
            'total_area_m2': self.area_width * self.area_height
        }
    
    def clear_all_coverage(self, pyb_client):
        """Clear all coverage visualization bodies"""
        try:
            for body_id in self.coverage_bodies.values():
                p.removeBody(body_id, physicsClientId=pyb_client)
            self.coverage_bodies.clear()
            print("🧹 Coverage visualization cleared")
        except Exception as e:
            print(f"⚠️ Error clearing coverage visualization: {e}")

class OverlapDetector:
    def __init__(self, area_width=50.0, area_height=20.0, resolution=0.4, 
                 focal_length=4e-3, sensor_width=6.17e-3, sensor_height=4.55e-3,
                 time_threshold=2.0):
        self.area_width = area_width
        self.area_height = area_height
        self.resolution = resolution
        self.time_threshold = time_threshold  # seconds - drones must be within this time to count as overlap
        
        # Camera parameters (same as coverage tracker)
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.h_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        self.v_fov = 2 * np.arctan(sensor_height / (2 * focal_length))
        
        # Overlap tracking
        self.current_overlaps = {}  # {(drone1, drone2): overlap_data}
        self.total_overlap_time = 0.0
        self.overlap_events = []  # List of overlap events
        self.overlap_coverage_cells = set()  # Grid cells that have overlapped
        
        print(f"✅ OverlapDetector initialized:")
        print(f"   Time threshold: {time_threshold}s")
        print(f"   Resolution: {resolution}m")
    
    def calculate_footprint(self, altitude):
        """Calculate camera footprint size at given altitude"""
        ground_w = 2 * altitude * np.tan(self.h_fov / 2)
        ground_h = 2 * altitude * np.tan(self.v_fov / 2)
        return (ground_w, ground_h)
    
    def get_footprint_cells(self, position, altitude):
        """Optimized: Get grid cells covered by drone's camera footprint"""
        # Quick bounds check first
        if (abs(position[0]) > self.area_width/2 or abs(position[1]) > self.area_height/2):
            return set()
        
        ground_w, ground_h = self.calculate_footprint(altitude)
        
        # Use faster integer calculations
        center_x = int((position[0] + self.area_width/2) / self.resolution)
        center_y = int((position[1] + self.area_height/2) / self.resolution)
        
        # Smaller footprint calculation (reduce precision for speed)
        footprint_cells_w = max(1, int(ground_w / self.resolution) // 2)  # Reduced size
        footprint_cells_h = max(1, int(ground_h / self.resolution) // 2)  # Reduced size
        
        # Pre-calculate bounds
        grid_width = int(self.area_width/self.resolution)
        grid_height = int(self.area_height/self.resolution)
        
        # Optimized cell generation with bounds checking
        cells = set()
        for dx in range(-footprint_cells_w, footprint_cells_w + 1):
            for dy in range(-footprint_cells_h, footprint_cells_h + 1):
                cell_x = center_x + dx
                cell_y = center_y + dy
                if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                    cells.add((cell_x, cell_y))
        
        return cells
    
    def detect_overlaps(self, drone_positions, current_time, altitude):
        """Detect simultaneous overlaps between drones"""
        num_drones = len(drone_positions)
        current_overlaps = {}
        overlap_detected = False
        
        # Get footprint cells for each drone
        drone_footprints = {}
        for i, pos in enumerate(drone_positions):
            drone_footprints[i] = self.get_footprint_cells(pos, altitude)
        
        # Check all pairs of drones
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                # Calculate overlap area
                overlap_cells = drone_footprints[i].intersection(drone_footprints[j])
                
                if overlap_cells:
                    # Calculate physical distance between drones
                    distance = np.linalg.norm(np.array(drone_positions[i]) - np.array(drone_positions[j]))
                    overlap_area = len(overlap_cells) * (self.resolution ** 2)
                    
                    pair_key = (i, j)
                    
                    # Check if this is a new overlap or continuation
                    if pair_key in self.current_overlaps:
                        # Continue existing overlap
                        self.current_overlaps[pair_key]['end_time'] = current_time
                        self.current_overlaps[pair_key]['duration'] = current_time - self.current_overlaps[pair_key]['start_time']
                        self.current_overlaps[pair_key]['overlap_cells'].update(overlap_cells)
                        self.current_overlaps[pair_key]['max_overlap_area'] = max(
                            self.current_overlaps[pair_key]['max_overlap_area'], overlap_area
                        )
                    else:
                        # New overlap detected
                        self.current_overlaps[pair_key] = {
                            'start_time': current_time,
                            'end_time': current_time,
                            'duration': 0.0,
                            'overlap_cells': overlap_cells.copy(),
                            'max_overlap_area': overlap_area,
                            'start_positions': (drone_positions[i].copy(), drone_positions[j].copy()),
                            'distance': distance
                        }
                        
                        print(f"🔄 OVERLAP DETECTED: Drones {i} & {j} at time {current_time:.1f}s")
                        print(f"   Distance: {distance:.2f}m, Overlap area: {overlap_area:.2f}m²")
                    
                    current_overlaps[pair_key] = True
                    overlap_detected = True
                    
                    # Add to global overlap cells
                    self.overlap_coverage_cells.update(overlap_cells)
        
        # End overlaps that are no longer active
        ended_overlaps = []
        for pair_key in list(self.current_overlaps.keys()):
            if pair_key not in current_overlaps:
                # Overlap ended
                overlap_data = self.current_overlaps[pair_key]
                if overlap_data['duration'] >= 0.5:  # Only record overlaps > 0.5 seconds
                    self.overlap_events.append(overlap_data.copy())
                    print(f"✅ OVERLAP ENDED: Drones {pair_key[0]} & {pair_key[1]} - Duration: {overlap_data['duration']:.1f}s")
                
                ended_overlaps.append(pair_key)
        
        # Remove ended overlaps
        for pair_key in ended_overlaps:
            del self.current_overlaps[pair_key]
        
        return overlap_detected, len(current_overlaps)
    
    def get_overlap_statistics(self):
        """Get comprehensive overlap statistics"""
        total_overlap_area = len(self.overlap_coverage_cells) * (self.resolution ** 2)
        total_area = self.area_width * self.area_height
        overlap_percentage = (total_overlap_area / total_area) * 100
        
        # Calculate total overlap duration
        total_duration = sum(event['duration'] for event in self.overlap_events)
        total_duration += sum(overlap['duration'] for overlap in self.current_overlaps.values())
        
        return {
            'total_overlap_events': len(self.overlap_events),
            'active_overlaps': len(self.current_overlaps),
            'total_overlap_duration': total_duration,
            'overlap_area_m2': total_overlap_area,
            'overlap_percentage': overlap_percentage,
            'overlap_cells_count': len(self.overlap_coverage_cells),
            'avg_overlap_duration': total_duration / max(1, len(self.overlap_events))
        }

class EnhancedCoverageVisualizer:
    def __init__(self, area_width=50.0, area_height=20.0, resolution=0.4):
        self.area_width = area_width
        self.area_height = area_height
        self.resolution = resolution
        self.coverage_bodies = {}  # Normal coverage (green)
        self.overlap_bodies = {}   # Overlap coverage (yellow)
        
        # Camera parameters
        self.focal_length = 4e-3
        self.sensor_width = 6.17e-3
        self.sensor_height = 4.55e-3
        self.h_fov = 2 * np.arctan(self.sensor_width / (2 * self.focal_length))
        self.v_fov = 2 * np.arctan(self.sensor_height / (2 * self.focal_length))
        
        # Calculate total possible coverage cells for percentage calculation
        self.total_possible_cells = int((area_width / resolution) * (area_height / resolution))
        
        print(f"✅ EnhancedCoverageVisualizer initialized with overlap detection")
        
    def calculate_realistic_footprint(self, altitude):
        """Calculate actual camera footprint based on altitude"""
        ground_w = 2 * altitude * np.tan(self.h_fov / 2)
        ground_h = 2 * altitude * np.tan(self.v_fov / 2)
        return (ground_w, ground_h)
        
    def update_coverage_from_positions(self, drone_positions, altitude, overlap_cells=None, pyb_client=None, show_overlap=True):
        """Create coverage zones with conditional overlap highlighting"""
        ground_w, ground_h = self.calculate_realistic_footprint(altitude)
        
        for i, pos in enumerate(drone_positions):
            # Check if position is within bounds
            if (abs(pos[0]) > self.area_width/2 or abs(pos[1]) > self.area_height/2):
                continue
            
            # Grid snapping
            key = (round(pos[0]/self.resolution)*self.resolution, 
                round(pos[1]/self.resolution)*self.resolution)
            
            # Convert position to grid coordinates for overlap checking
            grid_x = int((pos[0] + self.area_width/2) / self.resolution)
            grid_y = int((pos[1] + self.area_height/2) / self.resolution)
            
            # Check if this cell is in overlap (only if overlap detection is active)
            is_overlap_cell = (show_overlap and overlap_cells and 
                            (grid_x, grid_y) in overlap_cells)
            
            if is_overlap_cell:
                # Create/update yellow overlap coverage
                if key not in self.overlap_bodies:
                    # Remove from normal coverage if exists
                    if key in self.coverage_bodies:
                        p.removeBody(self.coverage_bodies[key], physicsClientId=pyb_client)
                        del self.coverage_bodies[key]
                    
                    # Create yellow overlap visualization
                    visual_id = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[ground_w/2, ground_h/2, 0.05],
                        rgbaColor=[1.0, 1.0, 0.0, 0.4],  # Yellow with transparency
                        physicsClientId=pyb_client
                    )
                    
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual_id,
                        basePosition=[key[0], key[1], 0.12],  # Slightly higher than normal coverage
                        physicsClientId=pyb_client
                    )
                    
                    self.overlap_bodies[key] = body_id
            else:
                # Normal coverage (green) - only if not already overlap
                if key not in self.coverage_bodies and key not in self.overlap_bodies:
                    visual_id = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[ground_w/2, ground_h/2, 0.05],
                        rgbaColor=[0.0, 1.0, 0.0, 0.2],  # Green
                        physicsClientId=pyb_client
                    )
                    
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual_id,
                        basePosition=[key[0], key[1], 0.1],
                        physicsClientId=pyb_client
                    )
                    
                    self.coverage_bodies[key] = body_id
    
    def get_coverage_percentage(self):
        """Calculate coverage percentage including overlap information"""
        normal_cells = len(self.coverage_bodies)
        overlap_cells = len(self.overlap_bodies)
        total_covered = normal_cells + overlap_cells
        
        coverage_percent = (total_covered / self.total_possible_cells * 100) if self.total_possible_cells > 0 else 0
        overlap_percent = (overlap_cells / max(1, total_covered) * 100) if total_covered > 0 else 0
        
        return {
            'covered_cells': total_covered,
            'normal_coverage_cells': normal_cells,
            'overlap_coverage_cells': overlap_cells,
            'total_possible_cells': self.total_possible_cells,
            'coverage_percentage': coverage_percent,
            'overlap_percentage_of_coverage': overlap_percent,
            'area_covered_m2': total_covered * (self.resolution ** 2),
            'overlap_area_m2': overlap_cells * (self.resolution ** 2),
            'total_area_m2': self.area_width * self.area_height
        }
    
    def clear_all_coverage(self, pyb_client):
        """Clear all coverage visualization bodies"""
        try:
            for body_id in self.coverage_bodies.values():
                p.removeBody(body_id, physicsClientId=pyb_client)
            for body_id in self.overlap_bodies.values():
                p.removeBody(body_id, physicsClientId=pyb_client)
            self.coverage_bodies.clear()
            self.overlap_bodies.clear()
            print("🧹 Enhanced coverage visualization cleared")
        except Exception as e:
            print(f"⚠️ Error clearing enhanced coverage visualization: {e}")

class SafeDroneRestartManager:
    """Safer restart manager with proper safeguards"""
    
    def __init__(self, num_drones, env, target_altitude=2.0):
        self.num_drones = num_drones
        self.env = env
        self.target_altitude = target_altitude
        
        # IMPROVED: More aggressive detection parameters
        self.critical_altitude = 0.5  # CHANGED from 0.2
        self.falling_velocity_threshold = -0.1  # CHANGED from -1.5
        self.detection_delay = 5  # CHANGED from 50
        
        # Restart tracking (same as before)
        self.falling_counters = [0 for _ in range(num_drones)]
        self.restart_cooldowns = [0 for _ in range(num_drones)]
        self.restart_counts = [0 for _ in range(num_drones)]
        self.max_restarts = 10  # INCREASED from 3
        
        # IMPROVED: Dynamic body ID discovery
        self.drone_body_ids = None
        self.body_ids_discovered = False
        
        # Try multiple methods to get drone body IDs
        if hasattr(env, 'DRONE_IDS'):
            self.drone_body_ids = env.DRONE_IDS
            self.body_ids_discovered = True
            print(f"✅ Found env.DRONE_IDS: {self.drone_body_ids}")
        elif hasattr(env, 'getDroneIds'):
            self.drone_body_ids = env.getDroneIds()
            self.body_ids_discovered = True
            print(f"✅ Found env.getDroneIds(): {self.drone_body_ids}")
        elif hasattr(env, '_droneIds'):
            self.drone_body_ids = env._droneIds
            self.body_ids_discovered = True
            print(f"✅ Found env._droneIds: {self.drone_body_ids}")
        else:
            print("⚠️ Will discover drone body IDs dynamically during simulation")

    def discover_body_ids_if_needed(self, positions, pyb_client):
        """Discover drone body IDs by matching positions with PyBullet bodies"""
        if self.body_ids_discovered:
            return  # Already discovered
            
        print("🔍 Discovering drone body IDs by position matching...")
        self.drone_body_ids = []
        num_bodies = p.getNumBodies(physicsClientId=pyb_client)
        
        for drone_idx, target_pos in enumerate(positions):
            best_match = None
            min_distance = float('inf')
            
            # Check all bodies in PyBullet simulation
            for body_id in range(num_bodies):
                try:
                    body_pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=pyb_client)
                    distance = np.linalg.norm(np.array(target_pos) - np.array(body_pos))
                    
                    if distance < min_distance and distance < 0.5:  # Within 0.5m
                        min_distance = distance
                        best_match = body_id
                except:
                    continue  # Skip if can't get body position
            
            if best_match is not None:
                self.drone_body_ids.append(best_match)
                print(f"✅ Drone {drone_idx} → PyBullet body {best_match} (distance: {min_distance:.3f}m)")
            else:
                print(f"❌ Could not find body for drone {drone_idx}, using fallback")
                self.drone_body_ids.append(drone_idx)  # Fallback to index
        
        self.body_ids_discovered = True
        print(f"🎯 Final drone body IDs: {self.drone_body_ids}")
    
    def check_and_restart_safely(self, positions, velocities, pyb_client):
        """Safely check and restart fallen drones with proper safeguards"""
        self.discover_body_ids_if_needed(positions, pyb_client)
        restart_occurred = False
        
        for j in range(self.num_drones):
            # Skip if in cooldown period
            if self.restart_cooldowns[j] > 0:
                self.restart_cooldowns[j] -= 1
                continue
            
            # Skip if too many restarts
            if self.restart_counts[j] >= self.max_restarts:
                continue
            
            current_pos = positions[j]
            current_vel = velocities[j]
            altitude = current_pos[2]
            vertical_velocity = current_vel[2] if len(current_vel) > 2 else 0
            
            # CONSERVATIVE detection: Only restart if really crashed
            is_crashed = (
                altitude < self.critical_altitude and  # Very low
                vertical_velocity < self.falling_velocity_threshold  # Falling fast
            )
            
            if is_crashed:
                self.falling_counters[j] += 1
                
                # Wait for detection delay before restart
                if self.falling_counters[j] >= self.detection_delay:
                    success = self._safe_restart_drone(j, current_pos, pyb_client)
                    if success:
                        restart_occurred = True
                        self.falling_counters[j] = 0
                        self.restart_cooldowns[j] = 100  # 100 iteration cooldown
                        self.restart_counts[j] += 1
            else:
                # Reset falling counter if drone recovers
                self.falling_counters[j] = max(0, self.falling_counters[j] - 1)
        
        return restart_occurred
    
    def _safe_restart_drone(self, drone_id, current_pos, pyb_client):
        """Safely restart a single drone"""
        try:
            # Get correct PyBullet body ID
            # Get correct PyBullet body ID
            if self.drone_body_ids is None or drone_id >= len(self.drone_body_ids):
                print(f"❌ No valid body ID for drone {drone_id}")
                return False

            body_id = self.drone_body_ids[drone_id]
            print(f"🔄 Restarting drone {drone_id} using PyBullet body {body_id}")
            
            # Calculate safe restart position
            restart_pos = [
                current_pos[0],  # Same X
                current_pos[1],  # Same Y  
                self.target_altitude  # Safe altitude
            ]
            
            # Reset position with proper orientation
            p.resetBasePositionAndOrientation(
                body_id,
                restart_pos,
                [0, 0, 0, 1],  # Level orientation
                physicsClientId=pyb_client
            )
            
            # Reset velocities to zero
            p.resetBaseVelocity(
                body_id,
                [0, 0, 0],  # Zero linear velocity
                [0, 0, 0],  # Zero angular velocity
                physicsClientId=pyb_client
            )
            
            print(f"✅ Safely restarted drone {drone_id} at {restart_pos}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restart drone {drone_id}: {e}")
            return False
    
    def get_restart_stats(self):
        """Get restart statistics"""
        return {
            'total_restarts': sum(self.restart_counts),
            'individual_restarts': self.restart_counts.copy(),
            'falling_counters': self.falling_counters.copy(),
            'cooldowns': self.restart_cooldowns.copy()
        }

def draw_realistic_camera_footprints(drone_positions, altitude, pyb_client):
    """Draw REALISTIC camera field of view based on actual camera parameters"""
    # Use the SAME camera parameters as your existing code
    focal_length = 4e-3
    sensor_width = 6.17e-3
    sensor_height = 4.55e-3
    
    # Calculate FOV angles
    h_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
    v_fov = 2 * np.arctan(sensor_height / (2 * focal_length))
    
    # Calculate actual ground footprint
    ground_w = 2 * altitude * np.tan(h_fov / 2)
    ground_h = 2 * altitude * np.tan(v_fov / 2)
    
    print(f"📷 Drawing REALISTIC camera footprints:")
    print(f"   Altitude: {altitude:.2f}m -> Footprint: {ground_w:.2f}x{ground_h:.2f}m")
    print(f"   FOV: H={np.degrees(h_fov):.1f}°, V={np.degrees(v_fov):.1f}°")
    
    for i, pos in enumerate(drone_positions):
        # Calculate actual footprint corners
        corners = [
            [pos[0] - ground_w/2, pos[1] - ground_h/2, 0.02],
            [pos[0] + ground_w/2, pos[1] - ground_h/2, 0.02],
            [pos[0] + ground_w/2, pos[1] + ground_h/2, 0.02],
            [pos[0] - ground_w/2, pos[1] + ground_h/2, 0.02]
        ]
        
        # Draw realistic yellow rectangle
        for j in range(4):
            p.addUserDebugLine(
                corners[j], corners[(j + 1) % 4],
                lineColorRGB=[1, 1, 0],  # Yellow
                lineWidth=3,
                lifeTime=0.3,
                physicsClientId=pyb_client
            )
        
        # Add center cross with actual dimensions
        p.addUserDebugLine(
            [pos[0] - ground_w/4, pos[1], pos[2]],
            [pos[0] + ground_w/4, pos[1], pos[2]],
            lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0.3,
            physicsClientId=pyb_client
        )
        p.addUserDebugLine(
            [pos[0], pos[1] - ground_h/4, pos[2]],
            [pos[0], pos[1] + ground_h/4, pos[2]],
            lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0.3,
            physicsClientId=pyb_client
        )

def find_nearest_unvisited_large_area(current_pos, coverage_map, resolution=0.1, origin=[-7.5, -7.5], 
                                     min_cluster_size=25, min_area_threshold=0.25, wall_buffer=0.0):

    from scipy.ndimage import label, center_of_mass
    import numpy as np
    
    h, w = coverage_map.shape
    
    # Create binary mask of unvisited areas (coverage < 10)
    unvisited_mask = coverage_map < 10

    # Add wall buffer functionality
    if wall_buffer > 0:
        h, w = coverage_map.shape
        wall_buffer_cells = int(wall_buffer / resolution)
        
        # Create safe zone mask (exclude border areas)
        safe_mask = np.ones_like(unvisited_mask, dtype=bool)
        safe_mask[:wall_buffer_cells, :] = False      # Top border
        safe_mask[-wall_buffer_cells:, :] = False     # Bottom border  
        safe_mask[:, :wall_buffer_cells] = False      # Left border
        safe_mask[:, -wall_buffer_cells:] = False     # Right border
        
        # Apply wall buffer to unvisited areas
        unvisited_mask = unvisited_mask & safe_mask
        
        print(f"🛡️ Wall buffer applied: {wall_buffer}m ({wall_buffer_cells} cells)")
        print(f"   Safe area: {h-2*wall_buffer_cells}x{w-2*wall_buffer_cells} of {h}x{w} cells")
    
    # Find connected components (clusters) of unvisited areas
    labeled_array, num_clusters = label(unvisited_mask)
    
    if num_clusters == 0:
        return None
    
    # Calculate minimum cluster size based on area threshold
    min_pixels_for_area = int(min_area_threshold / (resolution * resolution))
    effective_min_cluster_size = max(min_cluster_size, min_pixels_for_area)
    
    # Find large clusters and their properties
    large_clusters = []
    
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = (labeled_array == cluster_id)
        cluster_size = np.sum(cluster_mask)
        
        # Only consider clusters that meet size requirements
        if cluster_size >= effective_min_cluster_size:
            # Calculate cluster center of mass
            cluster_center_ij = center_of_mass(cluster_mask)
            
            # Convert to world coordinates
            center_x = cluster_center_ij[1] * resolution + origin[0]
            center_y = cluster_center_ij[0] * resolution + origin[1]
            
            # Calculate distance from current position
            distance = np.linalg.norm(np.array([center_x, center_y]) - current_pos[:2])
            
            # Calculate cluster area in square meters
            cluster_area = cluster_size * (resolution * resolution)
            
            large_clusters.append({
                'center': (center_x, center_y),
                'distance': distance,
                'size': cluster_size,
                'area': cluster_area,
                'cluster_id': cluster_id,
                'mask': cluster_mask
            })
    
    if not large_clusters:
        print(f"No large unvisited areas found (min size: {effective_min_cluster_size} pixels, {min_area_threshold:.1f}m²)")
        return None
    
    # Sort clusters by distance (nearest first)
    large_clusters.sort(key=lambda c: c['distance'])
    
    # Get the nearest large cluster
    nearest_cluster = large_clusters[0]
    
    print(f"📍 Found {len(large_clusters)} large unvisited areas:")
    for i, cluster in enumerate(large_clusters[:3]):  # Show top 3
        print(f"   {i+1}. Area: {cluster['area']:.1f}m², Distance: {cluster['distance']:.1f}m, Size: {cluster['size']} pixels")
    
    print(f"🎯 Targeting nearest large area: {nearest_cluster['area']:.1f}m² at distance {nearest_cluster['distance']:.1f}m")
    
    # Return the center of the nearest large cluster
    return (round(nearest_cluster['center'][0], 1), round(nearest_cluster['center'][1], 1))


def get_smart_exploration_target_aggressive(current_pos, coverage_tracker):
    """
    AGGRESSIVE exploration that finds ANY unvisited area
    """
    coverage_map = coverage_tracker.coverage_map
    resolution = coverage_tracker.resolution
    origin = [-coverage_tracker.area_width/2, -coverage_tracker.area_height/2]
    
    current_coverage = coverage_tracker.get_coverage_percentage()['coverage_percentage']
    print(f"🎯 AGGRESSIVE exploration - Coverage: {current_coverage:.1f}%")
    
    # For high coverage, use very aggressive parameters
    if current_coverage > 85.0:
        strategies = [
            # Strategy 1: Small areas
            {'min_cluster_size': 5, 'min_area_threshold': 0.1, 'wall_buffer': 0.3},
            # Strategy 2: Tiny areas  
            {'min_cluster_size': 3, 'min_area_threshold': 0.05, 'wall_buffer': 0.2},
            # Strategy 3: Any unvisited pixel
            {'min_cluster_size': 1, 'min_area_threshold': 0.01, 'wall_buffer': 0.15},
            # Strategy 4: Emergency - no buffer
            {'min_cluster_size': 1, 'min_area_threshold': 0.001, 'wall_buffer': 0.05},
        ]
    else:
        # Normal coverage - use original strategy
        strategies = [
            {'min_cluster_size': 20, 'min_area_threshold': 0.5, 'wall_buffer': 0.5},
            {'min_cluster_size': 10, 'min_area_threshold': 0.2, 'wall_buffer': 0.3},
        ]
    
    for i, strategy in enumerate(strategies):
        print(f"   Strategy {i+1}: cluster≥{strategy['min_cluster_size']}, area≥{strategy['min_area_threshold']}m², buffer={strategy['wall_buffer']}m")
        
        target_xy = find_nearest_unvisited_large_area(
            current_pos, coverage_map, resolution, origin,
            min_cluster_size=strategy['min_cluster_size'],
            min_area_threshold=strategy['min_area_threshold'],
            wall_buffer=strategy['wall_buffer']
        )
        
        if target_xy is not None:
            print(f"   ✅ Strategy {i+1} SUCCESS: found target [{target_xy[0]:.1f}, {target_xy[1]:.1f}]")
            return target_xy
        else:
            print(f"   ❌ Strategy {i+1} failed")
    
    # LAST RESORT: Manual scan for ANY unvisited cell
    print(f"   🚨 ALL STRATEGIES FAILED - Manual scan for any unvisited area...")
    return manual_find_unvisited_area(current_pos, coverage_tracker)


def manual_find_unvisited_area(current_pos, coverage_tracker):
    """
    EMERGENCY: Manually scan coverage map for ANY unvisited area
    """
    coverage_map = coverage_tracker.coverage_map
    resolution = coverage_tracker.resolution
    origin = [-coverage_tracker.area_width/2, -coverage_tracker.area_height/2]
    
    print(f"🔍 Manual scan: coverage_map shape {coverage_map.shape}, resolution {resolution}m")
    
    # Find ALL cells with low coverage
    low_coverage_threshold = 30  # Anything less than 30% coverage
    unvisited_mask = coverage_map < low_coverage_threshold
    unvisited_coords = np.where(unvisited_mask)
    
    print(f"   Found {len(unvisited_coords[0])} cells with <{low_coverage_threshold}% coverage")
    
    if len(unvisited_coords[0]) == 0:
        # Try even lower threshold
        low_coverage_threshold = 80
        unvisited_mask = coverage_map < low_coverage_threshold
        unvisited_coords = np.where(unvisited_mask)
        print(f"   Trying <{low_coverage_threshold}% threshold: Found {len(unvisited_coords[0])} cells")
    
    if len(unvisited_coords[0]) == 0:
        print(f"   ❌ No unvisited areas found in manual scan")
        return None
    
    # Find nearest unvisited cell
    min_distance = float('inf')
    best_target = None
    
    for i in range(len(unvisited_coords[0])):
        y, x = unvisited_coords[0][i], unvisited_coords[1][i]
        world_x = x * resolution + origin[0]
        world_y = y * resolution + origin[1]
        
        # Very minimal wall check
        min_wall_dist = min(
            abs(world_x + coverage_tracker.area_width/2),
            abs(world_x - coverage_tracker.area_width/2),
            abs(world_y + coverage_tracker.area_height/2),
            abs(world_y - coverage_tracker.area_height/2)
        )
        
        if min_wall_dist > 0.2:  # Very minimal 0.2m buffer
            distance = np.linalg.norm(np.array([world_x, world_y]) - current_pos[:2])
            if distance < min_distance:
                min_distance = distance
                best_target = (world_x, world_y)
    
    if best_target is not None:
        print(f"   ✅ Manual scan found target: [{best_target[0]:.1f}, {best_target[1]:.1f}] (dist: {min_distance:.1f}m)")
        return best_target
    else:
        print(f"   ❌ Manual scan found no safe targets")
        return None



def find_best_exploration_point_in_cluster(current_pos, coverage_map, resolution=0.1, origin=[-7.5, -7.5],
                                          min_cluster_size=50, strategy='nearest_edge'):
    """
    Find the best exploration point within a large unvisited cluster.
    
    Args:
        strategy: 'nearest_edge', 'cluster_center', or 'furthest_unvisited'
    """
    from scipy.ndimage import label, center_of_mass, distance_transform_edt
    import numpy as np
    
    h, w = coverage_map.shape
    unvisited_mask = coverage_map < 10
    labeled_array, num_clusters = label(unvisited_mask)
    
    if num_clusters == 0:
        return None
    
    # Find the largest cluster or nearest large cluster
    large_clusters = []
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = (labeled_array == cluster_id)
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size >= min_cluster_size:
            cluster_center_ij = center_of_mass(cluster_mask)
            center_x = cluster_center_ij[1] * resolution + origin[0]
            center_y = cluster_center_ij[0] * resolution + origin[1]
            distance = np.linalg.norm(np.array([center_x, center_y]) - current_pos[:2])
            
            large_clusters.append({
                'center_ij': cluster_center_ij,
                'center_xy': (center_x, center_y),
                'distance': distance,
                'size': cluster_size,
                'mask': cluster_mask,
                'cluster_id': cluster_id
            })
    
    if not large_clusters:
        return None
    
    # Choose cluster based on strategy
    if strategy == 'nearest':
        target_cluster = min(large_clusters, key=lambda c: c['distance'])
    else:  # 'largest'
        target_cluster = max(large_clusters, key=lambda c: c['size'])
    
    cluster_mask = target_cluster['mask']
    
    # Find best point within the cluster
    if strategy == 'cluster_center':
        # Use cluster center
        target_point = target_cluster['center_xy']
    
    elif strategy == 'nearest_edge':
        # Find the point in cluster closest to current position
        cluster_points = np.where(cluster_mask)
        min_dist = float('inf')
        best_point_ij = None
        
        for i in range(len(cluster_points[0])):
            iy, ix = cluster_points[0][i], cluster_points[1][i]
            x = ix * resolution + origin[0]
            y = iy * resolution + origin[1]
            dist = np.linalg.norm(np.array([x, y]) - current_pos[:2])
            
            if dist < min_dist:
                min_dist = dist
                best_point_ij = (iy, ix)
        
        if best_point_ij:
            target_point = (
                best_point_ij[1] * resolution + origin[0],
                best_point_ij[0] * resolution + origin[1]
            )
        else:
            target_point = target_cluster['center_xy']
    
    elif strategy == 'furthest_unvisited':
        # Find the point furthest from any visited area (cluster interior)
        distance_map = distance_transform_edt(cluster_mask)
        max_distance_ij = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        
        target_point = (
            max_distance_ij[1] * resolution + origin[0],
            max_distance_ij[0] * resolution + origin[1]
        )
    
    else:
        target_point = target_cluster['center_xy']
    
    return (round(target_point[0], 1), round(target_point[1], 1))


def visualize_exploration_clusters(coverage_map, large_clusters=None, current_pos=None, resolution=0.1, origin=[-7.5, -7.5]):
    """
    Visualize unvisited clusters for debugging (optional)
    """
    import cv2
    import numpy as np
    
    # Create visualization
    vis_map = coverage_map.copy()
    vis_map = cv2.normalize(vis_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    
    if large_clusters:
        # Draw cluster centers
        for i, cluster in enumerate(large_clusters):
            center_x, center_y = cluster['center']
            # Convert to pixel coordinates
            px = int((center_x - origin[0]) / resolution)
            py = int((center_y - origin[1]) / resolution)
            
            if 0 <= px < vis_map.shape[1] and 0 <= py < vis_map.shape[0]:
                cv2.circle(vis_map, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(vis_map, f"{cluster['area']:.0f}m²", (px+10, py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if current_pos:
        # Draw current position
        pos_px = int((current_pos[0] - origin[0]) / resolution)
        pos_py = int((current_pos[1] - origin[1]) / resolution)
        if 0 <= pos_px < vis_map.shape[1] and 0 <= pos_py < vis_map.shape[0]:
            cv2.circle(vis_map, (pos_px, pos_py), 8, (0, 0, 255), 2)
    
    cv2.imshow("Exploration Clusters", vis_map)
    cv2.waitKey(1)


def get_nadir_camera_image(positions, drone_quat, fov=60, resolution=(320, 240), near=0.1, far=100, physicsClientId=None):
    width, height = resolution
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=positions,
        distance=0.001,
        yaw=0,
        pitch=-90,
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
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=physicsClientId
    )
    
    rgb_array = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    return rgb_array

def detect_crack(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] != 3:
        return image

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask_black = cv2.inRange(image, lower_black, upper_black)

    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "Crack", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Simulation stopped by user (Ctrl+C)')
    sys.exit(0)
def check_for_escape_key():
    """Check if ESC key is pressed in PyBullet GUI"""
    try:
        keys = p.getKeyboardEvents()
        # Use the correct PyBullet constant for ESC key
        if keys.get(27):  # 27 is the ASCII code for ESC key
            return True
        return False
    except:
        return False
    
def extract_all_obstacle_distances(sensor_data):
    """Extract all obstacle distances from sensor data"""
    distances = sensor_data.get('distances', [])
    
    if not distances:
        return float('inf'), float('inf'), "no_obstacles"
    
    # Filter out max range readings (no obstacles detected)
    actual_distances = [d for d in distances if d < 3.0]  # 3.0 is SENSOR_RANGE
    
    if not actual_distances:
        return float('inf'), float('inf'), "no_close_obstacles"
    
    min_distance = min(actual_distances)
    avg_distance = sum(actual_distances) / len(actual_distances)
    
    # Format all distances as string for Excel storage
    all_distances_str = ";".join([f"{d:.3f}" for d in actual_distances])
    
  
    return min_distance, avg_distance, all_distances_str



def get_speed_zone_factor(current_pos, target_pos, sensor_data):
    """
    Determine speed factor based on current location and situation
    
    Returns:
        float: Speed factor (0.3 to 1.0)
    """
    # Distance to target
    dist_to_target = np.linalg.norm(target_pos - current_pos)
    
    # Distance to nearest obstacle
    min_obstacle_dist = float('inf')
    obstacles = sensor_data.get('obstacles', [])
    if obstacles:
        for obs in obstacles:
            if len(obs) >= 3:
                obs_pos = np.array(obs[:3])
                dist = np.linalg.norm(current_pos - obs_pos)
                min_obstacle_dist = min(min_obstacle_dist, dist)
    
    # Threat level from sensor
    threat_level = sensor_data.get('threat_level', 0.0)
    
    # Speed zone determination
    if min_obstacle_dist < 0.5 or threat_level > 0.8:
        return 0.3  # Very slow near obstacles
    elif min_obstacle_dist < 1.0 or threat_level > 0.5:
        return 0.6  # Moderate speed near obstacles
    elif dist_to_target < 1.0:
        return 0.7  # Slow down approaching target
    else:
        return 1.0  # Full speed in open areas
    

class LimitedDataCollector:
    def __init__(self, max_entries=None):
        self.max_entries = max_entries
        self.data = {
            'timestamp': [],
            'simulation_time': [],
            'coverage_percentage': [],
            'drone_positions': [],  # Will be initialized later
            'drone_batteries': [],  # Will be initialized later
            'drone_speeds': [],     # Will be initialized later
            'threat_levels': [],    # Will be initialized later
            'total_distance_traveled': [],  # Will be initialized later
            'Energy_Consumed_J': [],        # Will be initialized later
            'min_obstacle_distance': [],    # Will be initialized later
            'avg_obstacle_distance': [],    # Will be initialized later
            'all_obstacle_distances': [],   # Will be initialized later
            'drone_mode_status' : [],
            'overlap_detected': [],           # Boolean: True if any overlap detected this frame
            'active_overlap_count': [],       # Number of active overlaps
            'total_overlap_events': [],       # Cumulative number of overlap events
            'overlap_area_percentage': [],    # Percentage of covered area that overlapped
        }
        self.initialized = False
    
    def initialize_drone_arrays(self, num_drones):
        """Initialize drone-specific arrays once we know the number of drones"""
        if not self.initialized:
            self.data['drone_positions'] = [[] for _ in range(num_drones)]
            self.data['drone_batteries'] = [[] for _ in range(num_drones)]
            self.data['drone_speeds'] = [[] for _ in range(num_drones)]
            self.data['threat_levels'] = [[] for _ in range(num_drones)]
            self.data['total_distance_traveled'] = [[] for _ in range(num_drones)]
            self.data['Energy_Consumed_J'] = [[] for _ in range(num_drones)]
            self.data['min_obstacle_distance'] = [[] for _ in range(num_drones)]
            self.data['avg_obstacle_distance'] = [[] for _ in range(num_drones)]
            self.data['all_obstacle_distances'] = [[] for _ in range(num_drones)]
            self.data['drone_mode_status'] = [[] for _ in range(num_drones)]
            self.initialized = True
    
    def add_data_point(self, timestamp, sim_time, coverage, positions, batteries, speeds, 
                  threat_levels, distances, energy_consumed, min_obs_dist, avg_obs_dist, all_obs_dist, drone_modes,
                  overlap_data):
        """Add a data point and manage memory limits"""
        
        # Add new data
        self.data['timestamp'].append(timestamp)
        self.data['simulation_time'].append(sim_time)
        self.data['coverage_percentage'].append(coverage)
        self.data['overlap_detected'].append(overlap_data.get('overlap_detected', False))
        self.data['active_overlap_count'].append(overlap_data.get('active_overlap_count', 0))
        self.data['total_overlap_events'].append(overlap_data.get('total_overlap_events', 0))
        self.data['overlap_area_percentage'].append(overlap_data.get('overlap_area_percentage', 0.0))
        
        # Add drone-specific data
        num_drones = len(self.data['drone_positions'])
        for j in range(num_drones):
            if j < len(positions):
                self.data['drone_positions'][j].append(f"{positions[j][0]:.2f},{positions[j][1]:.2f},{positions[j][2]:.2f}")
                self.data['drone_batteries'][j].append(batteries[j] if j < len(batteries) else 0)
                self.data['drone_speeds'][j].append(speeds[j] if j < len(speeds) else 0)
                self.data['threat_levels'][j].append(threat_levels[j] if j < len(threat_levels) else 0)
                self.data['total_distance_traveled'][j].append(distances[j] if j < len(distances) else 0)
                self.data['Energy_Consumed_J'][j].append(energy_consumed[j] if j < len(energy_consumed) else 0)
                self.data['min_obstacle_distance'][j].append(min_obs_dist[j] if j < len(min_obs_dist) else 0)
                self.data['avg_obstacle_distance'][j].append(avg_obs_dist[j] if j < len(avg_obs_dist) else 0)
                self.data['all_obstacle_distances'][j].append(all_obs_dist[j] if j < len(all_obs_dist) else "")
                self.data['drone_mode_status'][j].append(drone_modes[j] if j < len(drone_modes) else 0)

        # ===== ADD MONITORING INSTEAD =====
        if len(self.data['timestamp']) % 500 == 0:  # Every 500 data points
            total_entries = len(self.data['timestamp'])
            print(f"📊 Data collected: {total_entries} entries (ALL PRESERVED for thesis)")
        
        # Clean up old data if too large
        """if len(self.data['timestamp']) > self.max_entries:
            remove_count = 50  # Remove 50 oldest entries
            
            # Clean up general data
            self.data['timestamp'] = self.data['timestamp'][remove_count:]
            self.data['simulation_time'] = self.data['simulation_time'][remove_count:]
            self.data['coverage_percentage'] = self.data['coverage_percentage'][remove_count:]
            
            # Clean up drone-specific data
            for j in range(num_drones):
                self.data['drone_positions'][j] = self.data['drone_positions'][j][remove_count:]
                self.data['drone_batteries'][j] = self.data['drone_batteries'][j][remove_count:]
                self.data['drone_speeds'][j] = self.data['drone_speeds'][j][remove_count:]
                self.data['threat_levels'][j] = self.data['threat_levels'][j][remove_count:]
                self.data['total_distance_traveled'][j] = self.data['total_distance_traveled'][j][remove_count:]
                self.data['Energy_Consumed_J'][j] = self.data['Energy_Consumed_J'][j][remove_count:]
                self.data['min_obstacle_distance'][j] = self.data['min_obstacle_distance'][j][remove_count:]
                self.data['avg_obstacle_distance'][j] = self.data['avg_obstacle_distance'][j][remove_count:]
                self.data['all_obstacle_distances'][j] = self.data['all_obstacle_distances'][j][remove_count:]
            
            print(f"🧹 Data cleanup: removed {remove_count} old entries, keeping {len(self.data['timestamp'])}")"""

    def initialize_control_analysis_arrays(self, num_drones):
        """Initialize arrays for control algorithm analysis"""
        self.pid_data = {j: {
            'altitude_error': [],
            'altitude_command': [],
            'speed_error': [],
            'speed_command': [],
            'target_speed': [],
            'position_error': [],
            'distance_to_waypoint': [],
            'speed_context': [],
            'speed_factor': []
        } for j in range(num_drones)}
        
        self.potential_field_data = {j: {
            'total_force_magnitude': [],
            'min_obstacle_distance': [],
            'near_miss_event': [],
            'stuck_event': [],
            'threat_level': []
        } for j in range(num_drones)}

    def add_pid_data(self, drone_id, pid_data):
        """Add PID performance data"""
        if hasattr(self, 'pid_data'):
            for key, value in pid_data.items():
                if key in self.pid_data[drone_id]:
                    self.pid_data[drone_id][key].append(value)
                else:
                    print(f"⚠️ Warning: Key '{key}' not found in PID data structure for drone {drone_id}")
        
        else:
            print(f"⚠️ Warning: No PID data structure or invalid drone_id {drone_id}")

    def add_potential_field_data(self, drone_id, pf_data):
        """Add potential field performance data"""
        if hasattr(self, 'potential_field_data'):
            for key, value in pf_data.items():
                if key in self.potential_field_data[drone_id]:
                    self.potential_field_data[drone_id][key].append(value)

class SensorCache:
    def __init__(self, cache_duration=5):
        self.cache = {}
        self.cache_times = {}
        self.cache_duration = cache_duration
        
    def get_sensor_data(self, pyb_client, drone_pos, drone_vel, iteration):
        # Create cache key from rounded position
        cache_key = f"{int(drone_pos[0]*5)},{int(drone_pos[1]*5)},{int(drone_pos[2]*5)}"
        
        # Return cached data if recent
        if (cache_key in self.cache and 
            iteration - self.cache_times[cache_key] < self.cache_duration):
            return self.cache[cache_key]
        
        # Get fresh sensor data
        sensor_data = get_enhanced_sensor_readings(pyb_client, drone_pos, drone_vel)
        
        # Cache the result
        self.cache[cache_key] = sensor_data
        self.cache_times[cache_key] = iteration
        
        # Cleanup old cache entries every 100 iterations
        if iteration % 100 == 0:
            old_keys = [k for k, t in self.cache_times.items() if iteration - t > self.cache_duration * 5]
            for key in old_keys:
                del self.cache[key]
                del self.cache_times[key]
        
        return sensor_data
    
def create_backup_csv(output_folder, num_drones):
    """Create backup CSV to ensure no data loss"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_folder}/thesis_backup_{num_drones}drones_{timestamp}.csv"
    
    headers = ['timestamp', 'simulation_time', 'coverage_percentage']
    for j in range(num_drones):
        headers.extend([f'drone_{j}_battery', f'drone_{j}_speed', f'drone_{j}_distance'])
    
    with open(csv_filename, 'w') as f:
        f.write(','.join(headers) + '\n')
    
    print(f"📝 Backup CSV created: {csv_filename}")
    return csv_filename

def save_to_backup_csv(csv_filename, timestamp, sim_time, coverage, batteries, speeds, distances, num_drones):
    """Save data point to backup CSV"""
    try:
        row = [timestamp.strftime("%Y-%m-%d %H:%M:%S"), sim_time, coverage]
        for j in range(num_drones):
            row.extend([
                batteries[j] if j < len(batteries) else 0,
                speeds[j] if j < len(speeds) else 0,
                distances[j] if j < len(distances) else 0
            ])
        
        with open(csv_filename, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')
    except:
        pass

class CoordinatedExplorationManager:
    def __init__(self, area_width, area_height, num_drones, min_separation=3.0):
        self.area_width = area_width
        self.area_height = area_height
        self.num_drones = num_drones
        self.min_separation = min_separation  # Minimum distance between drone targets
        
        # Track assigned exploration targets
        self.assigned_targets = {}  # drone_id -> target_position
        self.target_assignments = {}  # target_key -> drone_id
        self.exploration_zones = {}  # drone_id -> assigned_zone
        
        # Create exploration zones for each drone
        self._create_exploration_zones()
        
        print(f"🎯 Coordinated Exploration Manager initialized:")
        print(f"   Area: {area_width}x{area_height}m")
        print(f"   Drones: {num_drones}")
        print(f"   Minimum separation: {min_separation}m")
        print(f"   Exploration zones created")
    
    def _create_exploration_zones(self):
        """Create dedicated exploration zones for each drone"""
        # Divide the area into zones based on number of drones
        if self.num_drones == 1:
            # Single drone gets entire area
            self.exploration_zones[0] = {
                'x_min': -self.area_width/2, 'x_max': self.area_width/2,
                'y_min': -self.area_height/2, 'y_max': self.area_height/2
            }
        elif self.num_drones == 2:
            # Split vertically
            self.exploration_zones[0] = {
                'x_min': -self.area_width/2, 'x_max': 0,
                'y_min': -self.area_height/2, 'y_max': self.area_height/2
            }
            self.exploration_zones[1] = {
                'x_min': 0, 'x_max': self.area_width/2,
                'y_min': -self.area_height/2, 'y_max': self.area_height/2
            }
        elif self.num_drones == 4:
            # Split into quadrants
            zones = [
                {'x_min': -self.area_width/2, 'x_max': 0, 'y_min': 0, 'y_max': self.area_height/2},  # Top-left
                {'x_min': 0, 'x_max': self.area_width/2, 'y_min': 0, 'y_max': self.area_height/2},   # Top-right
                {'x_min': -self.area_width/2, 'x_max': 0, 'y_min': -self.area_height/2, 'y_max': 0}, # Bottom-left
                {'x_min': 0, 'x_max': self.area_width/2, 'y_min': -self.area_height/2, 'y_max': 0}  # Bottom-right
            ]
            for i in range(4):
                self.exploration_zones[i] = zones[i]
        else:

            # For other numbers, create grid-based zones
            rows = int(np.ceil(np.sqrt(self.num_drones)))
            cols = int(np.ceil(self.num_drones / rows))
            
            zone_width = self.area_width / cols
            zone_height = self.area_height / rows
            
            for i in range(self.num_drones):
                row = i // cols
                col = i % cols
                
                self.exploration_zones[i] = {
                    'x_min': -self.area_width/2 + col * zone_width,
                    'x_max': -self.area_width/2 + (col + 1) * zone_width,
                    'y_min': -self.area_height/2 + row * zone_height,
                    'y_max': -self.area_height/2 + (row + 1) * zone_height
                }
        
        # Print zone assignments
        for drone_id, zone in self.exploration_zones.items():
            print(f"   Drone {drone_id}: Zone [{zone['x_min']:.1f},{zone['y_min']:.1f}] to [{zone['x_max']:.1f},{zone['y_max']:.1f}]")
    
    def get_coordinated_exploration_target(self, drone_id, current_pos, coverage_tracker, other_drone_positions):
        """Get exploration target that doesn't conflict with other drones"""
        
        # Check if drone already has a valid target
        if drone_id in self.assigned_targets:
            existing_target = self.assigned_targets[drone_id]
            
            # Check if target is still valid (not too close to other drones)
            if self._is_target_safe(existing_target, drone_id, other_drone_positions):
                # Check if target area still needs coverage
                if self._needs_coverage(existing_target, coverage_tracker):
                    return existing_target
            
            # Target is no longer valid, remove it
            self._remove_target_assignment(drone_id)
        
        # Get drone's assigned exploration zone
        if drone_id not in self.exploration_zones:
            print(f"⚠️ No exploration zone for drone {drone_id}")
            return None
        
        zone = self.exploration_zones[drone_id]
        
        # Find unvisited areas within the drone's zone
        target = self._find_target_in_zone(drone_id, zone, current_pos, coverage_tracker, other_drone_positions)
        
        if target is not None:
            self._assign_target(drone_id, target)
            print(f"🎯 Drone {drone_id} assigned coordinated target: {target}")
        else:
            print(f"🔍 Drone {drone_id} found no valid targets in assigned zone")
            # Try neighboring zones if own zone is complete
            target = self._find_target_in_neighboring_zones(drone_id, current_pos, coverage_tracker, other_drone_positions)
            if target is not None:
                self._assign_target(drone_id, target)
                print(f"🎯 Drone {drone_id} assigned neighboring zone target: {target}")
        
        return target
    
    def _find_target_in_zone(self, drone_id, zone, current_pos, coverage_tracker, other_drone_positions):
        """Find exploration target within specific zone"""
        # Get coverage map parameters
        coverage_map = coverage_tracker.coverage_map
        resolution = coverage_tracker.resolution
        origin = [-coverage_tracker.area_width/2, -coverage_tracker.area_height/2]
        
        # Convert zone boundaries to grid coordinates
        x_min_grid = max(0, int((zone['x_min'] - origin[0]) / resolution))
        x_max_grid = min(coverage_map.shape[1], int((zone['x_max'] - origin[0]) / resolution))
        y_min_grid = max(0, int((zone['y_min'] - origin[1]) / resolution))
        y_max_grid = min(coverage_map.shape[0], int((zone['y_max'] - origin[1]) / resolution))
        
        # Extract zone coverage map
        zone_coverage = coverage_map[y_min_grid:y_max_grid, x_min_grid:x_max_grid]
        
        # Find unvisited areas in zone
        unvisited_mask = zone_coverage < 10  # Areas with less than 10% coverage
        
        if not np.any(unvisited_mask):
            return None  # Zone is fully covered
        
        # Find connected components of unvisited areas
        from scipy.ndimage import label, center_of_mass
        labeled_array, num_components = label(unvisited_mask)
        
        if num_components == 0:
            return None
        
        # Find the best target area
        best_target = None
        best_score = -1
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_array == component_id)
            component_size = np.sum(component_mask)
            
            # Skip very small areas
            if component_size < 20:
                continue
            
            # Calculate center of mass
            center_ij = center_of_mass(component_mask)
            
            # Convert to world coordinates
            center_x = (center_ij[1] + x_min_grid) * resolution + origin[0]
            center_y = (center_ij[0] + y_min_grid) * resolution + origin[1]
            target_pos = np.array([center_x, center_y])
            
            # Check if target is safe (not too close to other drones)
            if not self._is_target_safe(target_pos, drone_id, other_drone_positions):
                continue
            
            # Calculate score based on area size and distance
            distance_to_drone = np.linalg.norm(target_pos - current_pos[:2])
            score = component_size / (distance_to_drone + 1)  # Prefer larger, closer areas
            
            if score > best_score:
                best_score = score
                best_target = target_pos
        
        return best_target
    
    def _find_target_in_neighboring_zones(self, drone_id, current_pos, coverage_tracker, other_drone_positions):
        """Find target in neighboring zones if own zone is complete"""
        # Try zones of other drones that might need help
        for other_drone_id, zone in self.exploration_zones.items():
            if other_drone_id == drone_id:
                continue
            
            # Check if other drone is far from their zone
            if other_drone_id < len(other_drone_positions):
                other_pos = other_drone_positions[other_drone_id]
                zone_center = np.array([
                    (zone['x_min'] + zone['x_max']) / 2,
                    (zone['y_min'] + zone['y_max']) / 2
                ])
                
                # If other drone is far from their zone, help them
                if np.linalg.norm(other_pos[:2] - zone_center) > 5.0:
                    target = self._find_target_in_zone(drone_id, zone, current_pos, coverage_tracker, other_drone_positions)
                    if target is not None:
                        return target
        
        return None
    
    def _is_target_safe(self, target_pos, requesting_drone_id, other_drone_positions):
        """Check if target is safe (not too close to other drones or their targets)"""
        # Check distance to other drones
        for i, other_pos in enumerate(other_drone_positions):
            if i == requesting_drone_id:
                continue
            
            distance = np.linalg.norm(target_pos - other_pos[:2])
            if distance < self.min_separation:
                return False
        
        # Check distance to other assigned targets
        for other_drone_id, other_target in self.assigned_targets.items():
            if other_drone_id == requesting_drone_id:
                continue
            
            distance = np.linalg.norm(target_pos - other_target[:2])
            if distance < self.min_separation:
                return False
        
        return True
    
    def _needs_coverage(self, target_pos, coverage_tracker):
        """Check if target area still needs coverage"""
        coverage_map = coverage_tracker.coverage_map
        resolution = coverage_tracker.resolution
        origin = [-coverage_tracker.area_width/2, -coverage_tracker.area_height/2]
        
        # Convert to grid coordinates
        grid_x = int((target_pos[0] - origin[0]) / resolution)
        grid_y = int((target_pos[1] - origin[1]) / resolution)
        
        # Check bounds
        if (0 <= grid_x < coverage_map.shape[1] and 
            0 <= grid_y < coverage_map.shape[0]):
            
            # Check local area around target
            x_min = max(0, grid_x - 3)
            x_max = min(coverage_map.shape[1], grid_x + 4)
            y_min = max(0, grid_y - 3)
            y_max = min(coverage_map.shape[0], grid_y + 4)
            
            local_patch = coverage_map[y_min:y_max, x_min:x_max]
            avg_coverage = np.mean(local_patch) if local_patch.size > 0 else 100
            
            return avg_coverage < 80  # Need coverage if less than 80%
        
        return False
    
    def _assign_target(self, drone_id, target_pos):
        """Assign target to drone"""
        # Remove any previous assignment
        self._remove_target_assignment(drone_id)
        
        # Assign new target
        self.assigned_targets[drone_id] = target_pos
        target_key = f"{target_pos[0]:.1f},{target_pos[1]:.1f}"
        self.target_assignments[target_key] = drone_id
    
    def _remove_target_assignment(self, drone_id):
        """Remove target assignment for drone"""
        if drone_id in self.assigned_targets:
            old_target = self.assigned_targets[drone_id]
            old_key = f"{old_target[0]:.1f},{old_target[1]:.1f}"
            
            del self.assigned_targets[drone_id]
            if old_key in self.target_assignments:
                del self.target_assignments[old_key]
    
    def clear_target(self, drone_id):
        """Clear target when drone reaches it"""
        self._remove_target_assignment(drone_id)
        print(f"✅ Drone {drone_id} target cleared - ready for new assignment")
    
    def get_exploration_status(self):
        """Get status of exploration coordination"""
        return {
            'assigned_targets': len(self.assigned_targets),
            'active_assignments': dict(self.assigned_targets),
            'zones': dict(self.exploration_zones)
        }
    

# Add this simple function to your code (anywhere after your imports)
def draw_simple_waypoint_trails(drone_positions, target_positions, drone_colors, pyb_client):
    """
    Draw simple trails showing drone destinations
    
    Args:
        drone_positions: List of current drone positions [[x,y,z], ...]
        target_positions: List of target waypoint positions [[x,y,z], ...]
        drone_colors: List of RGB colors for each drone [[r,g,b], ...]
        pyb_client: PyBullet client ID
    """
    for i, (current_pos, target_pos) in enumerate(zip(drone_positions, target_positions)):
        if target_pos is not None and i < len(drone_colors):
            color = drone_colors[i]
            
            # Draw line from drone to destination
            p.addUserDebugLine(
                current_pos,
                target_pos,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=1.0,  # Line disappears after 1 second
                physicsClientId=pyb_client
            )
            
            # Draw simple cross at destination
            cross_size = 0.5
            p.addUserDebugLine(
                [target_pos[0] - cross_size, target_pos[1], target_pos[2]],
                [target_pos[0] + cross_size, target_pos[1], target_pos[2]],
                lineColorRGB=color,
                lineWidth=4,
                lifeTime=1.0,
                physicsClientId=pyb_client
            )
            
            p.addUserDebugLine(
                [target_pos[0], target_pos[1] - cross_size, target_pos[2]],
                [target_pos[0], target_pos[1] + cross_size, target_pos[2]],
                lineColorRGB=color,
                lineWidth=4,
                lifeTime=1.0,
                physicsClientId=pyb_client
            )

def run_simulation_with_gui_waypoints(waypoint_config):
    """
    Run simulation with waypoints generated from GUI - THREAD SAFE VERSION
    """
    
    print(f"🚁 Starting PyBullet simulation with GUI waypoints")
    print(f"   Drones: {waypoint_config['num_drones']}")
    print(f"   Building: {waypoint_config['building_width']}x{waypoint_config['building_height']}m")
    print(f"   Waypoints per drone: {[len(wp) for wp in waypoint_config['waypoints']]}")
    
    # Convert to the format expected by get_waypoint_configuration
    external_waypoints = (
        waypoint_config['num_drones'],
        waypoint_config['building_width'], 
        waypoint_config['building_height'],
        waypoint_config['waypoints'],
        waypoint_config['init_positions'],
        waypoint_config['altitude'],
        waypoint_config.get('building_shape', 'rectangle'),
        waypoint_config.get('starting_formation', 'corner')
    )
    
    # Run simulation with GUI waypoints and thread-safe parameters
    run(
        external_waypoints=external_waypoints,
        gui=True,  # Enable GUI
        duration_sec=1000,  # Shorter duration for testing
        obstacles=True,
        plot=False,  # Disable plotting to avoid thread issues
        record_video=False  # Disable video recording
    )
    
class ControlEvaluationPlotter:
    """Class to handle plotting for control algorithm evaluation"""
    
    def __init__(self, num_drones):
        self.num_drones = num_drones
        self.drone_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
    def plot_pid_speed_performance(self, simulation_data, save_path=None):
        """Plot DYNAMIC target speed vs actual speed for PID evaluation"""
        
        if not hasattr(simulation_data, 'pid_data'):
            print("❌ No PID data available for plotting")
            return
        
        # Single plot instead of 2x2 subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('PID Speed Control Performance Evaluation (Dynamic Target)', fontsize=16, fontweight='bold')
        
        # Plot 1: DYNAMIC Speed Tracking for all drones
        for j in range(self.num_drones):
            if j in simulation_data.pid_data and simulation_data.pid_data[j]['target_speed']:
                time_data = simulation_data.data['simulation_time'][:len(simulation_data.pid_data[j]['target_speed'])]
                
                # Use ACTUAL dynamic target speeds
                target_speeds = simulation_data.pid_data[j]['target_speed']
                actual_speeds = simulation_data.pid_data[j]['speed_command']
                
                ax.plot(time_data, target_speeds, '--', color=self.drone_colors[j], 
                        label=f'Drone {j} Dynamic Target', alpha=0.7, linewidth=2)
                ax.plot(time_data, actual_speeds, '-', color=self.drone_colors[j], 
                        label=f'Drone {j} Actual', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Dynamic Target vs Actual Speed')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/dynamic_pid_performance_evaluation.png", dpi=300, bbox_inches='tight')
            print(f"📊 Dynamic PID performance plot saved: {save_path}/dynamic_pid_performance_evaluation.png")
        
        plt.show()
    
    def plot_obstacle_avoidance_performance(self, simulation_data, min_safe_distance=0.5, save_path=None):
        """Plot obstacle avoidance system performance"""
        
        if not hasattr(simulation_data, 'potential_field_data'):
            print("❌ No potential field data available for plotting")
            return
        
        # Single plot instead of 2x2 subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Obstacle Avoidance System Performance Evaluation', fontsize=16, fontweight='bold')
        
        # Plot 1: Minimum obstacle distance over time
        for j in range(self.num_drones):
            if j in simulation_data.potential_field_data and simulation_data.potential_field_data[j]['min_obstacle_distance']:
                time_data = simulation_data.data['simulation_time'][:len(simulation_data.potential_field_data[j]['min_obstacle_distance'])]
                min_distances = simulation_data.potential_field_data[j]['min_obstacle_distance']
                
                # Filter out infinite values for better visualization
                filtered_distances = [d if d != float('inf') and d < 10 else 10 for d in min_distances]
                
                ax.plot(time_data, filtered_distances, '-', color=self.drone_colors[j], 
                        label=f'Drone {j}', linewidth=2, alpha=0.8)
        
        ax.axhline(y=min_safe_distance, color='red', linestyle='--', linewidth=2, 
                   label=f'Min Safe Distance ({min_safe_distance}m)')
        ax.axhline(y=min_safe_distance*2, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Warning Distance ({min_safe_distance*2}m)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to Nearest Obstacle (m)')
        ax.set_title('Obstacle Distance Monitoring')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/obstacle_avoidance_evaluation.png", dpi=300, bbox_inches='tight')
            print(f"📊 Obstacle avoidance plot saved: {save_path}/obstacle_avoidance_evaluation.png")
        
        plt.show()

# ============================================================================
# MAIN SIMULATION FUNCTION (Modified to use waypoint configuration)
# ============================================================================


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
        colab=DEFAULT_COLAB,
        external_waypoints=None
        ):
    
    # ===== ADD THIS SECTION FIRST (BEFORE YOUR EXISTING CODE) =====
    PERFORMANCE_MODE = True  # Set to False only if you need full visualization
    
    if PERFORMANCE_MODE:
        ENABLE_TRAILS = False
        ENABLE_FORCE_VECTORS = False  
        ENABLE_CAMERA_FOOTPRINTS = False
        ENABLE_OVERLAP_DETECTION = True
        VISUALIZATION_FREQ = 30       # Update every 30 iterations instead of 10
        COVERAGE_FREQ = 60           # Update coverage every 60 iterations  
        STATUS_FREQ = 60             # Update status text every 60 iterations
        EXCEL_FREQ = 300             # Save Excel every 500 iterations instead of 100
    else:
        ENABLE_TRAILS = True
        ENABLE_FORCE_VECTORS = True
        ENABLE_CAMERA_FOOTPRINTS = True
        ENABLE_OVERLAP_DETECTION = True
        VISUALIZATION_FREQ = 10
        COVERAGE_FREQ = 10
        STATUS_FREQ = 20
        EXCEL_FREQ = 100
    
    print(f"🎚️ Performance mode: {'ENABLED' if PERFORMANCE_MODE else 'DISABLED'}")

    
    # ===== NOW YOUR EXISTING CODE CONTINUES UNCHANGED =====
    NUM_DRONES, BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT, WAYPOINTS, INIT_XYZS, ALTITUDE = get_waypoint_configuration(external_waypoints)

    # Screenshot configuration
    building_shape = getattr(get_waypoint_configuration, 'building_shape', 'rectangle')
    starting_formation = getattr(get_waypoint_configuration, 'starting_formation', 'corner')
    screenshot_run_number = 1  # Will be updated when we determine the run number
    screenshot_captured_start = False
    
    print(f"🚁 Starting Swarm Simulation with Enhanced Stuck Detection & Battery System")
    print(f"Drones: {NUM_DRONES} | Building Size: {BUILDING_SIZE_WIDTH} X {BUILDING_SIZE_HEIGHT}m | Altitude: {ALTITUDE}m")
    print(f"Waypoints per drone: {[len(wp) for wp in WAYPOINTS]}")

    # Initialize data collector
    simulation_data = LimitedDataCollector(max_entries=None)
    simulation_data.initialize_drone_arrays(NUM_DRONES)  # Initialize with correct number of drones
    simulation_data.initialize_control_analysis_arrays(NUM_DRONES)

    backup_csv = create_backup_csv(output_folder, NUM_DRONES)
    
    
    PHY = Physics.PYB

    

    env = EnhancedVelocityAviary(
        drone_model=drone,
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

    # Get environment info for debugging
    print(f"🚁 Using Enhanced VelocityAviary:")
    print(f"   Speed limit: {env.get_speed_limit():.2f} m/s")

    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    # ===== FOLLOW CAMERA INITIALIZATION =====
    follow_camera = None
    if gui:  # Remove the ENABLE_FOLLOW_CAMERA check here
        follow_camera = DroneFollowCamera(
            target_drone_id=0,  # Follow drone 0 by default
            distance=5.0,
            height_offset=2.0,
            smooth_factor=0.1
        )
        print(f"📹 Follow camera initialized - Following Drone 0")
        print("📹 Camera controls:")
        print("   Keys 1-4: Switch between drones")
        print("   Key 5: Toggle follow camera on/off")

    # Define drone marker colors
    DRONE_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    if gui:
        max_dimension = max(BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT)
        p.resetDebugVisualizerCamera(
            cameraDistance=max_dimension * 0.5,
            cameraYaw=0,
            cameraPitch=-89.99,
            cameraTargetPosition=[0,0,0],
            physicsClientId=PYB_CLIENT
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=NUM_DRONES,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    # Create obstacles along the path
    obstacle_ids = []
    if obstacles:
        obstacle_ids = create_obstacles(PYB_CLIENT, WAYPOINTS, 
                                      num_obstacles=max(2, NUM_DRONES),
                                      size=1)

    # ===== ENHANCED WAYPOINT MANAGEMENT =====
    # Initialize waypoint managers for each drone
    waypoint_managers = [
        WaypointManager(drone_id=i, waypoints=WAYPOINTS[i]) 
        for i in range(NUM_DRONES)
    ]

    if gui:
        print("🎯 Initializing permanent waypoint visualization...")
        
        # Create permanent arrows and markers
        waypoint_visualization = create_enhanced_waypoint_visualization(waypoint_managers, PYB_CLIENT)
        
        print("✅ Permanent waypoint arrows and markers created")
        print("   🏹 Arrows show direction between waypoints")
        print("   🏷️ Labels show waypoint numbers")
        print("   🎯 Markers show waypoint positions")
    else:
        waypoint_visualization = None

    drone_states = [DroneState(ALTITUDE) for _ in range(NUM_DRONES)]

    exploration_manager = CoordinatedExplorationManager(
        area_width=BUILDING_SIZE_WIDTH,
        area_height=BUILDING_SIZE_HEIGHT,
        num_drones=NUM_DRONES,
        min_separation=2.0  # Minimum 3 meters between drone targets
    )

    
    # Enhanced improved controllers
    improved_controllers = [
        ImprovedPotentialFieldController(
            drone_id=i,
            drone_pos=INIT_XYZS[i],
            waypoint=WAYPOINTS[i][0] if len(WAYPOINTS[i]) > 0 else np.zeros(3),
            max_speed=1.5,
            max_force=0.8
        ) for i in range(NUM_DRONES)
    ]

    velocity_adapters = [
        VelocityAviaryAdapter(
            max_horizontal_speed=1.5,  # Adjust as needed
            max_vertical_speed=1.0,    # Adjust as needed  
            max_yaw_rate=0.5          # Adjust as needed
        ) for _ in range(NUM_DRONES)
    ]

    print(f"✅ Created {NUM_DRONES} VelocityAviaryAdapters")

    # Initialize all drones to face forward (0 degrees)
    #initialize_drone_headings(force_adapters, initial_headings=[0, 0, 0, 0])  # All face positive X
    
    # ===== BATTERY SIMULATION INITIALIZATION =====
    battery_simulators = create_battery_simulators(
        NUM_DRONES,
        initial_batteries=100.0,  # All start at 100%
        battery_capacities=350   # 350mAh batteries
    )

    safe_restart_manager = SafeDroneRestartManager(
        num_drones=NUM_DRONES,
        env=env,  # Pass environment reference
        target_altitude=ALTITUDE
    )

    sensor_cache = SensorCache(cache_duration=5)
    
    # Battery tracking
    battery_warnings_issued = [False for _ in range(NUM_DRONES)]


    
    # Updated sensor parameters
    NUM_SENSORS = 8
    SENSOR_RANGE = 3.0
    
    if gui:
        # Draw waypoint paths for each drone in different colors
        colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [0.5,0.5,0.5], [1,0.5,0]]
        for drone_id, waypoints in enumerate(WAYPOINTS):
            color = colors[drone_id % len(colors)]
            for i in range(len(waypoints)-1):
                p.addUserDebugLine(
                    waypoints[i], 
                    waypoints[i+1], 
                    lineColorRGB=color, 
                    lineWidth=3,
                    lifeTime=0,
                    physicsClientId=PYB_CLIENT
                )
                
        # Draw building boundary
        boundary_points = [
            [-BUILDING_SIZE_WIDTH/2, -BUILDING_SIZE_HEIGHT/2, 0],  
            [BUILDING_SIZE_WIDTH/2, -BUILDING_SIZE_HEIGHT/2, 0],   
            [BUILDING_SIZE_WIDTH/2, BUILDING_SIZE_HEIGHT/2, 0],    
            [-BUILDING_SIZE_WIDTH/2, BUILDING_SIZE_HEIGHT/2, 0]    
        ]
        
        for i in range(len(boundary_points)):
            start = boundary_points[i]
            end = boundary_points[(i+1) % len(boundary_points)]
            p.addUserDebugLine(start, end, lineColorRGB=[0,0,0], lineWidth=4, 
                             lifeTime=0, physicsClientId=PYB_CLIENT)
        
    coverage = NadirCoverageTracker(
        area_width=BUILDING_SIZE_WIDTH,
        area_height=BUILDING_SIZE_HEIGHT,
        resolution=0.5,  # CHANGED: 4x fewer pixels for better performance
        focal_length=4e-3,
        sensor_width=6.17e-3,
        sensor_height=4.55e-3
    )

    coverage_visualizer = EnhancedCoverageVisualizer(
        area_width=BUILDING_SIZE_WIDTH,
        area_height=BUILDING_SIZE_HEIGHT,
        resolution=0.5  # CHANGED: Much coarser resolution for performance
    )

    overlap_detector = OverlapDetector(
        area_width=BUILDING_SIZE_WIDTH,
        area_height=BUILDING_SIZE_HEIGHT,
        resolution=0.5,
        time_threshold=1.0  # 1 second threshold for simultaneous overlap
    )

    print(f"📊 Coverage tracking optimized for performance")

    def update_excel_file(simulation_data_obj, stats, num_drones, output_folder, battery_simulators, waypoints_skipped, excel_filename=None):
        """Update a single Excel file throughout the simulation - UPDATED for LimitedDataCollector"""
        try:
            if excel_filename is None:
                # Get configuration info
                building_shape = getattr(get_waypoint_configuration, 'building_shape', 'rectangle')
                starting_position = getattr(get_waypoint_configuration, 'starting_formation', 'corner')
                
                # Create base filename
                base_filename = f"Results_{num_drones}drones_{building_shape}_{starting_position}"
                
                # Find existing files to determine P number
                import glob
                import os
                
                os.makedirs(output_folder, exist_ok=True)
                existing_files = glob.glob(f"{output_folder}/{base_filename}_P*.xlsx")
                
                if existing_files:
                    p_numbers = []
                    for file in existing_files:
                        try:
                            filename = os.path.basename(file)
                            if '_P' in filename:
                                p_part = filename.split('_P')[-1].split('.')[0]
                                p_numbers.append(int(p_part))
                        except (ValueError, IndexError):
                            continue
                    next_p = max(p_numbers) + 1 if p_numbers else 1
                else:
                    next_p = 1
                
                excel_filename = f"{output_folder}/{base_filename}_P{next_p}.xlsx"
                print(f"📊 Excel filename: {excel_filename}")
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='w') as writer:
                # Summary sheet with current stats
                summary_data = {
                    'Metric': ['Current Simulation Time (s)', 'Current Coverage (%)', 'Number of Drones', 
                            'Building Width (m)', 'Building Height (m)', 'Flight Altitude (m)',
                            'Status', 'Last Update', 'Data Points Collected'],
                    'Value': [stats['current_time'], stats['current_coverage'], 
                            stats['num_drones'], stats['building_width'], 
                            stats['building_height'], stats['altitude'],
                            stats.get('status', 'RUNNING'), 
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            len(simulation_data_obj.data['simulation_time'])]  # CHANGED: Access through .data
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Time series data - ALL collected data
                if simulation_data_obj.data['simulation_time']:  # CHANGED: Access through .data
                    max_len = len(simulation_data_obj.data['simulation_time'])
                    time_series_data = {
                        'Timestamp': simulation_data_obj.data['timestamp'][:max_len],           # CHANGED
                        'Simulation_Time_s': simulation_data_obj.data['simulation_time'][:max_len],  # CHANGED
                        'Coverage_Percentage': simulation_data_obj.data['coverage_percentage'][:max_len],  # CHANGED
                        'Overlap_Detected': simulation_data_obj.data['overlap_detected'][:max_len],
                        'Active_Overlap_Count': simulation_data_obj.data['active_overlap_count'][:max_len], 
                        'Total_Overlap_Events': simulation_data_obj.data['total_overlap_events'][:max_len],
                        'Overlap_Area_Percentage': simulation_data_obj.data['overlap_area_percentage'][:max_len]
                    }
                    
                    # Add ALL drone data
                    for j in range(num_drones):
                        if j < len(simulation_data_obj.data['drone_batteries']):  # CHANGED
                            drone_len = len(simulation_data_obj.data['drone_batteries'][j])
                            time_series_data[f'Drone_{j}_Battery_%'] = simulation_data_obj.data['drone_batteries'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Speed_ms'] = simulation_data_obj.data['drone_speeds'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Distance_m'] = simulation_data_obj.data['total_distance_traveled'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Position'] = simulation_data_obj.data['drone_positions'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Energy_Consumed_J'] = simulation_data_obj.data['Energy_Consumed_J'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Min_Obstacle_Dist_m'] = simulation_data_obj.data['min_obstacle_distance'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Avg_Obstacle_Dist_m'] = simulation_data_obj.data['avg_obstacle_distance'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_All_Obstacle_Distances'] = simulation_data_obj.data['all_obstacle_distances'][j][:max_len] + [None]*(max_len-drone_len)
                            time_series_data[f'Drone_{j}_Mode_Status'] = simulation_data_obj.data['drone_mode_status'][j][:max_len] + [None]*(max_len-drone_len)
                                    
                    time_series_df = pd.DataFrame(time_series_data)
                    time_series_df.to_excel(writer, sheet_name='Time_Series_Data', index=False)
                
                # Current drone statistics (unchanged)
                drone_stats = []
                for j in range(num_drones):
                    battery_info = battery_simulators[j].get_battery_info()
                    E_total = 350 * (100 - battery_info['percentage']) / 100.0
                    drone_stats.append({
                        'Drone_ID': j,
                        'Current_Battery_%': battery_info['percentage'],
                        'Flight_Time_s': battery_info['flight_time'],
                        'Distance_Traveled_m': battery_info['distance_traveled'],
                        'Energy_Consumed_J': E_total,
                        'Emergency_Time_s': battery_info['emergency_time'],
                        'Battery_Status': battery_info['status'],
                        'Waypoints_Skipped': waypoints_skipped[j] if j < len(waypoints_skipped) else 0
                    })
                
                drone_df = pd.DataFrame(drone_stats)
                drone_df.to_excel(writer, sheet_name='Drone_Statistics', index=False)
            
            return excel_filename
            
        except Exception as e:
            print(f"❌ Error updating Excel: {e}")
            return excel_filename
        
    def emergency_performance_mode():
        """Activate when performance becomes critical"""
        global PERFORMANCE_MODE, ENABLE_TRAILS, ENABLE_FORCE_VECTORS, ENABLE_CAMERA_FOOTPRINTS
        global VISUALIZATION_FREQ, COVERAGE_FREQ, STATUS_FREQ
        
        print("🚨 EMERGENCY PERFORMANCE MODE ACTIVATED!")
        
        PERFORMANCE_MODE = True
        ENABLE_TRAILS = False
        ENABLE_FORCE_VECTORS = False  
        ENABLE_CAMERA_FOOTPRINTS = False
        VISUALIZATION_FREQ = 60
        COVERAGE_FREQ = 120
        STATUS_FREQ = 120
        
        # Clear all existing debug objects
        try:
            p.removeAllUserDebugItems(physicsClientId=PYB_CLIENT)
            cv2.destroyAllWindows()
        except:
            pass
        
        print("   All visualizations disabled for maximum performance")
    
    try:
        action = np.zeros((NUM_DRONES,4))
        START = time.time()
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, signal_handler)
            else:
                print("⚠️ Running in thread - signal handler disabled")
        except Exception as e:
            print(f"⚠️ Could not set signal handler: {e}")
        simulation_stopped = False
        exploration_targets = [None for _ in range(NUM_DRONES)]
        drone_trails = [[] for _ in range(NUM_DRONES)]
        TRAIL_LENGTH = 20

        run.last_perf_check = time.time()
        run.last_200_check = time.time()

        # Single Excel file setup
        excel_save_interval = 100  # Update Excel every 300 iterations
        excel_filename = None  # Will be set when first file is created

        # Create initial Excel file
        initial_stats = {
            'current_time': 0,
            'current_coverage': 0,
            'num_drones': NUM_DRONES,
            'building_width': BUILDING_SIZE_WIDTH,
            'building_height': BUILDING_SIZE_HEIGHT,
            'altitude': ALTITUDE,
            'status': 'STARTING'
        }

        print("📊 Creating Excel file for continuous logging...")


        # Performance tracking
        # Performance tracking
        threat_levels = [0.0 for _ in range(NUM_DRONES)]
        emergency_activations = [0 for _ in range(NUM_DRONES)]
        waypoints_skipped = [0 for _ in range(NUM_DRONES)]  # ADD THIS LINE
        current_targets = [None for _ in range(NUM_DRONES)]  # ADD THIS LINE (if not already present)
        emergency_activations = [0 for _ in range(NUM_DRONES)]
        waypoints_skipped = [0 for _ in range(NUM_DRONES)]
        current_targets = [None for _ in range(NUM_DRONES)]

        screenshot_captured_start = False
        
        for i in range(0, int(duration_sec*env.CTRL_FREQ)):
            iteration_start = time.time()

            current_simulation_time = i * env.CTRL_TIMESTEP

            if gui and check_for_escape_key():
                print('\n🛑 Simulation stopped by user (ESC key)')
                print('📊 Saving final data to Excel...')
                
                # Update Excel with final data
                final_stats = {
                    'current_time': current_simulation_time,
                    'current_coverage': coverage_visualizer.get_coverage_percentage()['coverage_percentage'],
                    'num_drones': NUM_DRONES,
                    'building_width': BUILDING_SIZE_WIDTH,
                    'building_height': BUILDING_SIZE_HEIGHT,
                    'altitude': ALTITUDE,
                    'status': 'STOPPED_BY_USER'
                }
                
                excel_filename = update_excel_file(
                    simulation_data, final_stats, NUM_DRONES, output_folder, 
                    battery_simulators, waypoints_skipped, excel_filename
                )
                
                simulation_stopped = True
                break

            
            obs, reward, terminated, truncated, info = env.step(action)
            positions = [obs[j][0:3] for j in range(NUM_DRONES)]
            velocities = [obs[j][10:13] if len(obs[j]) > 12 else np.zeros(3) for j in range(NUM_DRONES)]
            altitude = positions[0][2] if positions else ALTITUDE

            # ===== CAMERA CONTROLS AND UPDATE =====
            if gui and follow_camera is not None:
                # Check for camera control keys
                try:
                    keys = p.getKeyboardEvents(physicsClientId=PYB_CLIENT)
                    
                    # Keys 1-4: Switch between drones
                    for key_code in [49, 50, 51, 52]:  # ASCII codes for 1,2,3,4
                        if keys.get(key_code) == p.KEY_WAS_TRIGGERED:
                            drone_id = key_code - 49  # Convert to 0,1,2,3
                            if follow_camera.switch_target_drone(drone_id, NUM_DRONES):
                                pass  # Success message already printed in method
                            else:
                                print(f"❌ Cannot follow Drone {drone_id} (only {NUM_DRONES} drones available)")
                    
                    # Key 5: Toggle follow camera
                    if keys.get(53) == p.KEY_WAS_TRIGGERED:  # ASCII code for '5'
                        follow_camera.toggle_camera()
                
                except Exception as e:
                    pass  # Ignore keyboard errors
                
                # Update camera position
                follow_camera.update_camera(positions, velocities, PYB_CLIENT)


            if i % 10 == 0:
                restart_occurred = safe_restart_manager.check_and_restart_safely(
                    positions, velocities, PYB_CLIENT
                )
                
                if restart_occurred:
                    print(f"🔄 Restart occurred at iteration {i}")


            if gui and i % VISUALIZATION_FREQ == 0:  # Much less frequent updates
                add_drone_markers(PYB_CLIENT, positions, DRONE_COLORS, marker_size=0.4)
                visualize_battery_status(PYB_CLIENT, positions, battery_simulators, max(BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT))
        
            if ENABLE_TRAILS and i % (VISUALIZATION_FREQ // 2) == 0:  # Only if trails enabled
                for j in range(NUM_DRONES):
                    drone_trails[j].append(positions[j])
                    if len(drone_trails[j]) > 5:  # REDUCED trail length from 20 to 5
                        drone_trails[j].pop(0)
                    
                    if len(drone_trails[j]) > 2:
                        p.addUserDebugLine(
                            drone_trails[j][0],
                            drone_trails[j][-1],
                            lineColorRGB=DRONE_COLORS[j],
                            lineWidth=1,
                            lifeTime=2.0,  # Longer lifetime, less frequent updates
                            physicsClientId=PYB_CLIENT
                        )

            if i == 0:
                print("🔍 DEBUGGING: Checking environment setup...")
                print(f"Environment type: {type(env)}")
                
                # Check environment attributes
                env_attrs = [attr for attr in dir(env) if 'drone' in attr.lower() or 'id' in attr.lower()]
                print(f"Environment drone-related attributes: {env_attrs}")
                
                # Check PyBullet bodies
                num_bodies = p.getNumBodies(physicsClientId=PYB_CLIENT)
                print(f"Total PyBullet bodies: {num_bodies}")

            # Add this every 100 iterations to see restart status:
            if i % 100 == 0:
                restart_stats = safe_restart_manager.get_restart_stats()
                if restart_stats:
                    print(f"🔍 Restart stats at iteration {i}:")
                    print(f"   Body IDs discovered: {restart_stats.get('body_ids_discovered', False)}")
                    print(f"   Drone body IDs: {restart_stats.get('drone_body_ids', 'None')}")
                    print(f"   Restart counts: {restart_stats.get('restart_counts', [])}")

            """# Camera processing
            for d in range(NUM_DRONES):
                cam_img = get_nadir_camera_image(positions[d], None, physicsClientId=PYB_CLIENT)
                img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                output_img = detect_crack(img_bgr)
                cv2.imshow(f"Drone {d} Camera", output_img)
                cv2.waitKey(1)"""

            timesimulation = time.time() - START

            if i % COVERAGE_FREQ == 0:  # Much less frequent coverage updates
                current_altitude = positions[0][2] if positions else ALTITUDE
                coverage.update(positions, current_altitude)
                
                # Only visualize in GUI mode and less frequently
                if gui and i % (COVERAGE_FREQ * 2) == 0:
                    coverage.visualize_coverage_map()

            """if gui and i % COVERAGE_FREQ == 0:
                current_altitude = positions[0][2] if positions else ALTITUDE
                coverage_visualizer.update_coverage_from_positions(positions, current_altitude, pyb_client=PYB_CLIENT)"""
            

            if gui and i % (COVERAGE_FREQ * 3) == 0:  # Less frequent for performance
                current_altitude = positions[0][2] if positions else ALTITUDE
                
                # Check if any drone is in waypoint following mode
                waypoint_mode_active = False
                for j in range(NUM_DRONES):
                    if not waypoint_managers[j].is_finished():
                        waypoint_mode_active = True
                        break
                
                # Only detect overlaps during waypoint following mode
                if waypoint_mode_active and ENABLE_OVERLAP_DETECTION:
                    overlap_detected, active_overlap_count = overlap_detector.detect_overlaps(
                        positions, current_simulation_time, current_altitude
                    )
                    
                    overlap_stats = overlap_detector.get_overlap_statistics()
                    
                    print(f"🔄 Overlap detection active - Waypoint mode: {overlap_detected} overlaps detected")
                else:
                    # Default values when overlap detection is disabled or in exploration mode
                    overlap_detected = False
                    active_overlap_count = 0
                    overlap_stats = {
                        'total_overlap_events': overlap_detector.get_overlap_statistics().get('total_overlap_events', 0),
                        'overlap_percentage': 0.0, 
                        'active_overlaps': 0
                    }
                    
                    if i % 300 == 0 and not waypoint_mode_active:  # Occasional logging
                        print(f"⭐ Overlap detection paused - Exploration mode active")
                
                # Update the coverage visualizer call to pass the waypoint mode status
                coverage_visualizer.update_coverage_from_positions(
                    positions, current_altitude, 
                    overlap_cells=overlap_detector.overlap_coverage_cells if waypoint_mode_active else None,
                    pyb_client=PYB_CLIENT,
                    show_overlap=waypoint_mode_active  # Only show overlap in waypoint mode
                )

            if gui and ENABLE_CAMERA_FOOTPRINTS and i % COVERAGE_FREQ == 0:
                current_altitude = positions[0][2] if positions else ALTITUDE
                draw_realistic_camera_footprints(positions, current_altitude, PYB_CLIENT)
            
            drone_modes = []
            waypoint_following_drones = 0
            exploration_drones = 0

            for j in range(NUM_DRONES):
                # Determine drone mode more precisely
                is_in_waypoint_mode = not waypoint_managers[j].is_finished()
                is_in_exploration = exploration_targets[j] is not None
                
                if is_in_waypoint_mode:
                    # Drone is following predefined waypoints
                    drone_mode = 0  # Waypoint following mode
                    waypoint_following_drones += 1
                elif is_in_exploration:
                    # Drone is in exploration mode
                    drone_mode = 1  # Exploration mode  
                    exploration_drones += 1
                else:
                    # Drone completed or idle
                    drone_mode = 2  # Completed/idle mode
                
                drone_modes.append(drone_mode)

            # Add this information to your status display
            if i % STATUS_FREQ == 0:
                mode_status = f"Waypoint: {waypoint_following_drones}, Exploration: {exploration_drones}, Idle: {NUM_DRONES - waypoint_following_drones - exploration_drones}"
                print(f"📊 Drone modes - {mode_status}")

            if i % 10 == 0:  # Collect data every 10 iterations instead of every iteration
                current_coverage = coverage.get_coverage_percentage()['coverage_percentage']
                
                # Prepare data arrays
                battery_percentages = []
                speeds = []
                threat_levels_data = []
                distances_data = []
                energy_data = []
                min_obs_dist_data = []
                avg_obs_dist_data = []
                all_obs_dist_data = []
                
                for j in range(NUM_DRONES):
                    battery_info = battery_simulators[j].get_battery_info()
                    speed = np.linalg.norm(velocities[j][:2]) if len(velocities[j]) > 1 else 0
                    
                    battery_percentages.append(battery_info['percentage'])
                    speeds.append(speed)
                    threat_levels_data.append(threat_levels[j] if j < len(threat_levels) else 0)
                    distances_data.append(battery_info['distance_traveled'])
                    energy_data.append((350 * (100 - battery_info['percentage']) / 100.0))
                    
                    # Get sensor data for obstacle distances (less frequently)
                    if i % 30 == 0:  # Only every 30 iterations
                        sensor_data_j = get_enhanced_sensor_readings(PYB_CLIENT, positions[j], velocities[j])
                        min_obs_dist_j, avg_obs_dist_j, all_obs_dist_j = extract_all_obstacle_distances(sensor_data_j)
                    else:
                        # Use previous values or defaults
                        min_obs_dist_j = float('inf')
                        avg_obs_dist_j = float('inf') 
                        all_obs_dist_j = "cached"
                    
                    min_obs_dist_data.append(min_obs_dist_j)
                    avg_obs_dist_data.append(avg_obs_dist_j)
                    all_obs_dist_data.append(all_obs_dist_j)
                
                overlap_data = {
                    'overlap_detected': overlap_detected if 'overlap_detected' in locals() else False,
                    'active_overlap_count': active_overlap_count if 'active_overlap_count' in locals() else 0,
                    'total_overlap_events': overlap_stats['total_overlap_events'] if 'overlap_stats' in locals() else 0,
                    'overlap_area_percentage': overlap_stats['overlap_percentage'] if 'overlap_stats' in locals() else 0.0,
                    'waypoint_mode_active': waypoint_following_drones > 0,  # Add mode tracking
                    'exploration_mode_active': exploration_drones > 0
                }
                # Add all data at once
                simulation_data.add_data_point(
                    datetime.now(), current_simulation_time, current_coverage,
                    positions, battery_percentages, speeds, threat_levels_data,
                    distances_data, energy_data, min_obs_dist_data, avg_obs_dist_data, all_obs_dist_data, drone_modes, overlap_data
                )

                if len(simulation_data.data['timestamp']) % 100 == 0:
                    save_to_backup_csv(
                        backup_csv, datetime.now(), i * env.CTRL_TIMESTEP, current_coverage,
                        battery_percentages, speeds, distances_data, NUM_DRONES
                    )


            if i > 0 and i % EXCEL_FREQ == 0:
                current_stats = {
                    'current_time': time.time() - START,
                    'simulation_time': current_simulation_time,
                    'current_coverage': coverage.get_coverage_percentage()['coverage_percentage'],
                    'num_drones': NUM_DRONES,
                    'building_width': BUILDING_SIZE_WIDTH,
                    'building_height': BUILDING_SIZE_HEIGHT,
                    'altitude': ALTITUDE,
                    'status': 'RUNNING'
                }
                
                excel_filename = update_excel_file(
                    simulation_data.data, current_stats, NUM_DRONES, output_folder, 
                    battery_simulators, waypoints_skipped, excel_filename
                )
                
                print(f"📊 Excel updated at iteration {i} (every {EXCEL_FREQ} iterations)")
                
            # ===== ENHANCED MAIN DRONE CONTROL LOOP WITH BATTERY =====
            for j in range(NUM_DRONES):
                current_pos = obs[j][0:3]
                current_vel = velocities[j]

                # ===== BATTERY SIMULATION UPDATE =====
                # Check if emergency maneuver is active
                is_emergency = (hasattr(improved_controllers[j], 'emergency_maneuver_active') and 
                               improved_controllers[j].emergency_maneuver_active)
                
                # Update battery based on drone activity
                battery_percentage = battery_simulators[j].update_battery(
                    current_pos, current_vel, 
                    is_emergency_active=is_emergency,
                    is_camera_active=True,  # Camera always active
                    simulation_timestep=env.CTRL_TIMESTEP
                )

                if hasattr(get_waypoint_configuration, 'building_shape'):
                    building_shape = get_waypoint_configuration.building_shape
                    create_building_boundary_visualization(PYB_CLIENT, building_shape, BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT)
                
                # Get comprehensive battery info
                battery_info = battery_simulators[j].get_battery_info()
                
                # Handle battery emergencies (RTL/Landing)
                emergency_target, mission_status = handle_battery_emergency_actions(
                    j, battery_info, current_pos, waypoint_managers, exploration_targets
                )
                
                # Issue battery warnings
                if (battery_info['status'] in ['LOW', 'CRITICAL', 'RTL'] and 
                    not battery_warnings_issued[j]):
                    print(f"⚠️ Drone {j} {battery_info['status']} BATTERY: {battery_info['percentage']:.1f}%")
                    battery_warnings_issued[j] = True

                # ===== WAYPOINT MANAGEMENT WITH TAKEOFF AND BATTERY OVERRIDE =====
                if emergency_target is not None:
                    # Override with battery emergency target (RTL or Emergency Landing)
                    target_pos = emergency_target
                    print(f"🔋 Drone {j} battery emergency override: {mission_status}")
                    
                elif drone_states[j].state == "TAKEOFF":
                    # Takeoff phase - stay at current XY position, climb to target altitude
                    target_pos = np.array([current_pos[0], current_pos[1], ALTITUDE])
                    
                    # Check if takeoff is complete
                    if drone_states[j].update_takeoff_status(current_pos[2]):
                        print(f"✈️ Drone {j} takeoff complete! Starting mission...")
                        
                elif drone_states[j].state == "MISSION":
                    if waypoint_managers[j].is_finished():
                        # ===== EXPLORATION MODE WITH HIGH COVERAGE BYPASS =====
                        if exploration_targets[j] is None:
                            print(f"🔍 Drone {j} searching for exploration target...")
                            
                            # Check current coverage percentage
                            current_coverage = coverage.get_coverage_percentage()['coverage_percentage']
                            
                            # For high coverage (>90%), bypass coordinated exploration
                            if current_coverage > 90.0:
                                print(f"🎯 HIGH COVERAGE ({current_coverage:.1f}%) - Using aggressive direct exploration")
                                
                                # Use aggressive exploration directly
                                aggressive_target = get_smart_exploration_target_aggressive(current_pos, coverage)
                                
                                if aggressive_target is not None:
                                    target_pos = np.array([aggressive_target[0], aggressive_target[1], ALTITUDE])
                                    exploration_targets[j] = target_pos.copy()
                                    print(f"✅ Drone {j} assigned AGGRESSIVE target: [{target_pos[0]:.1f}, {target_pos[1]:.1f}]")
                                else:
                                    target_pos = current_pos
                                    print(f"✅ Drone {j} - Area appears fully covered!")
                            
                            else:
                                # Normal coverage - try coordinated exploration first
                                print(f"🔄 NORMAL COVERAGE ({current_coverage:.1f}%) - Using coordinated exploration")
                                coordinated_target = exploration_manager.get_coordinated_exploration_target(
                                    drone_id=j,
                                    current_pos=current_pos,
                                    coverage_tracker=coverage,
                                    other_drone_positions=[positions[k] for k in range(NUM_DRONES) if k != j]
                                )
                                
                                if coordinated_target is not None:
                                    target_pos = np.array([coordinated_target[0], coordinated_target[1], ALTITUDE])
                                    exploration_targets[j] = target_pos.copy()
                                    print(f"🎯 Drone {j} assigned COORDINATED target: [{target_pos[0]:.1f}, {target_pos[1]:.1f}]")
                                else:
                                    # Coordinated failed - use aggressive fallback
                                    print(f"🔄 Coordinated exploration failed - using aggressive fallback")
                                    aggressive_target = get_smart_exploration_target_aggressive(current_pos, coverage)
                                    if aggressive_target is not None:
                                        target_pos = np.array([aggressive_target[0], aggressive_target[1], ALTITUDE])
                                        exploration_targets[j] = target_pos.copy()
                                        print(f"✅ Drone {j} assigned FALLBACK aggressive target: [{target_pos[0]:.1f}, {target_pos[1]:.1f}]")
                                    else:
                                        target_pos = current_pos
                                        print(f"❌ Drone {j} no targets found")
                        
                        else:
                            # Already have exploration target - check if reached
                            target_pos = exploration_targets[j]
                            distance_to_exploration = np.linalg.norm(target_pos - current_pos)
                            
                            if distance_to_exploration < 1.0:
                                print(f"🎯 Drone {j} REACHED exploration target (dist: {distance_to_exploration:.2f}m)")
                                
                                # Clear target (no blacklisting needed for now)
                                exploration_targets[j] = None
                                exploration_manager.clear_target(j)
                                target_pos = current_pos
                                print(f"🔄 Drone {j} target cleared, will search for new target next iteration")
                            
                            # Add simple timeout protection
                            else:
                                if not hasattr(run, 'exploration_start_times'):
                                    run.exploration_start_times = {}
                                
                                if j not in run.exploration_start_times:
                                    run.exploration_start_times[j] = time.time()
                                
                                time_at_target = time.time() - run.exploration_start_times[j]
                                
                                if time_at_target > 15.0:  # 15 second timeout
                                    print(f"⏰ Drone {j} exploration TIMEOUT ({time_at_target:.1f}s)")
                                    exploration_targets[j] = None
                                    exploration_manager.clear_target(j)
                                    run.exploration_start_times[j] = time.time()
                                    target_pos = current_pos
                    
                    else:
                        # Still in mission waypoint phase
                        current_waypoint = waypoint_managers[j].get_current_waypoint()
                        if current_waypoint is not None:
                            target_pos = current_waypoint
                        else:
                            target_pos = current_pos
                            
                else:
                    # Default fallback
                    target_pos = current_pos

                current_targets[j] = target_pos.copy()

                if (not hasattr(run, 'last_targets') or 
                    run.last_targets[j] is None or 
                    np.linalg.norm(target_pos - run.last_targets[j]) > 0.5):
                    
                    distance = np.linalg.norm(target_pos - current_pos)
                    print(f"🎯 Drone {j} new target: [{target_pos[0]:6.2f}, {target_pos[1]:6.2f}, {target_pos[2]:6.2f}] (dist: {distance:.1f}m)")
                
                # ===== WALL-SPECIFIC OBSTACLE AVOIDANCE =====
                other_drones = [positions[k] for k in range(NUM_DRONES) if k != j]
                sensor_data = sensor_cache.get_sensor_data(PYB_CLIENT, current_pos, current_vel, i)
                speed_factor = get_speed_zone_factor(current_pos, target_pos, sensor_data)

                if hasattr(improved_controllers[j], 'current_speed_factor'):
                    improved_controllers[j].current_speed_factor = speed_factor
                else:
                    improved_controllers[j].current_speed_factor = speed_factor
                
                # Compute force
                pf_force, dist_to_waypoint, stuck_near_wall, waypoint_unreachable, wall_info = improved_controllers[j].compute_force(
                    current_pos, current_vel, target_pos, sensor_data['obstacles'], 
                    ALTITUDE, other_drones, PYB_CLIENT
                )

                if i % 10 == 0:  # Collect control analysis data every 10 iterations
                    # Get DYNAMIC target speed from the controller
                    dynamic_target_speed = improved_controllers[j].last_target_speed
                    
                    # Calculate actual dynamic factors
                    current_speed = np.linalg.norm(current_vel[:2])
                    dist_to_target = np.linalg.norm(target_pos - current_pos)
                    
                    # Determine speed context
                    speed_context = "FULL_SPEED"
                    if dist_to_target < 1.0:
                        speed_context = "APPROACHING_WAYPOINT"
                    elif hasattr(improved_controllers[j], 'current_speed_factor') and improved_controllers[j].current_speed_factor < 0.8:
                        speed_context = "OBSTACLE_SLOWDOWN"
                    elif stuck_near_wall:
                        speed_context = "EMERGENCY_SLOWDOWN"
                    
                    pid_data = {
                        'altitude_error': ALTITUDE - current_pos[2],
                        'altitude_command': 0,  # Will be filled after velocity_command calculation
                        'speed_error': dynamic_target_speed - current_speed,
                        'speed_command': current_speed,
                        'target_speed': dynamic_target_speed,
                        'position_error': np.linalg.norm(target_pos - current_pos),
                        'distance_to_waypoint': dist_to_target,
                        'speed_context': speed_context,
                        'speed_factor': getattr(improved_controllers[j], 'current_speed_factor', 1.0)
                    }
                    
                    # Potential field performance data
                    min_obstacle_dist, _, _ = extract_all_obstacle_distances(sensor_data)
                    if min_obstacle_dist == float('inf'):
                        min_obstacle_dist = 10.0
                    
                    pf_data = {
                        'total_force_magnitude': np.linalg.norm(pf_force),
                        'min_obstacle_distance': min_obstacle_dist,
                        'near_miss_event': min_obstacle_dist < 0.5,
                        'stuck_event': stuck_near_wall,
                        'threat_level': sensor_data.get('threat_level', 0.0)
                    }
                    
                    simulation_data.add_pid_data(j, pid_data)
                    simulation_data.add_potential_field_data(j, pf_data)

                # Get drone orientation for yaw control
                try:
                    drone_body_id = DRONE_IDS[j] if j < len(DRONE_IDS) else j
                    _, drone_quat = p.getBasePositionAndOrientation(drone_body_id, physicsClientId=PYB_CLIENT)
                    drone_rpy = p.getEulerFromQuaternion(drone_quat)
                except:
                    drone_rpy = None

                # Convert force to velocity command
                velocity_command = velocity_adapters[j].convert_force_to_velocity_command(
                    pf_force=pf_force,
                    current_velocity=current_vel,
                    target_altitude=ALTITUDE,
                    current_altitude=current_pos[2],
                    current_rpy=drone_rpy,
                    desired_heading=None
                )

                if i % 10 == 0 and hasattr(simulation_data, 'pid_data') and j in simulation_data.pid_data:
                    if simulation_data.pid_data[j]['altitude_command']:
                        simulation_data.pid_data[j]['altitude_command'][-1] = velocity_command[2]

                # Add this after your velocity_command calculation in the main loop:
                if i % 60 == 0 and j == 0:  # Debug every second for drone 0
                    print(f"🔍 YAW DEBUG - Drone {j}:")
                    print(f"   Velocity command: {velocity_command}")
                    print(f"   Current RPY: {np.degrees(drone_rpy) if drone_rpy is not None else 'None'}")
                    print(f"   Target position: {target_pos}")
                    print(f"   Current position: {current_pos}")
                    
                    # Check adapter status
                    adapter_status = velocity_adapters[j].get_status()
                    print(f"   Adapter status: {adapter_status}")

                # Debug output (optional)
                if i % 200 == 0 and j == 0:  # Debug drone 0 every 200 iterations
                    print(f"🔍 Drone {j} Command Debug:")
                    print(f"   PF Force: {pf_force}")
                    print(f"   Velocity Command: {velocity_command}")
                    print(f"   Current Position: {current_pos}")
                    print(f"   Current Velocity: {current_vel}")


                
                # Store metrics
                threat_levels[j] = sensor_data['threat_level']
                if hasattr(improved_controllers[j], 'emergency_maneuver_active') and \
                improved_controllers[j].emergency_maneuver_active:
                    emergency_activations[j] += 1
                
                # ===== WALL-SPECIFIC WAYPOINT SKIPPING LOGIC =====
                # Skip waypoint management if in battery emergency mode OR takeoff mode
                if (emergency_target is None and 
                    drone_states[j].state == "MISSION" and 
                    not waypoint_managers[j].is_finished()):
                    
                    # Check if waypoint should be skipped due to WALL conditions only
                    should_skip = waypoint_managers[j].should_skip_waypoint(
                        current_pos, stuck_near_wall, waypoint_unreachable, wall_info, current_simulation_time
                    )
                    
                    if should_skip:
                        if waypoint_managers[j].skip_to_next_waypoint(current_simulation_time):
                            waypoints_skipped[j] += 1
                            print(f"🧱 Drone {j} skipped waypoint due to WALL stuck condition")
                    
                    # Check if waypoint is reached normally
                    elif dist_to_waypoint < improved_controllers[j].waypoint_threshold:
                        waypoint_managers[j].advance_waypoint(current_simulation_time)
                
                # ===== EXPLORATION TARGET MANAGEMENT =====
                # Skip exploration management if in battery emergency
                if emergency_target is None and exploration_targets[j] is not None:
                    if dist_to_waypoint < improved_controllers[j].waypoint_threshold:
                        # FIXED: Check if exploration target is covered
                        
                        resolution = coverage.resolution  # Use actual resolution (0.2)
                        grid_x = int((exploration_targets[j][0] + BUILDING_SIZE_WIDTH/2) / resolution)
                        grid_y = int((exploration_targets[j][1] + BUILDING_SIZE_HEIGHT/2) / resolution)
                        
                        # FIXED: Use grid_width and grid_height instead of grid_size
                        if (0 <= grid_x < coverage.grid_width and 0 <= grid_y < coverage.grid_height):
                            local_patch = coverage.coverage_map[
                                max(0, grid_y-3):min(coverage.grid_height, grid_y+4), 
                                max(0, grid_x-3):min(coverage.grid_width, grid_x+4)
                            ]
                            local_coverage = np.mean(local_patch) if local_patch.size > 0 else 0

                            if local_coverage >= 0.95:
                                print(f"Drone {j} finished covering current exploration target.")
                                exploration_targets[j] = None
                    
                    elif stuck_near_wall and wall_info.get('is_wall', False):
                        print(f"🧱 Drone {j} abandoning exploration target due to wall obstruction")
                        exploration_targets[j] = None

                # Add this in your main drone control loop, after battery check:
                if emergency_target is None and drone_states[j].state == "MISSION":
                    # Reset yaw periodically to prevent drift
                    if i % 600 == 0:  # Every 10 seconds
                        # Reset target yaw to current yaw
                        if drone_rpy is not None:
                            velocity_adapters[j].target_yaw = drone_rpy[2]
                            velocity_adapters[j].yaw_initialized = True
                            print(f"🔄 Reset yaw for drone {j} to current: {np.degrees(drone_rpy[2]):.1f}°")

                # Set action
                action[j, 0:4] = velocity_command
                #action[j, 3] = 0.99

                if abs(action[j, 3]) < 0.05:  # 0.05 rad/s deadzone
                    action[j, 3] = 0.0

                if i % 20 == 0:  # Every 20 iterations
                    for j in range(NUM_DRONES):
                        if exploration_targets[j] is not None:  # In exploration mode
                            current_altitude = positions[j][2]
                            vertical_velocity = velocities[j][2] if len(velocities[j]) > 2 else 0
                            target_altitude = ALTITUDE
                            
                            # Check for altitude issues in exploration mode
                            altitude_error = abs(current_altitude - target_altitude)
                            
                            if altitude_error > 0.3:
                                print(f"⚠️ EXPLORATION ALTITUDE: Drone {j} at {current_altitude:.2f}m (target: {target_altitude:.2f}m)")
                            
                            if vertical_velocity < -0.3:
                                print(f"⚠️ EXPLORATION FALLING: Drone {j} vertical velocity: {vertical_velocity:.2f}m/s")
                                
                                # Emergency altitude correction for exploration mode
                                print(f"🚨 Emergency altitude correction for drone {j}")


                # Add this every 100 iterations to monitor wall safety
                if i % 100 == 0:
                    print("🔍 Wall Safety Status:")
                    for j, pos in enumerate(positions):
                        # Calculate distance to nearest wall
                        dist_to_left = abs(pos[0] + BUILDING_SIZE_WIDTH/2)
                        dist_to_right = abs(pos[0] - BUILDING_SIZE_WIDTH/2)
                        dist_to_bottom = abs(pos[1] + BUILDING_SIZE_HEIGHT/2)
                        dist_to_top = abs(pos[1] - BUILDING_SIZE_HEIGHT/2)
                        
                        min_wall_distance = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                        
                        if min_wall_distance < 2.0:
                            print(f"   ⚠️ Drone {j}: {min_wall_distance:.1f}m from wall (CLOSE)")
                        else:
                            print(f"   ✅ Drone {j}: {min_wall_distance:.1f}m from wall (SAFE)")
                        
                        # Check if in exploration mode
                        if exploration_targets[j] is not None:
                            print(f"     → Exploring target: {exploration_targets[j]}")

                # ===== ENHANCED VISUALIZATION WITH WALL DETECTION =====
                if gui and ENABLE_FORCE_VECTORS and i % VISUALIZATION_FREQ == 0:
                    # Force vector (red)
                    p.addUserDebugLine(
                        current_pos, 
                        current_pos + pf_force, 
                        lineColorRGB=[1,0,0], 
                        lifeTime=1.0,  # Longer lifetime
                        physicsClientId=PYB_CLIENT
                    )
                    # Target line (green)
                    p.addUserDebugLine(
                        current_pos, 
                        target_pos, 
                        lineColorRGB=[0,1,0], 
                        lifeTime=1.0,  # Longer lifetime
                        physicsClientId=PYB_CLIENT
                    )
                    
                    # Threat level indicator (only if significant threat)
                    if sensor_data['threat_level'] > 0.3:  # Only show if meaningful threat
                        threat_color = [1, 1-sensor_data['threat_level'], 0]
                        p.addUserDebugLine(
                            current_pos, 
                            current_pos + np.array([0, 0, 0.5]), 
                            lineColorRGB=threat_color, 
                            lineWidth=5,
                            lifeTime=1.0,
                            physicsClientId=PYB_CLIENT
                        )
                    
                    # Wall detection indicator (thick purple line if WALL detected)
                    if wall_info.get('is_wall', False):
                        p.addUserDebugLine(
                            current_pos, 
                            current_pos + np.array([0, 0, 1.0]), 
                            lineColorRGB=[1, 0, 1], 
                            lineWidth=10,
                            lifeTime=0.1,
                            physicsClientId=PYB_CLIENT
                        )
                    
                    # Small obstacle indicator (thin blue line)
                    elif wall_info.get('obstacle_type') == 'small_obstacle':
                        p.addUserDebugLine(
                            current_pos, 
                            current_pos + np.array([0, 0, 0.3]), 
                            lineColorRGB=[0, 0, 1], 
                            lineWidth=3,
                            lifeTime=0.1,
                            physicsClientId=PYB_CLIENT
                        )

                # ===== ENHANCED STATUS DISPLAY WITH BATTERY INFO =====
                emergency_status = "🚨" if (hasattr(improved_controllers[j], 'emergency_maneuver_active') and 
                                        improved_controllers[j].emergency_maneuver_active) else ""
                
                # Battery status with color coding
                battery_status = f"🔋{battery_info['percentage']:.1f}%({battery_info['status']})"
                
                # Different indicators for walls vs small obstacles
                obstacle_status = ""
                if wall_info.get('is_wall', False):
                    wall_coverage = wall_info.get('wall_coverage', 0)
                    obstacle_status = f"🧱WALL({wall_coverage:.1f})"
                elif wall_info.get('obstacle_type') == 'small_obstacle':
                    obstacle_status = "🔷OBJ"
                
                stuck_status = "🔒STUCK" if stuck_near_wall else ""
                skip_status = f"⏭️{waypoints_skipped[j]}" if waypoints_skipped[j] > 0 else ""
                
                # Mission status indicator
                mission_status_indicator = ""
                if emergency_target is not None:
                    if mission_status == 'rtl':
                        mission_status_indicator = "🏠RTL"
                    elif mission_status == 'land':
                        mission_status_indicator = "🛬LAND"
                
                # Current waypoint info
                current_wp_idx = waypoint_managers[j].current_waypoint_index
                total_wps = len(WAYPOINTS[j])
                wp_progress = f"WP:{current_wp_idx}/{total_wps}"
                current_vel = velocities[j]

                speed_horizontal = np.linalg.norm(current_vel[:2])
                speed_vertical = (current_vel[2])

                
                if gui and i % STATUS_FREQ == 0:  # Update status less frequently
                    text_position = np.array([-BUILDING_SIZE_WIDTH+18, BUILDING_SIZE_HEIGHT/1 - j*2.5, 0.5])
                    
                    # SIMPLIFIED status text for better performance
                    if PERFORMANCE_MODE:
                        # Short status for performance mode
                        text = (f"D{j}: Alt:{current_pos[2]:.1f}m | "
                                f"Spd:{speed_horizontal:.2f}m/s | "
                                f"Bat:{battery_info['percentage']:.0f}% | "
                                f"Cov:{coverage.get_coverage_percentage()['coverage_percentage']:.1f}%")
                    else:
                        # Full status for normal mode  
                        text = (f"Drone {j} {emergency_status}{mission_status_indicator}{obstacle_status}{stuck_status}{skip_status} | "
                                f"{wp_progress} | "
                                f"Threat: {sensor_data['threat_level']:.2f} | "
                                f"Alt: {current_pos[2]:.2f}m | "
                                f"Time: {current_simulation_time:.1f}s | "
                                f"Speed: {speed_horizontal:.2f}m/s H | {speed_vertical:.2f}m/s V | "
                                f"Coverage: {coverage.get_coverage_percentage()['coverage_percentage']:.2f}% |"
                                f"Distance: {battery_info['distance_traveled']:.2f}m | "
                                f"Battery: {battery_info['percentage']:.1f}% | ")
                    
                    # Color code text based on battery status
                    if battery_info['status'] == 'CRITICAL':
                        text_color = [1, 0, 0]  # Red
                    elif battery_info['status'] == 'LOW':
                        text_color = [1, 0.5, 0]  # Orange
                    elif battery_info['status'] == 'RTL':
                        text_color = [1, 1, 0]  # Yellow
                    else:
                        text_color = [0, 1, 0]  # Green
                    
                    p.addUserDebugText(text, text_position, textColorRGB=text_color, 
                                    textSize=1, lifeTime=2.0,  # Longer lifetime
                                    physicsClientId=PYB_CLIENT)
                    

                    
                if i % 200 == 0 and i > 500:  # Check every 200 iterations after initial startup
                    if hasattr(run, 'last_perf_check'):
                        recent_200_time = time.time() - getattr(run, 'last_200_check', time.time())
                        avg_iter_time = recent_200_time / 200 * 1000
                        
                        # Auto-activate emergency mode if performance is critical
                        if avg_iter_time > 100 and PERFORMANCE_MODE == False:
                            print(f"⚠️ Critical performance detected: {avg_iter_time:.1f}ms/iter")
                            emergency_performance_mode()
                        elif avg_iter_time > 60 and any([ENABLE_TRAILS, ENABLE_FORCE_VECTORS, ENABLE_CAMERA_FOOTPRINTS]):
                            print(f"⚠️ Poor performance detected: {avg_iter_time:.1f}ms/iter - disabling visualizations")
                            ENABLE_TRAILS = False
                            ENABLE_FORCE_VECTORS = False
                            ENABLE_CAMERA_FOOTPRINTS = False
                        
                        run.last_200_check = time.time()
            
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
                if i % 100 == 0:
                    max_dimension = max(BUILDING_SIZE_WIDTH, BUILDING_SIZE_HEIGHT)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=max_dimension * 0.5,
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
            
            if i % 100 == 0:  # Every 100 iterations
                current_time = time.time()
                if hasattr(run, 'last_perf_check'):
                    elapsed = current_time - run.last_perf_check
                    avg_iter_time = elapsed / 100 * 1000  # ms per iteration
                    fps = 100 / elapsed
                    
                    print(f"⚡ Performance: {avg_iter_time:.1f}ms/iter, {fps:.1f} FPS")
                    
                    if avg_iter_time > 50:
                        print(f"⚠️ Performance warning: >50ms per iteration")
                        if not PERFORMANCE_MODE:
                            print(f"💡 Consider enabling PERFORMANCE_MODE")
                
                run.last_perf_check = current_time

            if gui and i % 5 == 0:  # Draw every 5 iterations
                draw_simple_waypoint_trails(positions, current_targets, DRONE_COLORS, PYB_CLIENT)
            
            # Store targets for next iteration comparison
            run.last_targets = [t.copy() if t is not None else None for t in current_targets]

            if i % 500 == 0:  # Every 500 iterations
                import gc
                gc.collect()
                print(f"🧹 Memory cleanup at iteration {i}")

            env.render()

            if gui:
                # Adaptive sync - skip sync when running slow
                if hasattr(run, 'last_perf_check') and i > 100:
                    recent_time = time.time() - run.last_perf_check
                    if recent_time > 2.0:  # If last 100 iterations took more than 2 seconds
                        if i % 2 == 0:  # Only sync every other iteration when slow
                            sync(i, START, env.CTRL_TIMESTEP)
                    else:
                        sync(i, START, env.CTRL_TIMESTEP)
                else:
                    sync(i, START, env.CTRL_TIMESTEP)

            iteration_time = time.time() - iteration_start
            if iteration_time > 0.1:  # Warn about very slow iterations
                print(f"⚠️ Slow iteration {i}: {iteration_time*1000:.1f}ms")


    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user (Ctrl+C)")
        simulation_stopped = True
    except MemoryError:
        print("\n🚨 MEMORY ERROR - Activating emergency cleanup")
        emergency_performance_mode()
        import gc
        gc.collect()
        simulation_stopped = True
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        print("🔧 Activating emergency performance mode")
        emergency_performance_mode()
        simulation_stopped = True

    finally:
        # ===== PERFORMANCE SUMMARY (NEW) =====
        if hasattr(run, 'last_perf_check'):
            total_time = time.time() - START
            total_iterations = i if 'i' in locals() else 0
            
            if total_iterations > 0:
                avg_iter_time = total_time / total_iterations * 1000
                avg_fps = total_iterations / total_time
                
                print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
                print(f"   Total time: {total_time:.1f}s")
                print(f"   Total iterations: {total_iterations}")
                print(f"   Average iteration time: {avg_iter_time:.2f}ms")
                print(f"   Average FPS: {avg_fps:.1f}")
                
                if avg_iter_time < 20:
                    performance_rating = "EXCELLENT"
                elif avg_iter_time < 40:
                    performance_rating = "GOOD"  
                elif avg_iter_time < 80:
                    performance_rating = "FAIR"
                else:
                    performance_rating = "POOR"
                    
                print(f"   Performance rating: {performance_rating}")
        
        # ===== CLEANUP (ENHANCED) =====
        print("🧹 Performing final cleanup...")
        try:
            if 'coverage_visualizer' in locals():
                coverage_visualizer.clear_all_coverage(PYB_CLIENT)
            if gui:
                p.removeAllUserDebugItems(physicsClientId=PYB_CLIENT)
            cv2.destroyAllWindows()
        except:
            pass
        
        # ===== YOUR EXISTING FINAL RESULTS (KEEP ALL OF THIS) =====
        final_simulation_time = (i * env.CTRL_TIMESTEP) if 'i' in locals() else 0
        actual_duration = final_simulation_time
        
        # Check if simulation completed or was stopped early
        completion_status = "COMPLETED" if not simulation_stopped else "STOPPED_EARLY"
        completion_percentage = (actual_duration / duration_sec) * 100 if duration_sec > 0 else 100
        
        coverage_result = coverage.get_coverage_percentage()
        final_coverage = coverage_result['coverage_percentage']
        print(f"Final Coverage: {final_coverage:.2f}%")

        total_waypoints = sum(len(wp) for wp in WAYPOINTS)

        # Calculate enhanced metrics
        avg_threat_level = np.mean(threat_levels) if 'threat_levels' in locals() and len(threat_levels) > 0 else 0
        total_emergencies = sum(emergency_activations) if 'emergency_activations' in locals() else 0
        total_skipped = sum(waypoints_skipped) if 'waypoints_skipped' in locals() else 0

        # Battery statistics
        final_battery_levels = [sim.get_battery_info()['percentage'] for sim in battery_simulators]
        avg_battery_remaining = np.mean(final_battery_levels)
        min_battery_remaining = min(final_battery_levels)

        # Save final Excel file
        try:
            final_stats = {
                'current_time': actual_duration,
                'current_coverage': final_coverage,
                'num_drones': NUM_DRONES,
                'building_width': BUILDING_SIZE_WIDTH,
                'building_height': BUILDING_SIZE_HEIGHT,
                'altitude': ALTITUDE,
                'status': completion_status,
                'total_waypoints': total_waypoints,
                'total_skipped': total_skipped,
                'total_emergencies': total_emergencies,
                'avg_battery': avg_battery_remaining,
                'min_battery': min_battery_remaining
            }
            
            excel_filename = update_excel_file(
                simulation_data, final_stats, NUM_DRONES, output_folder, 
                battery_simulators, waypoints_skipped, excel_filename
            )

            
            if excel_filename:
                print(f"📊 Final Excel file saved: {excel_filename}")
                # UPDATED: Use .data for LimitedDataCollector
                print(f"📊 Total data points collected: {len(simulation_data.data['simulation_time'])}")   
            
        except Exception as e:
            print(f"❌ Error saving final Excel file: {e}")

        if plot and 'simulation_data' in locals():
            print("📊 Generating control evaluation plots...")
            
            try:
                plotter = ControlEvaluationPlotter(NUM_DRONES)
                
                # PID performance plots
                plotter.plot_pid_speed_performance(simulation_data, output_folder)
                
                # Obstacle avoidance plots
                plotter.plot_obstacle_avoidance_performance(simulation_data, min_safe_distance=0.5, save_path=output_folder)
                
                print("✅ Control evaluation plots completed!")
                
            except Exception as e:
                print(f"❌ Error generating plots: {e}")


        # ===== CLOSE ENVIRONMENT (MOVED TO END) =====
        env.close()
        print("✅ Cleanup completed")

        # ===== YOUR EXISTING LOGGER CODE (KEEP THIS) =====
        logger.save_as_csv(f"coverage_path_{NUM_DRONES}drones_enhanced_battery")
        if plot:
            logger.plot()
# ============================================================================
# COMMAND LINE INTERFACE (Optional)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Drone Coverage with Easy Waypoint Input')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 250)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()


    run(**vars(ARGS))
