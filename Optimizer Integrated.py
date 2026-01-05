"""
Copyright (c) 2026 Lalu Muhamad Alhadad

This code was developed as part of a Master's thesis under the
Faculty of Mechanical and Aerospace Engineering,
Institut Teknologi Bandung (ITB).

Licensed under the MIT License. See LICENSE file for details.
"""


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle, Polygon
import numpy as np
import math
import time
import threading
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
import subprocess
import sys
import os
import tempfile
import importlib.util

warnings.filterwarnings('ignore')

class EnhancedPathOptimizer:
    """Enhanced path optimization focusing on coverage maximization and distance minimization"""
    
    def __init__(self, camera_fov=60, altitude=15):


        self.camera_fov = camera_fov
        self.altitude = altitude
        self.footprint_radius = altitude * math.tan(math.radians(camera_fov / 2))
    
    def calculate_coverage_from_position(self, camera_position, all_points):
        """Calculate which points are covered from a camera position"""
        covered_indices = []
        for i, point in enumerate(all_points):
            distance = math.sqrt(
                (point['x'] - camera_position['x'])**2 + 
                (point['y'] - camera_position['y'])**2
            )
            if distance <= self.footprint_radius:
                covered_indices.append(i)
        return covered_indices
    
    def generate_coverage_optimal_waypoints(self, start_position, target_points):
        """Generate waypoints using weighted set cover for optimal coverage"""
        if not target_points:
            return [start_position]
        
        waypoints = [start_position]
        uncovered_indices = set(range(len(target_points)))
        
        # Check initial coverage
        initial_coverage = self.calculate_coverage_from_position(start_position, target_points)
        uncovered_indices -= set(initial_coverage)
        
        # Greedy set cover
        while uncovered_indices:
            best_position = None
            best_score = 0
            best_coverage = set()
            
            for i in uncovered_indices:
                candidate = target_points[i]
                coverage_indices = self.calculate_coverage_from_position(candidate, target_points)
                new_coverage = set(coverage_indices) & uncovered_indices
                
                if new_coverage:
                    distance = self.euclidean_distance(waypoints[-1], candidate)
                    score = len(new_coverage) / (1 + distance * 0.01)
                    
                    if score > best_score:
                        best_score = score
                        best_position = candidate
                        best_coverage = new_coverage
            
            if best_position:
                waypoints.append(best_position)
                uncovered_indices -= best_coverage
            else:
                break
        
        return waypoints
    
    def optimize_waypoint_order_tsp(self, waypoints):
        """Optimize waypoint order using enhanced TSP"""
        if len(waypoints) <= 2:
            return waypoints
        
        n = len(waypoints)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = self.euclidean_distance(waypoints[i], waypoints[j])
        
        # Multiple restarts for better solution
        best_path = None
        best_distance = float('inf')
        
        for _ in range(5):  # 5 random restarts
            path = self.nearest_neighbor_tsp(dist_matrix)
            path = self.two_opt_improvement(path, dist_matrix)
            path_distance = self.calculate_path_distance(path, dist_matrix)
            
            if path_distance < best_distance:
                best_distance = path_distance
                best_path = path
        
        return [waypoints[i] for i in best_path]
    
    def nearest_neighbor_tsp(self, dist_matrix):
        """Greedy nearest neighbor TSP construction"""
        n = len(dist_matrix)
        unvisited = set(range(1, n))
        path = [0]
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return path
    
    def two_opt_improvement(self, path, dist_matrix):
        """2-opt improvement for TSP"""
        improved = True
        current_path = path[:]
        
        while improved:
            improved = False
            for i in range(1, len(current_path) - 2):
                for j in range(i + 1, len(current_path)):
                    if j == len(current_path) - 1:
                        continue
                    
                    old_dist = (dist_matrix[current_path[i-1]][current_path[i]] +
                               dist_matrix[current_path[j]][current_path[j+1]])
                    new_dist = (dist_matrix[current_path[i-1]][current_path[j]] +
                               dist_matrix[current_path[i]][current_path[j+1]])
                    
                    if new_dist < old_dist:
                        current_path[i:j+1] = reversed(current_path[i:j+1])
                        improved = True
        
        return current_path
    
    def calculate_path_distance(self, path, dist_matrix):
        """Calculate total path distance"""
        total = 0
        for i in range(len(path) - 1):
            total += dist_matrix[path[i]][path[i+1]]
        return total
    
    def euclidean_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

class BayesianOptimizer:
    def __init__(self, bounds, acquisition_func='ei', n_random_starts=5):
        """
        Bayesian Optimizer for grid parameters optimization
        bounds: [(x_min, x_max), (y_min, y_max)] for grid dimensions only
        """
        self.bounds = np.array(bounds)
        self.acquisition_func = acquisition_func
        self.n_random_starts = n_random_starts
        
        # Storage for observations
        self.X = []  # Input parameters (grid_x, grid_y)
        self.y = []  # Objective values (swarm fitness scores)
        
        # GP parameters
        self.length_scale = 1.0
        self.noise_level = 0.1
        self.signal_variance = 1.0

        
    def rbf_kernel(self, X1, X2, length_scale=1.0, signal_variance=1.0):
        """RBF (Gaussian) kernel for GP"""
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return signal_variance * np.exp(-0.5 / length_scale**2 * sqdist)
    
    def gp_predict(self, X_test):
        """Gaussian Process prediction"""
        if len(self.X) == 0:
            mean = np.zeros(len(X_test))
            std = np.ones(len(X_test))
            return mean, std
        
        X_train = np.array(self.X)
        y_train = np.array(self.y)
        
        # Normalize observations
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-6
        y_train_norm = (y_train - y_mean) / y_std
        
        # Kernel matrices
        K = self.rbf_kernel(X_train, X_train, self.length_scale, self.signal_variance)
        K += self.noise_level * np.eye(len(X_train))
        
        K_s = self.rbf_kernel(X_train, X_test, self.length_scale, self.signal_variance)
        K_ss = self.rbf_kernel(X_test, X_test, self.length_scale, self.signal_variance)
        
        # GP prediction
        try:
            K_inv = np.linalg.inv(K)
            mu_s = K_s.T.dot(K_inv).dot(y_train_norm)
            cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
            std_s = np.sqrt(np.diag(cov_s))
            
            # Denormalize
            mu_s = mu_s * y_std + y_mean
            std_s = std_s * y_std
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            mu_s = np.full(len(X_test), y_mean)
            std_s = np.ones(len(X_test)) * y_std
        
        return mu_s, std_s
    
    def acquisition_function(self, X_test):
        """Expected Improvement acquisition function"""
        if len(self.y) == 0:
            return np.ones(len(X_test))
        
        mu, sigma = self.gp_predict(X_test)
        
        # Expected Improvement
        f_best = np.max(self.y)
        xi = 0.01  # exploration parameter
        
        with np.errstate(divide='warn'):
            imp = mu - f_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def suggest_next_point(self):
        """Suggest next point to evaluate using acquisition function"""
        if len(self.X) < self.n_random_starts:
            # Random exploration phase
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            return x.astype(int)
        
        # Optimize acquisition function
        best_x = None
        best_acq = -np.inf
        
        # Multiple random restarts for acquisition optimization with noise
        for i in range(10):
            # Add small random noise to prevent identical restarts
            noise = np.random.normal(0, 0.1, size=len(self.bounds))
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]) + noise
            x0 = np.clip(x0, self.bounds[:, 0], self.bounds[:, 1])  # Keep within bounds
            
            # Optimize using scipy
            def neg_acquisition(x):
                x_reshaped = x.reshape(1, -1)
                return -self.acquisition_function(x_reshaped)[0]
            
            try:
                res = minimize(neg_acquisition, x0, bounds=self.bounds, method='L-BFGS-B')
                if res.success and -res.fun > best_acq:
                    best_acq = -res.fun
                    best_x = res.x
            except:
                continue
        
        if best_x is None:
            # Fallback to random
            best_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        return best_x.astype(int)
    
    def add_observation(self, x, y):
        """Add new observation to the dataset"""
        self.X.append(x)
        self.y.append(y)

class DroneSwarm:
    def __init__(self, n_drones, area_bounds, colors=None, start_formation='corner', area_shape='rectangle'):
        """
        Initialize drone swarm with dynamic region assignment
        """
        self.n_drones = n_drones
        self.area_width, self.area_height = area_bounds
        self.area_shape = area_shape
        self.start_formation = start_formation
        self.drones = []
        
        # Default colors for drones
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.colors = colors or (default_colors * ((n_drones // len(default_colors)) + 1))[:n_drones]
        
        # Initialize drone states without predefined regions
        for i in range(n_drones):
            drone = {
                'id': i,
                'color': self.colors[i],
                'position': {'x': 0, 'y': 0},
                'target_position': {'x': 0, 'y': 0},
                'waypoints': [],
                'path': [],
                'current_waypoint_index': 0,
                'distance_traveled': 0,
                'covered_points': set(),
                'covered_area_cells': set(),
                'assigned_points': set(),  # Dynamically assigned grid points
                'is_active': True,
                'mission_complete': False,
                'start_position': {'x': 0, 'y': 0}  # Starting position for each drone
            }
            self.drones.append(drone)
        
        # Set starting positions based on formation strategy
        self.set_starting_positions()
    
    def get_building_bounds(self):
        """Get the bounding box of the building shape"""
        if self.area_shape == 'circle':
            radius = min(self.area_width, self.area_height) / 2
            return {
                'x_min': -radius,
                'x_max': radius,
                'y_min': -radius,
                'y_max': radius,
                'width': radius * 2,
                'height': radius * 2
            }
        elif self.area_shape == 'triangle':
            # Equilateral triangle - use area_width as side length
            side_length = self.area_width
            height = side_length * math.sqrt(3) / 2
            return {
                'x_min': -side_length/2,
                'x_max': side_length/2,
                'y_min': -height/3,  # Changed: position triangle so centroid is at origin
                'y_max': 2*height/3,  # Changed: adjust top position
                'width': side_length,
                'height': height
            }
        elif self.area_shape in ['hexagon', 'pentagon', 'octagon']:
            # Regular polygons - use circumscribed circle
            radius = min(self.area_width, self.area_height) / 2
            return {
                'x_min': -radius,
                'x_max': radius,
                'y_min': -radius,
                'y_max': radius,
                'width': radius * 2,
                'height': radius * 2
            }
        elif self.area_shape == 'diamond':
            # Diamond (rotated square)
            size = min(self.area_width, self.area_height) / 2
            return {
                'x_min': -size,
                'x_max': size,
                'y_min': -size,
                'y_max': size,
                'width': size * 2,
                'height': size * 2
            }

        else:  # rectangle or square
            return {
                'x_min': -self.area_width/2,
                'x_max': self.area_width/2,
                'y_min': -self.area_height/2,
                'y_max': self.area_height/2,
                'width': self.area_width,
                'height': self.area_height
            }
    
    def point_in_building(self, point):
        """Check if a point is inside the building shape"""
        x, y = point['x'], point['y']
        
        if self.area_shape == 'circle':
            radius = min(self.area_width, self.area_height) / 2
            distance = math.sqrt(x*x + y*y)
            return distance <= radius
        elif self.area_shape == 'triangle':
            # Equilateral triangle with side length = area_width
            side_length = self.area_width
            height = side_length * math.sqrt(3) / 2
            
            # Triangle vertices for equilateral triangle centered at origin
            v0_x, v0_y = 0, 2*height/3                    # Top vertex
            v1_x, v1_y = -side_length/2, -height/3       # Bottom-left vertex  
            v2_x, v2_y = side_length/2, -height/3        # Bottom-right vertex
            
            # Barycentric coordinates method
            denom = (v1_y - v2_y) * (v0_x - v2_x) + (v2_x - v1_x) * (v0_y - v2_y)
            
            if abs(denom) < 1e-10:
                return False
            
            a = ((v1_y - v2_y) * (x - v2_x) + (v2_x - v1_x) * (y - v2_y)) / denom
            b = ((v2_y - v0_y) * (x - v2_x) + (v0_x - v2_x) * (y - v2_y)) / denom
            c = 1 - a - b
            
            return a >= 0 and b >= 0 and c >= 0
        
        elif self.area_shape == 'hexagon':
            # Regular hexagon
            radius = min(self.area_width, self.area_height) / 2
            # Use distance from center and check against hexagon edges
            distance = math.sqrt(x*x + y*y)
            if distance > radius:
                return False
            
            # Check hexagon constraints (simplified)
            angle = math.atan2(y, x)
            for i in range(6):
                hex_angle = i * math.pi / 3
                # Simplified hexagon check using distance to edges
                edge_distance = abs(x * math.cos(hex_angle + math.pi/2) + y * math.sin(hex_angle + math.pi/2))
                if edge_distance > radius * math.cos(math.pi/6):
                    return False
            return True
        elif self.area_shape == 'pentagon':
            # Regular pentagon
            radius = min(self.area_width, self.area_height) / 2
            distance = math.sqrt(x*x + y*y)
            if distance > radius:
                return False
            
            # Pentagon check (simplified)
            angle = math.atan2(y, x)
            for i in range(5):
                pent_angle = i * 2 * math.pi / 5 - math.pi/2
                edge_distance = abs(x * math.cos(pent_angle + math.pi/2) + y * math.sin(pent_angle + math.pi/2))
                if edge_distance > radius * math.cos(math.pi/5):
                    return False
            return True
        elif self.area_shape == 'octagon':
            # Regular octagon
            radius = min(self.area_width, self.area_height) / 2
            distance = math.sqrt(x*x + y*y)
            if distance > radius:
                return False
            
            # Octagon check
            for i in range(8):
                oct_angle = i * math.pi / 4
                edge_distance = abs(x * math.cos(oct_angle + math.pi/2) + y * math.sin(oct_angle + math.pi/2))
                if edge_distance > radius * math.cos(math.pi/8):
                    return False
            return True
        elif self.area_shape == 'diamond':
            # Diamond (rotated square)
            size = min(self.area_width, self.area_height) / 2
            # Diamond is |x| + |y| <= size
            return abs(x) + abs(y) <= size
        else:  # rectangle or square
            return (-self.area_width/2 <= x <= self.area_width/2 and 
                    -self.area_height/2 <= y <= self.area_height/2)
    
    def set_starting_positions(self):
        """Set realistic starting positions for drones based on formation strategy"""
        bounds = self.get_building_bounds()
        
        if self.start_formation == 'corner':
            self.set_corner_formation(bounds)
        elif self.start_formation == 'edge':
            self.set_edge_formation(bounds)
        elif self.start_formation == 'single_point':
            self.set_single_point_formation(bounds)
        elif self.start_formation == 'custom_points':
            self.set_custom_points_formation(bounds)
        elif self.start_formation == 'opposite_corners':
            self.set_opposite_corners_formation(bounds)
        elif self.start_formation == 'perimeter':
            self.set_perimeter_formation(bounds)
        elif self.start_formation == 'center':
            self.set_center_formation(bounds)
        else:
            # Default to corner formation
            self.set_corner_formation(bounds)
        
        print(f"Set {self.start_formation} formation for {self.n_drones} drones:")
        for drone in self.drones:
            pos = drone['start_position']
            print(f"  Drone {drone['id']}: ({pos['x']:.1f}, {pos['y']:.1f})")
    
    def set_corner_formation(self, bounds):
        """Place drones starting from corners (most realistic for inspections)"""
        # ADAPTIVE safety margin based on building size
        base_margin = 8.0
        # For small buildings, use much smaller margins
        if bounds['width'] < 30 or bounds['height'] < 30:
            safety_margin = max(1.0, min(bounds['width'], bounds['height']) * 0.1)  # 10% of smallest dimension, min 1m
        else:
            safety_margin = base_margin
        
        print(f"Corner formation: Building {bounds['width']:.1f}x{bounds['height']:.1f}m, Safety margin: {safety_margin:.1f}m")
        
        # Define corner positions with adaptive safety margin
        corners = [
            {'x': bounds['x_min'] + safety_margin, 'y': bounds['y_min'] + safety_margin},  # Bottom-left
            {'x': bounds['x_max'] - safety_margin, 'y': bounds['y_min'] + safety_margin},  # Bottom-right  
            {'x': bounds['x_max'] - safety_margin, 'y': bounds['y_max'] - safety_margin},  # Top-right
            {'x': bounds['x_min'] + safety_margin, 'y': bounds['y_max'] - safety_margin},  # Top-left
        ]
        
        if self.n_drones == 1:
            # Single drone starts from bottom-left corner (most common)
            self.drones[0]['start_position'] = corners[0].copy()
            self.drones[0]['position'] = corners[0].copy()
        else:
            # Distribute drones among corners
            for i, drone in enumerate(self.drones):
                corner_idx = i % len(corners)
                base_corner = corners[corner_idx]
                
                # If multiple drones per corner, offset them slightly with smaller offsets for small buildings
                if self.n_drones > 4:
                    offset_count = i // 4
                    max_offset = min(2.5, safety_margin)  # Smaller offsets for small buildings
                    offset_x = (offset_count % 2) * max_offset - max_offset/2
                    offset_y = (offset_count // 2) * max_offset - max_offset/2
                    
                    start_pos = {
                        'x': base_corner['x'] + offset_x,
                        'y': base_corner['y'] + offset_y
                    }
                else:
                    start_pos = base_corner.copy()
                
                # Ensure position is within bounds with smaller safety margin
                min_margin = 0.5  # Minimum 0.5m from wall
                start_pos['x'] = max(bounds['x_min'] + min_margin, min(bounds['x_max'] - min_margin, start_pos['x']))
                start_pos['y'] = max(bounds['y_min'] + min_margin, min(bounds['y_max'] - min_margin, start_pos['y']))
                
                drone['start_position'] = start_pos
                drone['position'] = start_pos.copy()
    
    def set_edge_formation(self, bounds):
        """Place drones along the edges of the building"""
        # ADAPTIVE safety margin
        if bounds['width'] < 30 or bounds['height'] < 30:
            safety_margin = max(1.0, min(bounds['width'], bounds['height']) * 0.12)
        else:
            safety_margin = 8.0
        
        print(f"Edge formation: Building {bounds['width']:.1f}x{bounds['height']:.1f}m, Safety margin: {safety_margin:.1f}m")
        
        # Define edge positions
        edges = []
        
        # Bottom edge
        edges.append({'x': bounds['x_min'] + safety_margin, 'y': bounds['y_min'] + safety_margin})
        
        if self.n_drones > 1:
            # Right edge  
            edges.append({'x': bounds['x_max'] - safety_margin, 'y': bounds['y_min'] + safety_margin})
        
        if self.n_drones > 2:
            # Top edge
            edges.append({'x': bounds['x_max'] - safety_margin, 'y': bounds['y_max'] - safety_margin})
        
        if self.n_drones > 3:
            # Left edge
            edges.append({'x': bounds['x_min'] + safety_margin, 'y': bounds['y_max'] - safety_margin})
        
        # For more drones, distribute along edges
        if self.n_drones > 4:
            # Add intermediate positions along edges
            edge_length = (bounds['width'] + bounds['height']) * 2
            spacing = edge_length / self.n_drones
            
            for i in range(4, self.n_drones):
                # Distribute remaining drones along perimeter
                progress = (i - 4) / max(1, self.n_drones - 4)
                
                safe_width = bounds['width'] - 2 * safety_margin
                safe_height = bounds['height'] - 2 * safety_margin
                
                if progress < 0.25:  # Bottom edge
                    x = bounds['x_min'] + safety_margin + progress * 4 * safe_width
                    y = bounds['y_min'] + safety_margin
                elif progress < 0.5:  # Right edge
                    x = bounds['x_max'] - safety_margin
                    y = bounds['y_min'] + safety_margin + (progress - 0.25) * 4 * safe_height
                elif progress < 0.75:  # Top edge
                    x = bounds['x_max'] - safety_margin - (progress - 0.5) * 4 * safe_width
                    y = bounds['y_max'] - safety_margin
                else:  # Left edge
                    x = bounds['x_min'] + safety_margin
                    y = bounds['y_max'] - safety_margin - (progress - 0.75) * 4 * safe_height
                
                edges.append({'x': x, 'y': y})
        
        # Assign positions to drones
        for i, drone in enumerate(self.drones):
            if i < len(edges):
                drone['start_position'] = edges[i].copy()
                drone['position'] = edges[i].copy()
    
    def set_single_point_formation(self, bounds):
        """All drones start from a single launch point (realistic for small operations)"""
        # ADAPTIVE launch point positioning based on building size
        if bounds['width'] < 30 or bounds['height'] < 30:
            # Small building - place launch point closer to edge
            margin = max(1.0, min(bounds['width'], bounds['height']) * 0.15)  # 15% margin for small buildings
        else:
            # Normal building - use standard margin
            margin = 10.0
        
        print(f"Single point formation: Building {bounds['width']:.1f}x{bounds['height']:.1f}m, Launch margin: {margin:.1f}m")
        
        # Common launch point - usually accessible corner or edge
        launch_point = {
            'x': bounds['x_min'] + margin,
            'y': bounds['y_min'] + margin
        }
        
        for i, drone in enumerate(self.drones):
            # Slight offset for multiple drones to avoid collision - adaptive to building size
            if bounds['width'] < 30:
                # Small building - use smaller offsets
                offset_spacing = max(1.0, margin / 3)
                offset_x = (i % 3) * offset_spacing - offset_spacing
                offset_y = (i // 3) * offset_spacing
            else:
                # Normal building - use standard offsets
                offset_x = (i % 3) * 3 - 3  # -3, 0, 3 meter pattern
                offset_y = (i // 3) * 3     # 0, 3, 6 meter pattern
            
            start_pos = {
                'x': launch_point['x'] + offset_x,
                'y': launch_point['y'] + offset_y
            }
            
            # Ensure within bounds with minimal margin
            min_margin = 0.5
            start_pos['x'] = max(bounds['x_min'] + min_margin, min(bounds['x_max'] - min_margin, start_pos['x']))
            start_pos['y'] = max(bounds['y_min'] + min_margin, min(bounds['y_max'] - min_margin, start_pos['y']))
            
            drone['start_position'] = start_pos
            drone['position'] = start_pos.copy()
    
    def set_custom_points_formation(self, bounds):
        """Set drones at strategically chosen custom points for inspection"""
        # Predefined strategic positions for building inspection
        strategic_points = [
            {'x': bounds['x_min'] + 15, 'y': bounds['y_min'] + 15},  # Near bottom-left
            {'x': bounds['x_max'] - 15, 'y': bounds['y_min'] + 15},  # Near bottom-right
            {'x': 0, 'y': bounds['y_min'] + 20},                     # Bottom center
            {'x': bounds['x_min'] + 15, 'y': bounds['y_max'] - 15},  # Near top-left
            {'x': bounds['x_max'] - 15, 'y': bounds['y_max'] - 15},  # Near top-right
            {'x': 0, 'y': bounds['y_max'] - 20},                     # Top center
            {'x': bounds['x_min'] + 20, 'y': 0},                     # Left center
            {'x': bounds['x_max'] - 20, 'y': 0},                     # Right center
        ]
        
        for i, drone in enumerate(self.drones):
            if i < len(strategic_points):
                pos = strategic_points[i].copy()
            else:
                # For extra drones, use corner formation as fallback
                corner_idx = i % 4
                corners = [
                    {'x': bounds['x_min'] + 8, 'y': bounds['y_min'] + 8},
                    {'x': bounds['x_max'] - 8, 'y': bounds['y_min'] + 8},
                    {'x': bounds['x_max'] - 8, 'y': bounds['y_max'] - 8},
                    {'x': bounds['x_min'] + 8, 'y': bounds['y_max'] - 8},
                ]
                pos = corners[corner_idx].copy()
            
            # Ensure position is within building shape
            if self.point_in_building(pos):
                drone['start_position'] = pos
                drone['position'] = pos.copy()
            else:
                # Fallback to safe corner position
                safe_pos = {'x': bounds['x_min'] + 10, 'y': bounds['y_min'] + 10}
                drone['start_position'] = safe_pos
                drone['position'] = safe_pos.copy()
    
    def set_opposite_corners_formation(self, bounds):
        """Place drones at opposite corners for maximum coverage spread"""
        safety_margin = 8.0
        
        positions = [
            {'x': bounds['x_min'] + safety_margin, 'y': bounds['y_min'] + safety_margin},  # Bottom-left
            {'x': bounds['x_max'] - safety_margin, 'y': bounds['y_max'] - safety_margin},  # Top-right
            {'x': bounds['x_max'] - safety_margin, 'y': bounds['y_min'] + safety_margin},  # Bottom-right
            {'x': bounds['x_min'] + safety_margin, 'y': bounds['y_max'] - safety_margin},  # Top-left
        ]
        
        for i, drone in enumerate(self.drones):
            pos_idx = i % len(positions)
            
            # Add slight offset for multiple drones per position
            offset_count = i // len(positions)
            offset_x = (offset_count % 2) * 4 - 2
            offset_y = (offset_count // 2) * 4 - 2
            
            start_pos = {
                'x': positions[pos_idx]['x'] + offset_x,
                'y': positions[pos_idx]['y'] + offset_y
            }
            
            # Ensure within bounds
            start_pos['x'] = max(bounds['x_min'] + 5, min(bounds['x_max'] - 5, start_pos['x']))
            start_pos['y'] = max(bounds['y_min'] + 5, min(bounds['y_max'] - 5, start_pos['y']))
            
            drone['start_position'] = start_pos
            drone['position'] = start_pos.copy()
    
    def set_perimeter_formation(self, bounds):
        """Distribute drones evenly around the building perimeter"""
        safety_margin = 8.0
        perimeter_points = []
        
        # Calculate perimeter points
        if self.area_shape == 'circle':
            # For circular buildings
            radius = min(self.area_width, self.area_height) / 2 - safety_margin
            for i in range(self.n_drones):
                angle = 2 * math.pi * i / self.n_drones
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                perimeter_points.append({'x': x, 'y': y})
        else:
            # For rectangular buildings
            perimeter_length = 2 * (bounds['width'] + bounds['height'] - 4 * safety_margin)
            segment_length = perimeter_length / self.n_drones
            
            for i in range(self.n_drones):
                distance_along_perimeter = i * segment_length
                
                # Determine which side of rectangle we're on
                width_safe = bounds['width'] - 2 * safety_margin
                height_safe = bounds['height'] - 2 * safety_margin
                
                if distance_along_perimeter < width_safe:
                    # Bottom edge
                    x = bounds['x_min'] + safety_margin + distance_along_perimeter
                    y = bounds['y_min'] + safety_margin
                elif distance_along_perimeter < width_safe + height_safe:
                    # Right edge
                    x = bounds['x_max'] - safety_margin
                    y = bounds['y_min'] + safety_margin + (distance_along_perimeter - width_safe)
                elif distance_along_perimeter < 2 * width_safe + height_safe:
                    # Top edge
                    x = bounds['x_max'] - safety_margin - (distance_along_perimeter - width_safe - height_safe)
                    y = bounds['y_max'] - safety_margin
                else:
                    # Left edge
                    x = bounds['x_min'] + safety_margin
                    y = bounds['y_max'] - safety_margin - (distance_along_perimeter - 2 * width_safe - height_safe)
                
                perimeter_points.append({'x': x, 'y': y})
        
        # Assign perimeter positions to drones
        for i, drone in enumerate(self.drones):
            drone['start_position'] = perimeter_points[i].copy()
            drone['position'] = perimeter_points[i].copy()

    def set_center_formation(self, bounds):
        """Place drones at center (0,0) or in straight line formation at center"""
        print(f"Center formation: {self.n_drones} drones at building center")
        
        if self.n_drones == 1:
            # Single drone at exact center
            center_pos = {'x': 0.0, 'y': 0.0}
            
            self.drones[0]['start_position'] = center_pos.copy()
            self.drones[0]['position'] = center_pos.copy()
            
            print(f"  Single drone positioned at center: (0.0, 0.0)")
        
        else:
            # Multiple drones in straight line formation at center
            
            # Determine line direction and spacing based on building shape
            building_width = bounds['width']
            building_height = bounds['height']
            
            # Choose line direction based on building aspect ratio
            if building_width >= building_height:
                # Wide building: arrange drones horizontally (along X-axis)
                line_direction = 'horizontal'
                max_line_length = min(building_width * 0.3, 20)  # 30% of width, max 20m
            else:
                # Tall building: arrange drones vertically (along Y-axis)  
                line_direction = 'vertical'
                max_line_length = min(building_height * 0.3, 20)  # 30% of height, max 20m
            
            # Calculate spacing between drones
            if self.n_drones > 1:
                spacing = min(max_line_length / (self.n_drones - 1), 1.0)  # Max 4m spacing
            else:
                spacing = 0
            
            # Calculate starting offset to center the line
            total_line_length = spacing * (self.n_drones - 1)
            start_offset = -total_line_length / 2
            
            print(f"  {line_direction.capitalize()} line formation:")
            print(f"  Line length: {total_line_length:.1f}m, Spacing: {spacing:.1f}m")
            
            # Position drones along the line
            for i, drone in enumerate(self.drones):
                if line_direction == 'horizontal':
                    # Horizontal line (X-axis)
                    pos_x = start_offset + (i * spacing)
                    pos_y = 0.0
                else:
                    # Vertical line (Y-axis)
                    pos_x = 0.0
                    pos_y = start_offset + (i * spacing)
                
                start_pos = {'x': pos_x, 'y': pos_y}
                
                drone['start_position'] = start_pos
                drone['position'] = start_pos.copy()
                
                print(f"  Drone {drone['id']}: ({pos_x:.1f}, {pos_y:.1f})")
    
    def get_swarm_statistics(self):
        """Get overall swarm performance statistics"""
        total_distance = sum(drone['distance_traveled'] for drone in self.drones)
        total_covered_points = len(set().union(*[drone['covered_points'] for drone in self.drones]))
        total_covered_cells = len(set().union(*[drone['covered_area_cells'] for drone in self.drones]))
        
        active_drones = sum(1 for drone in self.drones if drone['is_active'])
        completed_drones = sum(1 for drone in self.drones if drone['mission_complete'])
        
        return {
            'total_distance': total_distance,
            'total_covered_points': total_covered_points,
            'total_covered_cells': total_covered_cells,
            'active_drones': active_drones,
            'completed_drones': completed_drones,
            'average_distance': total_distance / max(self.n_drones, 1)
        }

class LiveDroneSwarmSimulation:

    def calculate_optimal_grid_bounds(self):
        """
        Calculate optimal grid bounds based on building dimensions and camera specs
        Returns dynamic min/max grid values that make sense for the building size
        """
        # Get building dimensions
        bounds = self.get_building_bounds()
        building_width = bounds['width']
        building_height = bounds['height']
        
        # Calculate camera footprint coverage area
        footprint_radius = self.altitude * math.tan(math.radians(self.camera_fov / 2))
        footprint_diameter = footprint_radius * 2
        
        # Calculate reasonable grid density based on coverage overlap
        # Rule: Each grid point should cover ~70% of footprint area for efficiency
        optimal_spacing = footprint_diameter * 0.7
        
        # Calculate optimal grid dimensions
        optimal_x = max(3, int(building_width / optimal_spacing))
        optimal_y = max(3, int(building_height / optimal_spacing))
        
        # Calculate bounds with reasonable margins
        # Minimum: At least 3 points per dimension
        min_x = max(3, int(optimal_x * 0.5))  # 50% of optimal (sparse)
        min_y = max(3, int(optimal_y * 0.5))
        
        # Maximum: Dense coverage but not excessive
        # Rule: Don't exceed 1 point per 2m for very dense coverage
        max_x = min(int(building_width / 2), int(optimal_x * 2))  # 200% of optimal (dense)
        max_y = min(int(building_height / 2), int(optimal_y * 2))
        
        # Ensure reasonable absolute limits
        min_x = max(min_x, 3)   # Absolute minimum
        min_y = max(min_y, 3)
        max_x = min(max_x, 50)  # Absolute maximum (performance limit)
        max_y = min(max_y, 40)
        
        return {
            'x_min': min_x,
            'x_max': max_x,
            'y_min': min_y, 
            'y_max': max_y,
            'optimal_x': optimal_x,
            'optimal_y': optimal_y,
            'footprint_diameter': footprint_diameter,
            'recommended_spacing': optimal_spacing
        }

    def __init__(self):
        # Initialize main window
        import random

        random.seed(42)  # For reproducibility
        np.random.seed(42)  # For reproducibility

        self.root = tk.Tk()
        self.root.title("🚁 Dynamic Region Drone Swarm System")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#f0f0f0')
        
        # Simulation parameters
        self.area_width = 100
        self.area_height = 80
        self.area_shape = 'rectangle'
        self.area_size = 100
        self.grid_points_x = 10
        self.grid_points_y = 8
        self.n_drones = 2
        self.camera_fov = 60
        self.altitude = 15
        self.drone_speed = 5
        self.strategy = 'dynamic_coverage'

        self.enhanced_optimizer = EnhancedPathOptimizer(self.camera_fov, self.altitude)


        self.gp_colorbar = None
        self.acq_colorbar = None
        self.param_colorbar = None
        
        # Dynamic assignment parameters - Please set the values here for setting your swarm behavior preference
        self.overlap_penalty = 2.0  # Penalty for covering same areas
        self.distance_weight = 0.5  # Weight for distance in assignment
        self.coverage_weight = 1.5  # Weight for coverage potential
        self.start_formation = 'corner'  # Starting formation strategy
        
        # Camera parameters
        self.camera_tilt = 0
        self.area_resolution = 2.0
        
        # Path optimization strategy
        self.path_strategy = 'hybrid'
        
        # Swarm system
        self.swarm = None
        
        # Bayesian Optimization
        # Calculate initial grid bounds FIRST
        initial_bounds = self.calculate_optimal_grid_bounds()
        self.bayesian_optimizer = BayesianOptimizer(bounds=[
            (initial_bounds['x_min'], initial_bounds['x_max']),
            (initial_bounds['y_min'], initial_bounds['y_max'])
        ])
        self.current_grid_bounds = initial_bounds
        self.optimization_running = False
        self.optimization_thread = None
        self.optimization_history = []
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.start_time = 0
        self.elapsed_time = 0
        
        # Global grid points for reference
        self.global_grid_points = []

        grid_bounds = self.calculate_optimal_grid_bounds()
        self.bayesian_optimizer = BayesianOptimizer(bounds=[
                (grid_bounds['x_min'], grid_bounds['x_max']),
                (grid_bounds['y_min'], grid_bounds['y_max'])
            ])
    
        # Store bounds for UI updates
        self.current_grid_bounds = grid_bounds
        
        self.setup_gui()
        self.initialize_swarm()
        self.update_display()
        
    def setup_gui(self):
        # Main container with notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Simulation Tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Swarm Simulation")
        
        # Optimization Tab
        self.opt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.opt_frame, text="Swarm Optimization")
        
        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Swarm Analysis")
        
        self.setup_simulation_tab()
        self.setup_optimization_tab()
        self.setup_analysis_tab()
    
    def setup_simulation_tab(self):
        # Header
        header_frame = ttk.Frame(self.sim_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🚁 Dynamic Region Drone Swarm System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="AI-powered multi-drone coordination with dynamic region assignment")
        subtitle_label.pack()
        
        # Controls frame
        controls_frame = ttk.LabelFrame(self.sim_frame, text="Swarm Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create control variables
        self.area_shape_var = tk.StringVar(value='rectangle')
        self.area_size_var = tk.IntVar(value=100)
        self.area_width_var = tk.IntVar(value=100)
        self.area_height_var = tk.IntVar(value=80)
        self.grid_points_x_var = tk.IntVar(value=10)
        self.grid_points_y_var = tk.IntVar(value=8)
        self.n_drones_var = tk.IntVar(value=2)
        self.camera_fov_var = tk.IntVar(value=60)
        self.altitude_var = tk.IntVar(value=15)
        self.drone_speed_var = tk.DoubleVar(value=5.0)
        self.strategy_var = tk.StringVar(value='dynamic_coverage')
        self.overlap_penalty_var = tk.DoubleVar(value=2.0)
        self.distance_weight_var = tk.DoubleVar(value=0.5)
        self.coverage_weight_var = tk.DoubleVar(value=1.5)
        self.start_formation_var = tk.StringVar(value='corner')
        
        # Parameter controls
        params_frame = ttk.Frame(controls_frame)
        params_frame.pack(fill=tk.X)
        
        # First row - Basic parameters
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="Building Shape:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        shape_combo = ttk.Combobox(row1, textvariable=self.area_shape_var, width=12,
                                  values=['rectangle', 'square', 'circle', 'triangle', 'hexagon', 'pentagon', 'octagon', 'diamond'], 
                                  state='readonly')
        shape_combo.grid(row=0, column=1, padx=(0, 15))
        shape_combo.bind('<<ComboboxSelected>>', lambda e: self.on_shape_change())
        
        ttk.Label(row1, text="Width (m):").grid(row=0, column=2, sticky='w', padx=(0, 5))
        ttk.Spinbox(row1, from_=20, to=200, textvariable=self.area_width_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Label(row1, text="Height (m):").grid(row=0, column=4, sticky='w', padx=(0, 5))
        ttk.Spinbox(row1, from_=20, to=200, textvariable=self.area_height_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=5, padx=(0, 15))
        
        # Second row - Grid and drones
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row2, text="Number of Drones:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        ttk.Spinbox(row2, from_=1, to=8, textvariable=self.n_drones_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=1, padx=(0, 15))
        
        grid_bounds = self.calculate_optimal_grid_bounds()
        
        ttk.Label(row2, text="Grid Points X:").grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.grid_x_spinbox = ttk.Spinbox(row2, 
                                         from_=grid_bounds['x_min'], 
                                         to=grid_bounds['x_max'], 
                                         textvariable=self.grid_points_x_var, 
                                         width=8,
                                         command=self.on_parameter_change)
        self.grid_x_spinbox.grid(row=0, column=3, padx=(0, 15))
        
        ttk.Label(row2, text="Grid Points Y:").grid(row=0, column=4, sticky='w', padx=(0, 5))
        self.grid_y_spinbox = ttk.Spinbox(row2, 
                                         from_=grid_bounds['y_min'], 
                                         to=grid_bounds['y_max'], 
                                         textvariable=self.grid_points_y_var, 
                                         width=8,
                                         command=self.on_parameter_change)
        self.grid_y_spinbox.grid(row=0, column=5, padx=(0, 15))
        
        # Store bounds for later use
        self.current_grid_bounds = grid_bounds
        
        # Add recommendation row under row2
        row2_info = ttk.Frame(params_frame)
        row2_info.pack(fill=tk.X, pady=(0, 5))
        
        # Add helpful label showing recommended values
        recommend_text = f"Recommended: {grid_bounds['optimal_x']}x{grid_bounds['optimal_y']} (Range: {grid_bounds['x_min']}-{grid_bounds['x_max']} x {grid_bounds['y_min']}-{grid_bounds['y_max']})"
        ttk.Label(row2_info, text=recommend_text, font=('Arial', 8), foreground='blue').grid(
            row=0, column=0, columnspan=6, sticky='w', padx=(0, 5)
        )
        
        # Third row - Camera and strategy
        row3 = ttk.Frame(params_frame)
        row3.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row3, text="Camera FOV (°):").grid(row=0, column=0, sticky='w', padx=(0, 5))
        ttk.Spinbox(row3, from_=30, to=120, textvariable=self.camera_fov_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(row3, text="Speed (m/s):").grid(row=0, column=2, sticky='w', padx=(0, 5))
        ttk.Spinbox(row3, from_=1, to=15, increment=0.5, textvariable=self.drone_speed_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Label(row3, text="Assignment Strategy:").grid(row=0, column=4, sticky='w', padx=(0, 5))
        strategy_combo = ttk.Combobox(row3, textvariable=self.strategy_var, width=15,
                                     values=['dynamic_coverage', 'competitive_assignment', 'cooperative_planning'], 
                                     state='readonly')
        strategy_combo.grid(row=0, column=5, padx=(0, 15))
        strategy_combo.bind('<<ComboboxSelected>>', lambda e: self.on_parameter_change())
        
        # Fifth row - Path optimization strategy
        row5 = ttk.Frame(params_frame)
        row5.pack(fill=tk.X, pady=(0, 5))
        
        self.path_strategy_var = tk.StringVar(value='hybrid')
        ttk.Label(row5, text="Path Strategy:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        path_combo = ttk.Combobox(row5, textvariable=self.path_strategy_var, width=12,
                                 values=['enhanced_coverage', 'enhanced_hybrid', 'smart_pov', 'coverage', 'hybrid'], 
                                 state='readonly')
        path_combo.grid(row=0, column=1, padx=(0, 15))
        path_combo.bind('<<ComboboxSelected>>', lambda e: self.on_parameter_change())
        
        ttk.Label(row5, text="Altitude (m):").grid(row=0, column=2, sticky='w', padx=(0, 5))
        ttk.Spinbox(row5, from_=5, to=50, textvariable=self.altitude_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=3, padx=(0, 15))
        
        # Fourth row - Dynamic assignment parameters
        row4 = ttk.Frame(params_frame)
        row4.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row4, text="Start Formation:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        formation_combo = ttk.Combobox(row4, textvariable=self.start_formation_var, width=12,
                                     values=['corner', 'edge', 'single_point', 'custom_points', 'opposite_corners', 'perimeter', 'center'], 
                                     state='readonly')
        formation_combo.grid(row=0, column=1, padx=(0, 15))
        formation_combo.bind('<<ComboboxSelected>>', lambda e: self.on_parameter_change())
        
        ttk.Label(row4, text="Overlap Penalty:").grid(row=0, column=2, sticky='w', padx=(0, 5))
        ttk.Spinbox(row4, from_=0.5, to=5.0, increment=0.1, textvariable=self.overlap_penalty_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Label(row4, text="Distance Weight:").grid(row=0, column=4, sticky='w', padx=(0, 5))
        ttk.Spinbox(row4, from_=0.1, to=2.0, increment=0.1, textvariable=self.distance_weight_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=5, padx=(0, 15))
        
        # Sixth row - Additional parameters
        row6 = ttk.Frame(params_frame)
        row6.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row6, text="Coverage Weight:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        ttk.Spinbox(row6, from_=0.5, to=3.0, increment=0.1, textvariable=self.coverage_weight_var, width=8,
                   command=self.on_parameter_change).grid(row=0, column=1, padx=(0, 15))
        
        # Action buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_btn = ttk.Button(buttons_frame, text="🚀 Start Swarm Mission", command=self.start_mission)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_btn = ttk.Button(buttons_frame, text="Pause", command=self.pause_mission, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_mission)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Optimization buttons
        ttk.Separator(buttons_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.optimize_btn = ttk.Button(buttons_frame, text="🧠 Optimize Grid", 
                                      command=self.start_optimization, style='Accent.TButton')
        self.optimize_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_opt_btn = ttk.Button(buttons_frame, text="Stop Optimization", 
                                      command=self.stop_optimization, state='disabled')
        self.stop_opt_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Reassign button for dynamic assignment
        ttk.Separator(buttons_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.reassign_btn = ttk.Button(buttons_frame, text="🔄 Reassign Points", 
                                      command=self.reassign_grid_points)
        self.reassign_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Waypoint export button
        ttk.Separator(buttons_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.export_waypoints_btn = ttk.Button(buttons_frame, text="📋 Copy Waypoints", 
                                             command=self.copy_waypoints_to_clipboard)
        self.export_waypoints_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.run_pybullet_btn = ttk.Button(buttons_frame, text="🚁 Run PyBullet Sim", 
                                        command=self.run_pybullet_simulation,
                                        style='Accent.TButton')
        self.run_pybullet_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Info display
        self.swarm_info_label = ttk.Label(buttons_frame, text="")
        self.swarm_info_label.pack(side=tk.RIGHT, padx=(15, 0))
        
        # Main content area
        content_frame = ttk.Frame(self.sim_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - simulation canvas
        canvas_frame = ttk.LabelFrame(content_frame, text="Dynamic Swarm Simulation", padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right side - status panel
        status_frame = ttk.LabelFrame(content_frame, text="Swarm Status", padding=10)
        status_frame.pack(side=tk.RIGHT, fill=tk.Y, ipadx=20)
        
        # Status variables
        self.mission_time_var = tk.StringVar(value="00:00")
        self.active_drones_var = tk.StringVar(value="0 / 0")
        self.total_distance_var = tk.StringVar(value="0.0 m")
        self.swarm_coverage_var = tk.StringVar(value="0%")
        self.overlap_ratio_var = tk.StringVar(value="0%")
        self.efficiency_var = tk.StringVar(value="0.000")
        
        # Status display
        status_items = [
            ("Mission Time:", self.mission_time_var),
            ("Active Drones:", self.active_drones_var),
            ("Total Distance:", self.total_distance_var),
            ("Swarm Coverage:", self.swarm_coverage_var),
            ("Overlap Ratio:", self.overlap_ratio_var),
            ("Efficiency:", self.efficiency_var)
        ]
        
        for label_text, var in status_items:
            frame = ttk.Frame(status_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Individual drone status
        ttk.Label(status_frame, text="Dynamic Assignment Status:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(15, 5))
        
        # Scrollable frame for drone status
        drone_canvas = tk.Canvas(status_frame, height=300)
        drone_scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=drone_canvas.yview)
        self.drone_status_frame = ttk.Frame(drone_canvas)
        
        self.drone_status_frame.bind(
            "<Configure>",
            lambda e: drone_canvas.configure(scrollregion=drone_canvas.bbox("all"))
        )
        
        drone_canvas.create_window((0, 0), window=self.drone_status_frame, anchor="nw")
        drone_canvas.configure(yscrollcommand=drone_scrollbar.set)
        
        drone_canvas.pack(side="left", fill="both", expand=True)
        drone_scrollbar.pack(side="right", fill="y")
        
        # Progress bars
        ttk.Label(status_frame, text="Overall Progress:").pack(anchor='w', pady=(10, 0))
        self.swarm_progress_var = tk.DoubleVar()
        self.swarm_progress_bar = ttk.Progressbar(status_frame, variable=self.swarm_progress_var, maximum=100)
        self.swarm_progress_bar.pack(fill=tk.X, pady=2)

    def update_grid_bounds(self):
        """Update grid bounds when building size or camera parameters change"""
        new_bounds = self.calculate_optimal_grid_bounds()
        
        # Update Bayesian optimizer bounds
        self.bayesian_optimizer.bounds = np.array([
            [new_bounds['x_min'], new_bounds['x_max']],
            [new_bounds['y_min'], new_bounds['y_max']]
        ])
        
        # Update GUI spinbox limits
        if hasattr(self, 'grid_x_spinbox'):
            self.grid_x_spinbox.config(from_=new_bounds['x_min'], to=new_bounds['x_max'])
        if hasattr(self, 'grid_y_spinbox'):
            self.grid_y_spinbox.config(from_=new_bounds['y_min'], to=new_bounds['y_max'])
        
        # Update current bounds
        self.current_grid_bounds = new_bounds
        
        # Clamp current values to new bounds
        current_x = self.grid_points_x_var.get()
        current_y = self.grid_points_y_var.get()
        
        self.grid_points_x_var.set(max(new_bounds['x_min'], min(new_bounds['x_max'], current_x)))
        self.grid_points_y_var.set(max(new_bounds['y_min'], min(new_bounds['y_max'], current_y)))

    def display_grid_recommendations(self):
        """Display grid recommendations in the UI"""
        if hasattr(self, 'current_grid_bounds'):
            bounds = self.current_grid_bounds
            
            print(f"\n📊 Grid Recommendations for {self.area_width}m x {self.area_height}m building:")
            print(f"Camera: {self.camera_fov}° FOV at {self.altitude}m altitude")
            print(f"Footprint diameter: {bounds['footprint_diameter']:.1f}m")
            print(f"Recommended spacing: {bounds['recommended_spacing']:.1f}m")
            print(f"Optimal grid: {bounds['optimal_x']}x{bounds['optimal_y']}")
            print(f"Valid range: X[{bounds['x_min']}-{bounds['x_max']}], Y[{bounds['y_min']}-{bounds['y_max']}]")

    
    def setup_optimization_tab(self):
        # Header
        opt_header = ttk.Frame(self.opt_frame)
        opt_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(opt_header, text="🧠 Dynamic Assignment Optimization Dashboard", 
                 font=('Arial', 16, 'bold')).pack()
        ttk.Label(opt_header, text="AI-powered optimization for dynamic region assignment").pack()
        
        # Optimization controls
        opt_controls = ttk.LabelFrame(self.opt_frame, text="Optimization Settings", padding=10)
        opt_controls.pack(fill=tk.X, pady=(0, 10))
        
        controls_row = ttk.Frame(opt_controls)
        controls_row.pack()
        
        ttk.Label(controls_row, text="Max Iterations:").grid(row=0, column=0, padx=(0, 5))
        self.max_iterations_var = tk.IntVar(value=25)
        ttk.Spinbox(controls_row, from_=15, to=50, textvariable=self.max_iterations_var, width=8).grid(row=0, column=1, padx=(0, 15))
        
        self.optimization_status_var = tk.StringVar(value="Ready to optimize dynamic assignment")
        ttk.Label(controls_row, textvariable=self.optimization_status_var, 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(15, 0))
        
        # Optimization plots
        plots_frame = ttk.Frame(self.opt_frame)
        plots_frame.pack(fill=tk.BOTH, expand=True)
        
        self.opt_fig, ((self.gp_ax, self.acq_ax), (self.score_ax, self.param_ax)) = plt.subplots(2, 2, figsize=(12, 10))
        self.opt_canvas = FigureCanvasTkAgg(self.opt_fig, plots_frame)
        self.opt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_optimization_plots()
    
    def setup_analysis_tab(self):
        # Header
        analysis_header = ttk.Frame(self.analysis_frame)
        analysis_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(analysis_header, text="📊 Dynamic Assignment Analysis", 
                 font=('Arial', 16, 'bold')).pack()
        ttk.Label(analysis_header, text="Performance metrics and dynamic assignment analysis").pack()
        
        # Analysis content
        analysis_content = ttk.Frame(self.analysis_frame)
        analysis_content.pack(fill=tk.BOTH, expand=True)
        
        # Analysis plots
        self.analysis_fig, ((self.coverage_ax, self.efficiency_ax), (self.assignment_ax, self.overlap_ax)) = plt.subplots(2, 2, figsize=(12, 10))
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, analysis_content)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_shape_change(self):
        """Handle building shape change"""
        shape = self.area_shape_var.get()
        
        if shape == 'square':
            # Make height equal to width for square
            width = self.area_width_var.get()
            self.area_height_var.set(width)
        elif shape == 'circle':
            # For circle, use width as diameter
            width = self.area_width_var.get()
            self.area_height_var.set(width)

        elif shape == 'triangle':
            width = self.area_width_var.get()
            height = self.area_height_var.get()
            side_length = max(width, height)
            self.area_width_var.set(side_length)
            self.area_height_var.set(side_length)
        
        self.on_parameter_change()
    
    def initialize_swarm(self):
        """Initialize or reinitialize the drone swarm"""
        self.update_parameters()
        self.swarm = DroneSwarm(self.n_drones, (self.area_width, self.area_height), 
                               start_formation=self.start_formation, area_shape=self.area_shape)
        self.generate_global_grid()
        
        # Dynamic assignment of grid points to drones
        self.assign_grid_points_dynamically()
        
        # Generate paths for each drone based on their assigned points
        self.generate_swarm_paths()
        self.update_swarm_info()
        self.create_drone_status_widgets()
        self.draw()
    
    def update_parameters(self):
        """Update simulation parameters from GUI"""
        self.area_shape = self.area_shape_var.get()
        self.area_size = self.area_size_var.get()
        self.area_width = self.area_width_var.get()
        self.area_height = self.area_height_var.get()
        self.grid_points_x = self.grid_points_x_var.get()
        self.grid_points_y = self.grid_points_y_var.get()
        self.n_drones = self.n_drones_var.get()
        self.camera_fov = self.camera_fov_var.get()
        self.altitude = self.altitude_var.get()
        self.drone_speed = self.drone_speed_var.get()
        self.strategy = self.strategy_var.get()
        self.path_strategy = self.path_strategy_var.get()
        self.overlap_penalty = self.overlap_penalty_var.get()
        self.distance_weight = self.distance_weight_var.get()
        self.coverage_weight = self.coverage_weight_var.get()
        self.start_formation = self.start_formation_var.get()
    
    def generate_global_grid(self):
        """Generate the global grid points across the entire area with shape-aware distribution"""
        bounds = self.get_building_bounds()
        
        # ADAPTIVE safety margin for grid points based on building size
        base_margin = 5.0
        if bounds['width'] < 30 or bounds['height'] < 30:
            safety_margin = max(0.5, min(bounds['width'], bounds['height']) * 0.05)
        else:
            safety_margin = base_margin
        
        print(f"Grid generation: Building {bounds['width']:.1f}x{bounds['height']:.1f}m, Grid safety margin: {safety_margin:.1f}m")
        
        # Calculate effective grid size based on building dimensions and shape
        min_spacing = 1.5
        max_spacing = 8.0
        
        safe_width = bounds['width'] - 2 * safety_margin
        safe_height = bounds['height'] - 2 * safety_margin
        
        effective_grid_x = max(3, min(self.grid_points_x, int(safe_width / min_spacing)))
        effective_grid_y = max(3, min(self.grid_points_y, int(safe_height / min_spacing)))
        
        # For very small buildings, still ensure reasonable coverage
        if safe_width < 10:
            effective_grid_x = max(3, min(effective_grid_x, 4))
        if safe_height < 10:
            effective_grid_y = max(3, min(effective_grid_y, 4))
        
        # For 15m buildings, allow more points for better coverage
        if 10 <= safe_width <= 20:
            effective_grid_x = max(4, min(effective_grid_x, 8))
        if 10 <= safe_height <= 20:
            effective_grid_y = max(4, min(effective_grid_y, 6))
        
        print(f"Grid adjustment: {self.grid_points_x}x{self.grid_points_y} → {effective_grid_x}x{effective_grid_y}")
        
        # Generate grid points based on building shape
        self.global_grid_points = []
        
        if self.area_shape == 'triangle':
            self.generate_triangle_grid(bounds, safety_margin, effective_grid_x, effective_grid_y)
        elif self.area_shape == 'circle':
            self.generate_circle_grid(bounds, safety_margin, effective_grid_x, effective_grid_y)
        elif self.area_shape in ['hexagon', 'pentagon', 'octagon']:
            self.generate_polygon_grid(bounds, safety_margin, effective_grid_x, effective_grid_y)
        elif self.area_shape == 'diamond':
            self.generate_diamond_grid(bounds, safety_margin, effective_grid_x, effective_grid_y)
        else:
            # Rectangle/square - use original rectangular grid
            self.generate_rectangular_grid(bounds, safety_margin, effective_grid_x, effective_grid_y)
        
        print(f"Generated {len(self.global_grid_points)} grid points with shape-aware distribution")
        
        # Update the displayed grid size for information
        self.effective_grid_x = effective_grid_x
        self.effective_grid_y = effective_grid_y
    
    def generate_triangle_grid(self, bounds, safety_margin, grid_x, grid_y):
        """
        Generate grid for equilateral triangle with proper geometry
        """
        side_length = self.area_width
        height = side_length * math.sqrt(3) / 2
        
        print(f"Triangle grid: Equilateral triangle side={side_length:.1f}m, height={height:.1f}m")
        print(f"Safety margin: {safety_margin:.1f}m")
        
        # Apply safety margin by scaling the triangle inward
        margin_ratio = max(0.1, 1 - (safety_margin / min(side_length/2, height * 0.5)))
        
        safe_side_length = side_length * margin_ratio
        safe_height = height * margin_ratio
        
        print(f"Safe triangle: side={safe_side_length:.1f}m, height={safe_height:.1f}m")
        
        # Triangle vertices for the safe (scaled) triangle, centered at origin
        top_vertex_y = 2*safe_height/3
        bottom_y = -safe_height/3
        
        # Generate points row by row from bottom to top
        for row in range(grid_y):
            if grid_y > 1:
                y_progress = row / (grid_y - 1)  # 0 to 1
            else:
                y_progress = 0.5
                
            # Y coordinate from bottom to top of safe triangle
            y = bottom_y + y_progress * (top_vertex_y - bottom_y)
            
            # Calculate the maximum width at this height for equilateral triangle
            # Distance from bottom of triangle
            height_from_bottom = y - bottom_y
            total_triangle_height = safe_height
            
            if total_triangle_height > 0:
                # For equilateral triangle: width decreases linearly from base to apex
                height_ratio = height_from_bottom / total_triangle_height
                current_max_width = (safe_side_length/2) * (1 - height_ratio)
            else:
                current_max_width = safe_side_length/2
            
            # Skip rows where width becomes too small
            if current_max_width < safe_side_length/2 * 0.05:
                continue
                
            # Calculate number of points for this row
            points_in_row = max(1, int(grid_x * (current_max_width / (safe_side_length/2))))
            
            print(f"Row {row}: y={y:.1f}, width={current_max_width*2:.1f}, points={points_in_row}")
            
            # Generate points across this row
            for col in range(points_in_row):
                if points_in_row == 1:
                    x = 0  # Single point at center
                else:
                    col_progress = col / (points_in_row - 1)  # 0 to 1
                    x = -current_max_width + col_progress * (2 * current_max_width)
                
                point = {
                    'x': x, 'y': y,
                    'covered': False,
                    'assigned_drone': -1,
                    'priority': 1.0,
                    'coverage_count': 0
                }
                
                if self.point_in_building(point):
                    self.global_grid_points.append(point)
                else:
                    print(f"  Warning: Point ({x:.1f}, {y:.1f}) outside triangle - skipped")
    
    def generate_circle_grid(self, bounds, safety_margin, grid_x, grid_y):
        """Generate grid points specifically for circular buildings"""
        radius = min(self.area_width, self.area_height) / 2
        safe_radius = radius - safety_margin
        
        # Use polar coordinate grid for better circle coverage
        num_rings = min(grid_y, 6)  # Limit to 6 rings for performance
        
        # Center point
        self.global_grid_points.append({
            'x': 0, 'y': 0,
            'covered': False, 'assigned_drone': -1,
            'priority': 1.0, 'coverage_count': 0
        })
        
        # Generate concentric rings
        for ring in range(1, num_rings):
            ring_radius = safe_radius * ring / (num_rings - 1)
            points_in_ring = max(6, int(grid_x * ring / num_rings * 2))  # More points in outer rings
            
            for i in range(points_in_ring):
                angle = 2 * math.pi * i / points_in_ring
                x = ring_radius * math.cos(angle)
                y = ring_radius * math.sin(angle)
                
                point = {
                    'x': x, 'y': y,
                    'covered': False, 'assigned_drone': -1,
                    'priority': 1.0, 'coverage_count': 0
                }
                
                if self.point_in_building(point):
                    self.global_grid_points.append(point)
    
    def generate_polygon_grid(self, bounds, safety_margin, grid_x, grid_y):
        """Generate grid points for regular polygons (hexagon, pentagon, octagon)"""
        radius = min(self.area_width, self.area_height) / 2
        safe_radius = radius - safety_margin
        
        # Similar to circle but with polygon-aware distribution
        num_rings = min(grid_y, 5)
        
        # Center point
        self.global_grid_points.append({
            'x': 0, 'y': 0,
            'covered': False, 'assigned_drone': -1,
            'priority': 1.0, 'coverage_count': 0
        })
        
        # Generate rings with polygon-aligned points
        for ring in range(1, num_rings):
            ring_radius = safe_radius * ring / (num_rings - 1)
            points_in_ring = max(6, int(grid_x * ring / num_rings))
            
            for i in range(points_in_ring):
                angle = 2 * math.pi * i / points_in_ring
                x = ring_radius * math.cos(angle)
                y = ring_radius * math.sin(angle)
                
                point = {
                    'x': x, 'y': y,
                    'covered': False, 'assigned_drone': -1,
                    'priority': 1.0, 'coverage_count': 0
                }
                
                if self.point_in_building(point):
                    self.global_grid_points.append(point)
    
    def generate_diamond_grid(self, bounds, safety_margin, grid_x, grid_y):
        """Generate grid points for diamond shape"""
        size = min(self.area_width, self.area_height) / 2
        safe_size = size - safety_margin
        
        # Generate points in diamond pattern
        for row in range(grid_y):
            y_progress = row / (grid_y - 1) if grid_y > 1 else 0.5
            y = -safe_size + y_progress * 2 * safe_size
            
            # Diamond constraint: |x| + |y| <= size
            max_x_at_y = safe_size - abs(y)
            if max_x_at_y <= 0:
                continue
                
            points_in_row = max(1, int(grid_x * max_x_at_y / safe_size))
            
            for col in range(points_in_row):
                if points_in_row == 1:
                    x = 0
                else:
                    x_progress = col / (points_in_row - 1)
                    x = -max_x_at_y + x_progress * 2 * max_x_at_y
                
                point = {
                    'x': x, 'y': y,
                    'covered': False, 'assigned_drone': -1,
                    'priority': 1.0, 'coverage_count': 0
                }
                
                if self.point_in_building(point):
                    self.global_grid_points.append(point)
    
        
    
    def generate_rectangular_grid(self, bounds, safety_margin, grid_x, grid_y):
        """Generate grid points for rectangular/square buildings (original method)"""
        safe_x_min = bounds['x_min'] + safety_margin
        safe_x_max = bounds['x_max'] - safety_margin
        safe_y_min = bounds['y_min'] + safety_margin
        safe_y_max = bounds['y_max'] - safety_margin
        
        # Ensure we have valid area after margins
        if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
            safety_margin = min(bounds['width'], bounds['height']) * 0.02
            safe_x_min = bounds['x_min'] + safety_margin
            safe_x_max = bounds['x_max'] - safety_margin
            safe_y_min = bounds['y_min'] + safety_margin
            safe_y_max = bounds['y_max'] - safety_margin
        
        safe_width = safe_x_max - safe_x_min
        safe_height = safe_y_max - safe_y_min
        
        spacing_x = safe_width / (grid_x - 1) if grid_x > 1 else 0
        spacing_y = safe_height / (grid_y - 1) if grid_y > 1 else 0
        
        for i in range(grid_y):
            for j in range(grid_x):
                if grid_x == 1:
                    x = (safe_x_min + safe_x_max) / 2
                else:
                    x = safe_x_min + j * spacing_x
                    
                if grid_y == 1:
                    y = (safe_y_min + safe_y_max) / 2
                else:
                    y = safe_y_min + i * spacing_y
                    
                point = {
                    'x': x, 'y': y,
                    'covered': False,
                    'assigned_drone': -1,
                    'priority': 1.0,
                    'coverage_count': 0
                }
                
                if self.point_in_building(point):
                    self.global_grid_points.append(point)
    
    def assign_grid_points_dynamically(self):
        """Dynamically assign grid points to drones to minimize overlap and energy consumption"""
        # Reset previous assignments
        for drone in self.swarm.drones:
            drone['assigned_points'] = set()
        
        for point in self.global_grid_points:
            point['assigned_drone'] = -1
            point['coverage_count'] = 0
        
        # For single point formation, use special balanced assignment
        if self.start_formation == 'single_point':
            self.assign_points_single_point_balanced()
        elif self.strategy == 'dynamic_coverage':
            self.assign_points_dynamic_coverage()
        elif self.strategy == 'competitive_assignment':
            self.assign_points_competitive()
        else:  # cooperative_planning
            self.assign_points_cooperative()
        
        # Verify assignment
        total_assigned = sum(len(drone['assigned_points']) for drone in self.swarm.drones)
        print(f"Dynamically assigned {total_assigned} grid points to {self.n_drones} drones")
        
        # Show assignment distribution
        for drone in self.swarm.drones:
            assigned_points = [self.global_grid_points[i] for i in drone['assigned_points']]
            if assigned_points:
                x_coords = [p['x'] for p in assigned_points]
                y_coords = [p['y'] for p in assigned_points]
                print(f"Drone {drone['id']}: {len(assigned_points)} points "
                      f"X[{min(x_coords):.1f}, {max(x_coords):.1f}] "
                      f"Y[{min(y_coords):.1f}, {max(y_coords):.1f}]")
    
    def assign_points_single_point_balanced(self):
        """Dynamic assignment for single point formation using clustering and optimization"""
        if len(self.global_grid_points) == 0:
            print("Warning: No grid points to assign!")
            return
            
        print(f"Dynamic clustering assignment: {len(self.global_grid_points)} points for {self.n_drones} drones")
        
        # Use K-means style clustering to find optimal point groupings
        optimal_assignment = self.find_optimal_point_clusters()
        
        # Assign based on optimal clusters
        for point_idx, drone_id in optimal_assignment.items():
            self.swarm.drones[drone_id]['assigned_points'].add(point_idx)
            self.global_grid_points[point_idx]['assigned_drone'] = drone_id
        
        # Verify and improve assignment to minimize overlap
        self.minimize_assignment_overlap()
        
        assignment_count = [len(drone['assigned_points']) for drone in self.swarm.drones]
        print("Dynamic clustering assignment results:")
        for i, count in enumerate(assignment_count):
            print(f"  Drone {i}: {count} points")
        
        print("Used dynamic clustering assignment with overlap minimization")
    
    def find_optimal_point_clusters(self):
        """Find optimal clusters of points for each drone using modified K-means"""
        import random
        
        # Initialize cluster centers (starting positions for each drone)
        cluster_centers = []
        for drone in self.swarm.drones:
            cluster_centers.append([drone['start_position']['x'], drone['start_position']['y']])
        
        assignments = {}
        
        # Iterative optimization to find best clusters
        for iteration in range(10):  # Max 10 iterations
            old_assignments = assignments.copy()
            assignments = {}
            
            # Assign each point to the best drone considering multiple factors
            for point_idx, point in enumerate(self.global_grid_points):
                best_drone = 0
                best_score = float('-inf')
                
                for drone_id, center in enumerate(cluster_centers):
                    # Calculate multiple factors for assignment
                    distance = math.sqrt((point['x'] - center[0])**2 + (point['y'] - center[1])**2)
                    
                    # Current workload of this drone
                    current_workload = sum(1 for p_idx, d_id in assignments.items() if d_id == drone_id)
                    
                    # Coverage potential from this point
                    coverage_potential = self.calculate_coverage_potential_at_point(point, point_idx)
                    
                    # Calculate assignment score with multiple objectives
                    workload_penalty = current_workload * 2  # Penalize overloaded drones
                    distance_penalty = distance * self.distance_weight
                    coverage_bonus = coverage_potential * self.coverage_weight
                    
                    score = coverage_bonus - distance_penalty - workload_penalty
                    
                    if score > best_score:
                        best_score = score
                        best_drone = drone_id
                
                assignments[point_idx] = best_drone
            
            # Update cluster centers to center of assigned points
            for drone_id in range(self.n_drones):
                assigned_points = [self.global_grid_points[p_idx] for p_idx, d_id in assignments.items() if d_id == drone_id]
                
                if assigned_points:
                    # Move center to centroid of assigned points
                    center_x = sum(p['x'] for p in assigned_points) / len(assigned_points)
                    center_y = sum(p['y'] for p in assigned_points) / len(assigned_points)
                    cluster_centers[drone_id] = [center_x, center_y]
            
            # Check for convergence
            if assignments == old_assignments:
                print(f"Clustering converged after {iteration + 1} iterations")
                break
        
        return assignments
    
    def calculate_coverage_potential_at_point(self, point, point_idx):
        """Calculate how many nearby points can be covered from this position"""
        footprint_vertices = self.calculate_camera_footprint(point)
        coverage_count = 0
        
        for other_idx, other_point in enumerate(self.global_grid_points):
            if other_idx != point_idx:
                if self.point_in_polygon((other_point['x'], other_point['y']), footprint_vertices):
                    coverage_count += 1
        
        return coverage_count + 1  # +1 for the point itself
    
    def minimize_assignment_overlap(self):
        """Post-process assignment to minimize overlap between drones"""
        improved = True
        iterations = 0
        max_iterations = 5
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Find overlapping coverage areas
            overlaps = self.find_coverage_overlaps()
            
            if not overlaps:
                break
            
            # Try to reassign points to reduce overlap
            for (drone1_id, drone2_id, overlap_points) in overlaps:
                if len(overlap_points) == 0:
                    continue
                
                # Try reassigning some overlapping points
                for point_idx in overlap_points[:len(overlap_points)//2]:  # Reassign half
                    current_drone = self.global_grid_points[point_idx]['assigned_drone']
                    
                    # Find the drone with less workload
                    workload1 = len(self.swarm.drones[drone1_id]['assigned_points'])
                    workload2 = len(self.swarm.drones[drone2_id]['assigned_points'])
                    
                    if workload1 > workload2:
                        # Reassign from drone1 to drone2
                        if current_drone == drone1_id:
                            self.swarm.drones[drone1_id]['assigned_points'].remove(point_idx)
                            self.swarm.drones[drone2_id]['assigned_points'].add(point_idx)
                            self.global_grid_points[point_idx]['assigned_drone'] = drone2_id
                            improved = True
                    else:
                        # Reassign from drone2 to drone1
                        if current_drone == drone2_id:
                            self.swarm.drones[drone2_id]['assigned_points'].remove(point_idx)
                            self.swarm.drones[drone1_id]['assigned_points'].add(point_idx)
                            self.global_grid_points[point_idx]['assigned_drone'] = drone1_id
                            improved = True
        
        print(f"Overlap minimization completed in {iterations} iterations")
    
    def find_coverage_overlaps(self):
        """Find points that could be covered by multiple drones' camera footprints"""
        overlaps = []
        
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                drone1 = self.swarm.drones[i]
                drone2 = self.swarm.drones[j]
                
                # Get points assigned to each drone
                drone1_points = [self.global_grid_points[idx] for idx in drone1['assigned_points']]
                drone2_points = [self.global_grid_points[idx] for idx in drone2['assigned_points']]
                
                # Check for potential overlap in coverage areas
                overlap_points = []
                
                for point1 in drone1_points:
                    footprint1 = self.calculate_camera_footprint(point1)
                    
                    for idx in drone2['assigned_points']:
                        point2 = self.global_grid_points[idx]
                        if self.point_in_polygon((point2['x'], point2['y']), footprint1):
                            overlap_points.append(idx)
                
                if overlap_points:
                    overlaps.append((i, j, overlap_points))
        
        return overlaps
    
    def assign_points_dynamic_coverage(self):
        """Assign points using dynamic coverage strategy with workload balancing"""
        unassigned_points = set(range(len(self.global_grid_points)))
        
        while unassigned_points:
            # For each drone, find the best next point to assign
            best_assignments = []
            
            for drone in self.swarm.drones:
                if not unassigned_points:
                    break
                    
                best_point_idx = None
                best_score = float('-inf')
                
                current_pos = drone['position']
                
                # Calculate workload balance factor
                current_workload = len(drone['assigned_points'])
                average_workload = sum(len(d['assigned_points']) for d in self.swarm.drones) / len(self.swarm.drones)
                workload_factor = max(0.1, 2.0 - (current_workload / max(average_workload, 1)))
                
                for point_idx in unassigned_points:
                    point = self.global_grid_points[point_idx]
                    
                    # Calculate assignment score
                    distance = self.distance(current_pos, point)
                    coverage_potential = self.calculate_coverage_potential(point, drone)
                    overlap_penalty = self.calculate_overlap_penalty(point, drone)
                    
                    # Multi-objective score with workload balancing
                    score = (
                        self.coverage_weight * coverage_potential -
                        self.distance_weight * distance -
                        self.overlap_penalty * overlap_penalty +
                        workload_factor * 10  # Bonus for underloaded drones
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_point_idx = point_idx
                
                if best_point_idx is not None:
                    best_assignments.append((drone['id'], best_point_idx, best_score))
            
            if not best_assignments:
                break
            
            # Sort by score and assign the best one
            best_assignments.sort(key=lambda x: x[2], reverse=True)
            drone_id, point_idx, score = best_assignments[0]
            
            # Make assignment
            self.swarm.drones[drone_id]['assigned_points'].add(point_idx)
            self.global_grid_points[point_idx]['assigned_drone'] = drone_id
            unassigned_points.remove(point_idx)
            
            # Update drone position to this point for next iteration
            self.swarm.drones[drone_id]['position'] = {
                'x': self.global_grid_points[point_idx]['x'],
                'y': self.global_grid_points[point_idx]['y']
            }
    
    def assign_points_competitive(self):
        """Assign points using competitive assignment (auction-based)"""
        unassigned_points = list(range(len(self.global_grid_points)))
        
        for point_idx in unassigned_points:
            point = self.global_grid_points[point_idx]
            
            # Each drone bids for this point
            bids = []
            for drone in self.swarm.drones:
                distance = self.distance(drone['position'], point)
                coverage_potential = self.calculate_coverage_potential(point, drone)
                overlap_cost = self.calculate_overlap_penalty(point, drone)
                
                # Bid = benefit - cost
                bid = coverage_potential - self.distance_weight * distance - overlap_cost
                bids.append((drone['id'], bid))
            
            # Assign to highest bidder
            if bids:
                winner_id = max(bids, key=lambda x: x[1])[0]
                self.swarm.drones[winner_id]['assigned_points'].add(point_idx)
                point['assigned_drone'] = winner_id
    
    def assign_points_cooperative(self):
        """Assign points using cooperative planning"""
        # Calculate coverage matrix (which drones can cover which points efficiently)
        coverage_matrix = np.zeros((self.n_drones, len(self.global_grid_points)))
        
        for i, drone in enumerate(self.swarm.drones):
            for j, point in enumerate(self.global_grid_points):
                distance = self.distance(drone['position'], point)
                coverage_potential = self.calculate_coverage_potential(point, drone)
                
                # Efficiency score
                if distance > 0:
                    efficiency = coverage_potential / distance
                else:
                    efficiency = coverage_potential
                
                coverage_matrix[i, j] = efficiency
        
        # Use greedy assignment to minimize total cost while ensuring coverage
        unassigned_points = set(range(len(self.global_grid_points)))
        
        # Assign points iteratively to balance workload
        while unassigned_points:
            # Find drone with least assigned points
            workloads = [len(drone['assigned_points']) for drone in self.swarm.drones]
            min_workload_drone = workloads.index(min(workloads))
            
            # Find best point for this drone
            best_point = None
            best_efficiency = -1
            
            for point_idx in unassigned_points:
                efficiency = coverage_matrix[min_workload_drone, point_idx]
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_point = point_idx
            
            if best_point is not None:
                self.swarm.drones[min_workload_drone]['assigned_points'].add(best_point)
                self.global_grid_points[best_point]['assigned_drone'] = min_workload_drone
                unassigned_points.remove(best_point)
            else:
                break
    
    def calculate_coverage_potential(self, point, drone):
        """Calculate how much coverage potential this point has for the drone"""
        # Check how many other grid points this position could cover
        footprint_vertices = self.calculate_camera_footprint(point)
        coverage_count = 0
        
        for other_point in self.global_grid_points:
            if other_point != point and not other_point['covered']:
                if self.point_in_polygon((other_point['x'], other_point['y']), footprint_vertices):
                    coverage_count += 1
        
        return coverage_count + 1  # +1 for the point itself
    
    def calculate_overlap_penalty(self, point, drone):
        """Calculate penalty for potential overlap with other drones"""
        penalty = 0
        footprint_vertices = self.calculate_camera_footprint(point)
        
        for other_drone in self.swarm.drones:
            if other_drone['id'] == drone['id']:
                continue
                
            # Check if other drone's assigned points would be covered by this position
            for other_point_idx in other_drone['assigned_points']:
                other_point = self.global_grid_points[other_point_idx]
                if self.point_in_polygon((other_point['x'], other_point['y']), footprint_vertices):
                    penalty += 1
        
        return penalty
    
    def reassign_grid_points(self):
        """Reassign grid points dynamically based on current situation"""
        if not self.swarm:
            return
        
        print("Reassigning grid points dynamically...")
        self.assign_grid_points_dynamically()
        
        # Regenerate paths with new assignments
        self.generate_swarm_paths()
        self.update_swarm_info()
        self.draw()
        
        print("Dynamic reassignment complete!")
    
    def generate_swarm_paths(self):
        """Generate paths for all drones based on their dynamically assigned points"""
        for drone in self.swarm.drones:
            assigned_points = [self.global_grid_points[i] for i in drone['assigned_points']]
            
            if assigned_points:
                drone['waypoints'] = self.generate_optimal_path(drone, assigned_points)
                if drone['waypoints']:
                    drone['position'] = drone['start_position'].copy()
                    drone['target_position'] = drone['waypoints'][0].copy()
                else:
                    drone['waypoints'] = [drone['start_position']]
                    drone['position'] = drone['start_position'].copy()
                    drone['target_position'] = drone['position'].copy()
            else:
                # No assigned points - drone stays at start position
                drone['waypoints'] = [drone['start_position']]
                drone['position'] = drone['start_position'].copy()
                drone['target_position'] = drone['position'].copy()
                drone['is_active'] = False
    
    def generate_optimal_path(self, drone, assigned_points):
        """Generate optimal path through assigned points using advanced strategies"""
        if not assigned_points:
            return [drone['start_position']]

        self.enhanced_optimizer.camera_fov = self.camera_fov
        self.enhanced_optimizer.altitude = self.altitude
        self.enhanced_optimizer.footprint_radius = self.altitude * math.tan(math.radians(self.camera_fov / 2))

        
        if self.path_strategy == 'enhanced_coverage':

            waypoints = self.enhanced_optimizer.generate_coverage_optimal_waypoints(
                drone['start_position'], assigned_points
            )
            return waypoints
        
        elif self.path_strategy == 'enhanced_hybrid':
            waypoints = self.enhanced_optimizer.generate_coverage_optimal_waypoints(
                drone['start_position'], assigned_points
            )

            waypoints = self.enhanced_optimizer.optimize_waypoint_order_tsp(waypoints)
            return waypoints
        
        elif self.path_strategy == 'genetic_algorithm':
            # Use Genetic Algorithm for path optimization
            print(f"Using Genetic Algorithm for Drone {drone['id']} path optimization")
            waypoints = self.ga_optimizer.generate_path_with_ga(
                drone, assigned_points, self.camera_fov, self.altitude
            )
            return waypoints
        
        else:

            # Use the selected path optimization strategy with enhanced algorithms
            if self.path_strategy == 'smart_pov':
                return self.generate_advanced_smart_pov_path(drone, assigned_points)
            elif self.path_strategy == 'coverage':
                return self.generate_advanced_coverage_path(drone, assigned_points)
            elif self.path_strategy == 'hybrid':
                return self.generate_advanced_hybrid_path(drone, assigned_points)
            elif self.path_strategy == 'nearest':
                return self.generate_nearest_neighbor_path(drone, assigned_points)
            elif self.path_strategy == 'tsp_optimized':
                return self.generate_advanced_tsp_path(drone, assigned_points)
            else:
                # Default to advanced hybrid
                return self.generate_advanced_hybrid_path(drone, assigned_points)
    
    def generate_advanced_smart_pov_path(self, drone, assigned_points):
        """Advanced smart POV path with multi-point coverage optimization"""
        waypoints = [drone['start_position']]
        remaining_points = assigned_points.copy()
        current_pos = drone['start_position']
        covered_point_indices = set()
        
        # Initial coverage check from start position
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_point_indices.update(initial_covered)
        
        iteration = 0
        max_iterations = len(assigned_points) * 2
        
        while len(covered_point_indices) < len(assigned_points) and iteration < max_iterations:
            iteration += 1
            best_waypoint = None
            best_score = -1
            
            # Find uncovered points
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_point_indices]
            
            if not uncovered_points:
                break
            
            # Evaluate potential waypoints with advanced scoring
            for point in uncovered_points:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                # Calculate comprehensive coverage metrics
                would_cover_indices = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage_count = len(set(would_cover_indices) - covered_point_indices)
                
                if new_coverage_count > 0:
                    distance = self.distance(current_pos, candidate_pos)
                    
                    # Advanced scoring with multiple factors
                    coverage_efficiency = new_coverage_count / max(distance, 1)
                    
                    # Bonus for covering many points at once
                    multi_coverage_bonus = new_coverage_count * 2 if new_coverage_count > 3 else 0
                    
                    # Penalty for very long distances
                    distance_penalty = max(0, (distance - 30) * 0.5)
                    
                    # Bonus for points near cluster centers
                    cluster_bonus = self.calculate_cluster_centrality(candidate_pos, uncovered_points)
                    
                    score = (coverage_efficiency * 20 + 
                           multi_coverage_bonus + 
                           cluster_bonus * 5 - 
                           distance_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best_waypoint = candidate_pos
            
            # If no good waypoint found, go to nearest uncovered point
            if best_waypoint is None and uncovered_points:
                best_waypoint = min(uncovered_points, key=lambda p: self.distance(current_pos, p))
                best_waypoint = {'x': best_waypoint['x'], 'y': best_waypoint['y']}
            
            if best_waypoint is None:
                break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            
            # Update covered points
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_point_indices.update(newly_covered)
        
        return waypoints
    
    def generate_advanced_coverage_path(self, drone, assigned_points):
        """Advanced coverage path with greedy set cover optimization"""
        waypoints = [drone['start_position']]
        current_pos = drone['start_position']
        covered_point_indices = set()
        
        # Initial coverage
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_point_indices.update(initial_covered)
        
        while len(covered_point_indices) < len(assigned_points):
            best_waypoint = None
            max_new_coverage = 0
            best_efficiency = 0
            
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_point_indices]
            
            # Use greedy set cover approach
            for point in uncovered_points:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                would_cover = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage = len(set(would_cover) - covered_point_indices)
                distance = self.distance(current_pos, candidate_pos)
                
                # Efficiency: coverage per unit distance
                efficiency = new_coverage / max(distance, 1)
                
                if (new_coverage > max_new_coverage or 
                    (new_coverage == max_new_coverage and efficiency > best_efficiency)):
                    max_new_coverage = new_coverage
                    best_efficiency = efficiency
                    best_waypoint = candidate_pos
            
            if best_waypoint is None:
                break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_point_indices.update(newly_covered)
        
        return waypoints
    
    def generate_advanced_hybrid_path(self, drone, assigned_points):
        """Advanced hybrid path with adaptive strategy selection"""
        waypoints = [drone['start_position']]
        current_pos = drone['start_position']
        covered_point_indices = set()
        
        # Initial coverage
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_point_indices.update(initial_covered)
        
        while len(covered_point_indices) < len(assigned_points):
            best_waypoint = None
            best_score = float('-inf')
            
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_point_indices]
            
            if not uncovered_points:
                break
            
            # Adaptive candidate selection based on remaining points
            if len(uncovered_points) > 15:
                # Many points remaining: use aggressive coverage strategy
                candidates = uncovered_points[:20]  # Limit for performance
            else:
                # Few points remaining: consider all
                candidates = uncovered_points
            
            for point in candidates:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                would_cover = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage = len(set(would_cover) - covered_point_indices)
                distance = self.distance(current_pos, candidate_pos)
                
                # Adaptive hybrid scoring
                coverage_score = new_coverage * 12
                
                # Distance penalty with adaptive weight
                remaining_ratio = len(uncovered_points) / len(assigned_points)
                distance_weight = 0.3 + (0.5 * (1 - remaining_ratio))  # Increase distance concern as mission progresses
                distance_penalty = distance * distance_weight
                
                # Efficiency bonus
                efficiency_bonus = (new_coverage / max(distance, 1)) * 8
                
                # Clustering bonus for better spatial distribution
                cluster_bonus = self.calculate_cluster_centrality(candidate_pos, uncovered_points) * 3
                
                # Penalty for very long jumps
                jump_penalty = max(0, (distance - 40) * 0.8)
                
                total_score = (coverage_score + efficiency_bonus + cluster_bonus - 
                             distance_penalty - jump_penalty)
                
                if total_score > best_score:
                    best_score = total_score
                    best_waypoint = candidate_pos
            
            if best_waypoint is None:
                break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_point_indices.update(newly_covered)
        
        return waypoints
    
    def generate_advanced_tsp_path(self, drone, assigned_points):
        """Advanced TSP with coverage-aware optimization"""
        if len(assigned_points) <= 1:
            waypoints = [drone['start_position']]
            if assigned_points:
                waypoints.append({'x': assigned_points[0]['x'], 'y': assigned_points[0]['y']})
            return waypoints
        
        # Use coverage-aware waypoint selection instead of visiting all points
        essential_points = self.select_essential_waypoints(drone, assigned_points)
        
        if len(essential_points) <= 1:
            waypoints = [drone['start_position']]
            if essential_points:
                waypoints.append({'x': essential_points[0]['x'], 'y': essential_points[0]['y']})
            return waypoints
        
        # Apply TSP optimization to essential points
        waypoints = [drone['start_position']]
        unvisited = essential_points.copy()
        current_pos = drone['start_position']
        
        # Greedy TSP with 2-opt improvement
        while unvisited:
            nearest_point = min(unvisited, key=lambda p: self.distance(current_pos, p))
            waypoints.append({'x': nearest_point['x'], 'y': nearest_point['y']})
            current_pos = nearest_point
            unvisited.remove(nearest_point)
        
        # Apply 2-opt improvement
        if len(waypoints) > 4:
            waypoints = self.apply_2opt_improvement(waypoints)
        
        return waypoints
    
    def select_essential_waypoints(self, drone, assigned_points):
        """Select minimum essential waypoints that can cover all assigned points"""
        if not assigned_points:
            return []
        
        uncovered_indices = set(range(len(assigned_points)))
        essential_points = []
        
        while uncovered_indices:
            best_point = None
            max_coverage = 0
            
            for point in assigned_points:
                coverage_indices = self.get_points_in_camera_footprint(point, assigned_points)
                new_coverage = len(set(coverage_indices) & uncovered_indices)
                
                if new_coverage > max_coverage:
                    max_coverage = new_coverage
                    best_point = point
            
            if best_point is None:
                break
            
            essential_points.append(best_point)
            covered_indices = self.get_points_in_camera_footprint(best_point, assigned_points)
            uncovered_indices -= set(covered_indices)
        
        return essential_points
    
    def calculate_cluster_centrality(self, candidate_pos, point_list):
        """Calculate how central a position is to nearby clusters of points"""
        if not point_list:
            return 0
        
        # Find nearby points within reasonable distance
        nearby_points = [p for p in point_list 
                        if self.distance(candidate_pos, p) <= 25]
        
        if not nearby_points:
            return 0
        
        # Calculate centrality score
        total_distance = sum(self.distance(candidate_pos, p) for p in nearby_points)
        avg_distance = total_distance / len(nearby_points)
        
        # Lower average distance = higher centrality
        centrality = max(0, 25 - avg_distance) / 25
        
        # Bonus for being near many points
        density_bonus = min(len(nearby_points) / 5, 2)
        
        return centrality + density_bonus
    
    def generate_smart_pov_path(self, drone, assigned_points):
        """Generate smart POV optimized path focusing on camera coverage efficiency"""
        waypoints = [drone['start_position']]
        remaining_points = assigned_points.copy()
        current_pos = drone['start_position']
        covered_indices = set()
        
        # Mark initially covered points from starting position
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_indices.update(initial_covered)
        
        while len(covered_indices) < len(assigned_points):
            best_waypoint = None
            best_score = -1
            
            # Find uncovered points
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_indices]
            
            if not uncovered_points:
                break
            
            # Evaluate potential waypoints based on coverage efficiency
            for point in uncovered_points:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                # Calculate how many new points this position would cover
                would_cover = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage = len(set(would_cover) - covered_indices)
                
                if new_coverage > 0:
                    distance = self.distance(current_pos, candidate_pos)
                    # Score: prioritize new coverage, minimize distance
                    score = new_coverage * 15 - distance * 0.1
                    
                    # Bonus for covering multiple points at once
                    if new_coverage > 3:
                        score += new_coverage * 5
                    
                    if score > best_score:
                        best_score = score
                        best_waypoint = candidate_pos
            
            if best_waypoint is None:
                # If no good waypoint found, go to nearest uncovered point
                if uncovered_points:
                    best_waypoint = min(uncovered_points, 
                                      key=lambda p: self.distance(current_pos, p))
                    best_waypoint = {'x': best_waypoint['x'], 'y': best_waypoint['y']}
                else:
                    break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            
            # Update covered points
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_indices.update(newly_covered)
        
        return waypoints
    
    def generate_coverage_optimized_path(self, drone, assigned_points):
        """Generate path optimized for maximum coverage with minimal waypoints"""
        waypoints = [drone['start_position']]
        remaining_points = assigned_points.copy()
        current_pos = drone['start_position']
        covered_indices = set()
        
        # Initial coverage check
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_indices.update(initial_covered)
        
        while len(covered_indices) < len(assigned_points):
            best_waypoint = None
            max_new_coverage = 0
            best_distance = float('inf')
            
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_indices]
            
            for point in uncovered_points:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                would_cover = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage = len(set(would_cover) - covered_indices)
                distance = self.distance(current_pos, candidate_pos)
                
                # Prioritize maximum new coverage, use distance as tiebreaker
                if (new_coverage > max_new_coverage or 
                    (new_coverage == max_new_coverage and distance < best_distance)):
                    max_new_coverage = new_coverage
                    best_distance = distance
                    best_waypoint = candidate_pos
            
            if best_waypoint is None:
                break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_indices.update(newly_covered)
        
        return waypoints
    
    def generate_hybrid_path(self, drone, assigned_points):
        """Generate hybrid path balancing coverage efficiency and travel distance"""
        waypoints = [drone['start_position']]
        remaining_points = assigned_points.copy()
        current_pos = drone['start_position']
        covered_indices = set()
        
        # Initial coverage
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_indices.update(initial_covered)
        
        while len(covered_indices) < len(assigned_points):
            best_waypoint = None
            best_score = float('-inf')
            
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_indices]
            
            # Limit candidates for efficiency (consider top candidates)
            candidates = uncovered_points[:min(20, len(uncovered_points))]
            
            for point in candidates:
                candidate_pos = {'x': point['x'], 'y': point['y']}
                
                would_cover = self.get_points_in_camera_footprint(candidate_pos, assigned_points)
                new_coverage = len(set(would_cover) - covered_indices)
                distance = self.distance(current_pos, candidate_pos)
                
                # Hybrid scoring: balance coverage and distance
                coverage_score = new_coverage * 10
                distance_penalty = distance * 0.2
                
                # Bonus for efficient positions (high coverage per distance)
                if distance > 0:
                    efficiency_bonus = (new_coverage / distance) * 5
                else:
                    efficiency_bonus = new_coverage * 5
                
                # Penalty for very long distances
                if distance > 30:
                    distance_penalty += (distance - 30) * 0.5
                
                total_score = coverage_score + efficiency_bonus - distance_penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_waypoint = candidate_pos
            
            if best_waypoint is None:
                break
            
            waypoints.append(best_waypoint)
            current_pos = best_waypoint
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_indices.update(newly_covered)
        
        return waypoints
    
    def generate_nearest_neighbor_path(self, drone, assigned_points):
        """Generate path using nearest neighbor algorithm"""
        waypoints = [drone['start_position']]
        remaining_points = assigned_points.copy()
        current_pos = drone['start_position']
        covered_indices = set()
        
        # Initial coverage
        initial_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
        covered_indices.update(initial_covered)
        
        while len(covered_indices) < len(assigned_points):
            uncovered_points = [p for i, p in enumerate(assigned_points) if i not in covered_indices]
            
            if not uncovered_points:
                break
            
            # Find nearest uncovered point
            nearest_point = min(uncovered_points, key=lambda p: self.distance(current_pos, p))
            nearest_pos = {'x': nearest_point['x'], 'y': nearest_point['y']}
            
            waypoints.append(nearest_pos)
            current_pos = nearest_pos
            
            # Update coverage
            newly_covered = self.get_points_in_camera_footprint(current_pos, assigned_points)
            covered_indices.update(newly_covered)
        
        return waypoints
    
    def generate_tsp_optimized_path(self, drone, assigned_points):
        """Generate TSP-optimized path through all assigned points"""
        if len(assigned_points) <= 1:
            waypoints = [drone['start_position']]
            if assigned_points:
                waypoints.append({'x': assigned_points[0]['x'], 'y': assigned_points[0]['y']})
            return waypoints
        
        # Use a greedy TSP approximation for efficiency
        waypoints = [drone['start_position']]
        unvisited = assigned_points.copy()
        current_pos = drone['start_position']
        
        while unvisited:
            # Find shortest distance to next point
            nearest_point = min(unvisited, key=lambda p: self.distance(current_pos, p))
            waypoints.append({'x': nearest_point['x'], 'y': nearest_point['y']})
            current_pos = nearest_point
            unvisited.remove(nearest_point)
        
        # Optional: Apply 2-opt improvement for better TSP solution
        if len(waypoints) > 4:
            waypoints = self.apply_2opt_improvement(waypoints)
        
        return waypoints
    
    def apply_2opt_improvement(self, waypoints):
        """Apply 2-opt improvement to TSP path"""
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(waypoints) - 2):
                for j in range(i + 1, len(waypoints)):
                    if j == len(waypoints) - 1:
                        continue
                    
                    # Calculate current distance
                    current_dist = (self.distance(waypoints[i-1], waypoints[i]) +
                                  self.distance(waypoints[j], waypoints[j+1]))
                    
                    # Calculate distance after swap
                    new_dist = (self.distance(waypoints[i-1], waypoints[j]) +
                               self.distance(waypoints[i], waypoints[j+1]))
                    
                    if new_dist < current_dist:
                        # Perform 2-opt swap
                        waypoints[i:j+1] = reversed(waypoints[i:j+1])
                        improved = True
        
        return waypoints

    
    def get_points_in_camera_footprint(self, position, points_list):
        """Get indices of points covered by camera footprint at given position"""
        footprint_vertices = self.calculate_camera_footprint(position)
        covered_indices = []
        
        for i, point in enumerate(points_list):
            if self.point_in_polygon((point['x'], point['y']), footprint_vertices):
                covered_indices.append(i)
        
        return covered_indices
    
    def get_building_bounds(self):
        """Get the bounding box of the building shape"""
        if self.area_shape == 'circle':
            radius = min(self.area_width, self.area_height) / 2
            return {
                'x_min': -radius,
                'x_max': radius,
                'y_min': -radius,
                'y_max': radius,
                'width': radius * 2,
                'height': radius * 2
            }
        elif self.area_shape =='triangle':
            side_length = self.area_width
            height = side_length * math.sqrt(3)/2
            return {
                'x_min': -side_length/2,
                'x_max': side_length/2,
                'y_min': -height/2,
                'y_max': height,
                'width': side_length,
                'height': height * 1.5
            }
        else:  # rectangle or square
            return {
                'x_min': -self.area_width/2,
                'x_max': self.area_width/2,
                'y_min': -self.area_height/2,
                'y_max': self.area_height/2,
                'width': self.area_width,
                'height': self.area_height
            }
        

    def copy_waypoints_to_clipboard(self):
        """Copy all drone waypoints to clipboard in the specified format"""
        if not self.swarm or not self.swarm.drones:
            self.show_message("No waypoints available", "Please generate swarm paths first!")
            return
        
        waypoint_text = self.generate_waypoint_export_text()
        
        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(waypoint_text)
        self.root.update()  # Required for clipboard to work
        
        # Show success message
        self.show_message("Waypoints Copied!", 
                         f"Waypoints for {len(self.swarm.drones)} drones copied to clipboard.\n"
                         f"Total waypoints: {sum(len(drone['waypoints']) for drone in self.swarm.drones)}")
        
        # Also save to file option
        self.save_waypoints_to_file(waypoint_text)
    
    def generate_waypoint_export_text(self):
        """Generate waypoint text in NumPy array format"""
        export_lines = []
        
        # Add header with configuration info
        export_lines.append("# Drone Swarm Waypoints Export")
        #export_lines.append(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        export_lines.append(f"# Configuration:")
        export_lines.append(f"# - Number of Drones: {self.n_drones}")
        export_lines.append(f"# - Area Size: {self.area_width}m x {self.area_height}m")
        export_lines.append(f"# - Grid Size: {self.grid_points_x} x {self.grid_points_y}")
        export_lines.append(f"# - Strategy: {self.strategy}")
        export_lines.append(f"# - Camera FOV: {self.camera_fov}°")
        export_lines.append(f"# - Altitude: {self.altitude}m")
        export_lines.append(f"# - Speed: {self.drone_speed}m/s")
        #export_lines.append("")
        #export_lines.append("b3Printf: crack_patch")
        
        # Export waypoints for each drone in NumPy array format
        for drone in self.swarm.drones:
            export_lines.append(f"# Drone {drone['id']} - {len(drone['waypoints'])} waypoints")
            #export_lines.append(f"# Region: X[{drone['region']['x_min']:.1f}, {drone['region']['x_max']:.1f}] Y[{drone['region']['y_min']:.1f}, {drone['region']['y_max']:.1f}]")
            export_lines.append(f"# Assigned Grid Points: {len(drone['assigned_points'])}")
            
            if len(drone['waypoints']) > 0:
                # Create NumPy array format
                array_lines = [f"Waypoint{drone['id']+1} = [array(["]
                
                # Format each waypoint as [x, y, z] where z = 1 (as in your example)
                waypoint_rows = []
                for wp in drone['waypoints']:
                    # Format numbers to match your example (remove trailing zeros, proper spacing)
                    x_val = wp['x']
                    y_val = wp['y']
                    z_val = self.altitude  # Set Z to 1 as in your example
                    
                    # Format as in your example: proper spacing and decimal handling
                    if x_val == int(x_val):
                        x_str = f"{int(x_val):>6}"
                    else:
                        x_str = f"{x_val:>6.1f}"
                    
                    if y_val == int(y_val):
                        y_str = f"{int(y_val):>6}"
                    else:
                        y_str = f"{y_val:>6.1f}"
                    
                    z_str = f"{int(z_val):>6}"
                    
                    waypoint_rows.append(f"       [{x_str} , {y_str} , {z_str} ]")
                
                # Join all waypoint rows
                array_content = ",\n".join(waypoint_rows)
                array_lines.append(array_content)
                array_lines.append("])]")
                
                # Combine into single string
                export_lines.extend(array_lines)
            else:
                export_lines.append(f"Waypoint{drone['id']+1} = [array([[  0. ,   0. ,   1. ]])]")
            
            export_lines.append("")
        
        # Add summary statistics
        export_lines.append("# Summary Statistics:")
        total_waypoints = sum(len(drone['waypoints']) for drone in self.swarm.drones)
        total_distance = 0
        
        for drone in self.swarm.drones:
            drone_distance = 0
            for i in range(1, len(drone['waypoints'])):
                drone_distance += self.distance(drone['waypoints'][i-1], drone['waypoints'][i])
            total_distance += drone_distance
            export_lines.append(f"# Drone {drone['id']}: {len(drone['waypoints'])} waypoints, {drone_distance:.2f}m distance")
        
        export_lines.append(f"# Total Waypoints: {total_waypoints}")
        export_lines.append(f"# Total Distance: {total_distance:.2f}m")
        export_lines.append(f"# Estimated Mission Time: {total_distance/self.drone_speed:.1f}s (parallel execution)")
        
        return "\n".join(export_lines)
    
    def save_waypoints_to_file(self, waypoint_text):
        """Save waypoints to a text file"""
        try:
            import tkinter.filedialog as fd
            
            # Ask user if they want to save to file
            save_file = fd.asksaveasfilename(
                title="Save Waypoints to File",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialname=f"drone_waypoints_{self.n_drones}drones_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if save_file:
                with open(save_file, 'w') as f:
                    f.write(waypoint_text)
                print(f"Waypoints saved to: {save_file}")
        except Exception as e:
            print(f"Could not save file: {e}")

    def run_pybullet_simulation(self):
        """Launch PyBullet simulation with current waypoints"""
        if not self.swarm or not self.swarm.drones:
            self.show_message("No Waypoints Available", "Please generate swarm paths first!")
            return
        
        try:
            # Show confirmation dialog
            import tkinter.messagebox as mb
            result = mb.askyesno("Launch PyBullet Simulation", 
                            f"Launch PyBullet simulation with {len(self.swarm.drones)} drones?\n"
                            f"Total waypoints: {sum(len(drone['waypoints']) for drone in self.swarm.drones)}")
            
            if result:
                print(f"🚁 Launching PyBullet simulation...")
                self.launch_simulation_direct()
            
        except Exception as e:
            self.show_message("Error", f"Failed to launch simulation: {str(e)}")

    def prepare_waypoint_config(self):
        """Prepare waypoint configuration for PyBullet"""
        import numpy as np
        
        waypoint_config = {
            'num_drones': self.n_drones,
            'building_width': self.area_width,
            'building_height': self.area_height,
            'altitude': self.altitude,
            'building_shape': self.area_shape,
            'starting_formation': self.start_formation,
            'waypoints': [],
            'init_positions': []
        }
        
        # Convert waypoints to numpy arrays
        for drone in self.swarm.drones:
            drone_waypoints = []
            for wp in drone['waypoints']:
                drone_waypoints.append([wp['x'], wp['y'], self.altitude])
            waypoint_config['waypoints'].append(np.array(drone_waypoints))
            
            # Initial position from first waypoint
            if drone_waypoints:
                first_wp = drone_waypoints[0]
                waypoint_config['init_positions'].append([first_wp[0], first_wp[1], 0.1])
            else:
                waypoint_config['init_positions'].append([0.0, 0.0, 0.1])
        
        waypoint_config['init_positions'] = np.array(waypoint_config['init_positions'])
        return waypoint_config

    def launch_simulation_direct(self):
        """Launch simulation directly with waypoints"""
        try:
            # Create simulation status window
            sim_window = tk.Toplevel(self.root)
            sim_window.title("PyBullet Simulation Status")
            sim_window.geometry("500x400")
            
            # Create status display
            status_frame = ttk.Frame(sim_window)
            status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(status_frame, text="PyBullet Simulation Status", 
                    font=('Arial', 14, 'bold')).pack(pady=(0, 10))
            
            status_text = tk.Text(status_frame, height=20, width=60, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=status_text.yview)
            status_text.configure(yscrollcommand=scrollbar.set)
            
            status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            def log_message(message):
                status_text.insert(tk.END, message + "\n")
                status_text.see(tk.END)
                status_text.update()
            
            # Prepare configuration
            log_message("🚁 Preparing PyBullet simulation configuration...")
            waypoint_config = self.prepare_waypoint_config()
            
            log_message(f"✅ Configuration ready:")
            log_message(f"   • Drones: {waypoint_config['num_drones']}")
            log_message(f"   • Building: {waypoint_config['building_width']}x{waypoint_config['building_height']}m")
            log_message(f"   • Altitude: {waypoint_config['altitude']}m")
            log_message(f"   • Total waypoints: {sum(len(wp) for wp in waypoint_config['waypoints'])}")
            
            for i, wp in enumerate(waypoint_config['waypoints']):
                log_message(f"   • Drone {i}: {len(wp)} waypoints")
            
            log_message("\n🚀 Starting PyBullet simulation...")
            log_message("📝 Check console for simulation details...")
            log_message("📸 Screenshots will be saved to: screenshots/")
            log_message("   • Start screenshot after initialization")
            log_message("   • End screenshot when simulation completes")
            
            # Launch simulation in separate thread
            def run_simulation():
                try:
                    # IMPORTANT: You need to modify this import path to match your PyBullet file
                    # If your PyBullet file is named 'simulation_main.py', change the import accordingly
                    from MainSim import run_simulation_with_gui_waypoints
                    
                    def update_success():
                        log_message("🎮 PyBullet simulation window should now be open!")
                        log_message("🎯 Using optimized waypoints from GUI...")
                    
                    sim_window.after(0, update_success)
                    
                    # Run the simulation
                    run_simulation_with_gui_waypoints(waypoint_config)
                    
                    def update_complete():
                        log_message("✅ Simulation completed successfully!")
                    
                    sim_window.after(0, update_complete)
                    
                except ImportError as import_err:
                    error_msg = str(import_err)
                    def update_import_error():
                        log_message(f"❌ Import Error: {error_msg}")
                        log_message("💡 Make sure to modify the import path in launch_simulation_direct()")
                        log_message("💡 Check that both files are in the same directory")
                        log_message(f"💡 Looking for: your_pybullet_filename.py")
                    
                    sim_window.after(0, update_import_error)
                    
                except Exception as general_err:
                    error_msg = str(general_err)
                    def update_general_error():
                        log_message(f"❌ Simulation Error: {error_msg}")
                        log_message("💡 Check console for more details")
                    
                    sim_window.after(0, update_general_error)
            
            # Start simulation thread
            sim_thread = threading.Thread(target=run_simulation, daemon=True)
            sim_thread.start()
            
            # Add close button
            ttk.Button(sim_window, text="Close", command=sim_window.destroy).pack(pady=10)
            
        except Exception as e:
            self.show_message("Error", f"Failed to launch simulation: {str(e)}")
    
    def show_message(self, title, message):
        """Show a message dialog"""
        try:
            import tkinter.messagebox as mb
            mb.showinfo(title, message)
        except:
            print(f"{title}: {message}")
    
    def generate_waypoint_summary(self):
        """Generate a quick summary of waypoints for display"""
        if not self.swarm:
            return "No swarm data available"
        
        summary_lines = []
        for drone in self.swarm.drones:
            waypoint_count = len(drone['waypoints'])
            if waypoint_count > 0:
                first_wp = drone['waypoints'][0]
                last_wp = drone['waypoints'][-1]
                summary_lines.append(
                    f"D{drone['id']}: {waypoint_count}pts "
                    f"({first_wp['x']:.1f},{first_wp['y']:.1f})→"
                    f"({last_wp['x']:.1f},{last_wp['y']:.1f})"
                )
            else:
                summary_lines.append(f"D{drone['id']}: No waypoints")
        
        return " | ".join(summary_lines)

    def point_in_building(self, point):
        """Check if a point is inside the building shape"""
        x, y = point['x'], point['y']
        
        if self.area_shape == 'circle':
            radius = min(self.area_width, self.area_height) / 2
            distance = math.sqrt(x*x + y*y)
            return distance <= radius
        elif self.area_shape == 'triangle':
            # Equilateral triangle with side length = area_width
            side_length = self.area_width
            height = side_length * math.sqrt(3) / 2
            
            # Triangle vertices: (0, height), (-side_length/2, -height/2), (side_length/2, -height/2)
            v0_x, v0_y = 0, height           # Top vertex
            v1_x, v1_y = -side_length/2, -height/2  # Bottom-left vertex
            v2_x, v2_y = side_length/2, -height/2   # Bottom-right vertex
            
            # Barycentric coordinates method
            denom = (v1_y - v2_y) * (v0_x - v2_x) + (v2_x - v1_x) * (v0_y - v2_y)
            
            if abs(denom) < 1e-10:
                return False
            
            a = ((v1_y - v2_y) * (x - v2_x) + (v2_x - v1_x) * (y - v2_y)) / denom
            b = ((v2_y - v0_y) * (x - v2_x) + (v0_x - v2_x) * (y - v2_y)) / denom
            c = 1 - a - b
            
            return a >= 0 and b >= 0 and c >= 0        
        else:  # rectangle or square
            return (-self.area_width/2 <= x <= self.area_width/2 and 
                    -self.area_height/2 <= y <= self.area_height/2)
    
    def calculate_camera_footprint(self, position):
        """Calculate camera footprint vertices"""
        fov_rad = math.radians(self.camera_fov)
        half_width = self.altitude * math.tan(fov_rad / 2)
        half_height = half_width
        
        vertices = [
            (position['x'] - half_width, position['y'] - half_height),
            (position['x'] + half_width, position['y'] - half_height),
            (position['x'] + half_width, position['y'] + half_height),
            (position['x'] - half_width, position['y'] + half_height)
        ]
        
        return vertices
    
    def point_in_polygon(self, point, vertices):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def calculate_optimal_points_per_drone(self):
        """
        Calculate optimal points per drone based on multiple factors
        """
        # Method 1: Coverage-based calculation
        building_area = self.area_width * self.area_height
        footprint_radius = self.altitude * math.tan(math.radians(self.camera_fov / 2))
        camera_footprint_area = math.pi * (footprint_radius ** 2)
        
        # Account for 30% overlap for thorough coverage
        effective_coverage = camera_footprint_area * 0.7
        total_points_needed = building_area / effective_coverage
        coverage_based_target = total_points_needed / self.n_drones
        
        # Method 2: Shape complexity adjustment
        shape_factors = {
            'rectangle': 1.0, 'square': 1.0, 'circle': 0.9,
            'triangle': 0.8, 'hexagon': 1.1, 'pentagon': 1.1,
            'octagon': 1.1, 'diamond': 0.9
        }
        shape_factor = shape_factors.get(self.area_shape, 1.0)
        
        # Method 3: Building size scaling
        base_area = 8000  # 100m × 80m reference
        area_ratio = building_area / base_area
        size_factor = math.sqrt(area_ratio)  # Square root to moderate scaling
        
        # Combine all factors
        adaptive_target = coverage_based_target * shape_factor * size_factor
        
        # Apply practical bounds based on drone capabilities
        min_target = 8   # Minimum for meaningful coverage
        max_target = 45  # Maximum for processing capacity
        
        optimal_target = max(min_target, min(max_target, adaptive_target))
        
        print(f"Adaptive target calculation:")
        print(f"  Building: {self.area_width}×{self.area_height}m ({self.area_shape})")
        print(f"  Coverage-based: {coverage_based_target:.1f}")
        print(f"  Shape factor: {shape_factor}")
        print(f"  Size factor: {size_factor:.2f}")
        print(f"  Final target: {optimal_target:.1f} points/drone")
        
        return optimal_target
    
    def calculate_swarm_fitness(self, grid_x, grid_y, strategy=None):
        """Calculate fitness score for dynamic swarm configuration with detailed logging"""
        # Temporarily set parameters
        original_x, original_y = self.grid_points_x, self.grid_points_y
        original_strategy = self.strategy
        
        self.grid_points_x, self.grid_points_y = int(grid_x), int(grid_y)
        
        if strategy:
            self.strategy = strategy
        
        # Generate temporary swarm configuration
        temp_swarm = DroneSwarm(self.n_drones, (self.area_width, self.area_height),
                               start_formation=self.start_formation, area_shape=self.area_shape)
        self.generate_global_grid()

        old_swarm = self.swarm
        self.swarm = temp_swarm
        self.assign_grid_points_dynamically()
        
        # Calculate detailed metrics
        total_distance = 0
        total_coverage = 0
        overlap_count = 0
        balance_score = 0
        
        # Debug: Print current configuration
        print(f"\n=== Fitness Calculation Debug ===")
        print(f"Grid: {grid_x}x{grid_y}, Drones: {self.n_drones}")
        print(f"Total grid points: {len(self.global_grid_points)}")
        
        # Generate paths and calculate metrics
        drone_distances = []
        drone_assignments = []
        
        for drone in temp_swarm.drones:
            assigned_points = [self.global_grid_points[i] for i in drone['assigned_points']]
            drone_assignments.append(len(assigned_points))

            print(f"DEBUG: Drone {drone['id']} has {len(assigned_points)} assigned points")
            
            if assigned_points:
                waypoints = self.generate_optimal_path(drone, assigned_points)
                
                # Calculate total distance for this drone
                drone_distance = 0
                for i in range(1, len(waypoints)):
                    drone_distance += self.distance(waypoints[i-1], waypoints[i])
                
                drone_distances.append(drone_distance)
                total_distance += drone_distance
                total_coverage += len(assigned_points)
                
                print(f"Drone {drone['id']}: {len(assigned_points)} points, {drone_distance:.1f}m distance")
            else:
                drone_distances.append(0)
                print(f"Drone {drone['id']}: 0 points, 0m distance")
        self.swarm = old_swarm
        
        # Calculate overlap penalty
        for i, drone1 in enumerate(temp_swarm.drones):
            for j, drone2 in enumerate(temp_swarm.drones):
                if i >= j:
                    continue
                
                # Check for overlapping assignments
                overlap = len(drone1['assigned_points'].intersection(drone2['assigned_points']))
                overlap_count += overlap
        
        # Calculate workload balance (lower std = better balance)
        if drone_assignments:
            mean_assignment = np.mean(drone_assignments)
            std_assignment = np.std(drone_assignments)
            if mean_assignment > 0:
                balance_score = 10 - (std_assignment / mean_assignment) * 5
                balance_score = max(balance_score, 0)
            else:
                balance_score = 0
        
        # Calculate individual fitness components
        if total_distance > 0:
            coverage_efficiency = total_coverage / total_distance
        else:
            coverage_efficiency = total_coverage
            
        overlap_penalty = overlap_count * self.overlap_penalty
        distance_penalty = total_distance * 0.01
        
        # Grid quality metrics
        points_per_drone = total_coverage / max(self.n_drones, 1)
        optimal_points_per_drone = self.calculate_optimal_points_per_drone()  # Target points per drone
        grid_quality = 10 - abs(points_per_drone - optimal_points_per_drone) * 0.2
        grid_quality = max(grid_quality, 0)
        
        # Multi-objective fitness with detailed components
        fitness_components = {
            'coverage_efficiency': coverage_efficiency * 50,
            'balance_score': balance_score * 15,
            'grid_quality': grid_quality * 10,
            'overlap_penalty': -overlap_penalty * 5,
            'distance_penalty': -distance_penalty,
            'assignment_bonus': total_coverage * 0.5
        }
        
        fitness = sum(fitness_components.values())
        
        # Add variance to break ties and encourage exploration
        grid_variance_bonus = abs(grid_x - grid_y) * 0.1  # Small bonus for different aspect ratios
        fitness += grid_variance_bonus
        
        # Debug: Print fitness breakdown
        print(f"Fitness components:")
        for component, value in fitness_components.items():
            print(f"  {component}: {value:.2f}")
        print(f"Grid variance bonus: {grid_variance_bonus:.2f}")
        print(f"Total fitness: {fitness:.2f}")
        print("=" * 40)
        
        # Restore original parameters
        self.grid_points_x, self.grid_points_y = original_x, original_y
        self.strategy = original_strategy

        # Enhanced coverage efficiency calculation
        if hasattr(self, 'enhanced_optimizer'):
            total_enhanced_coverage = 0
            total_enhanced_distance = 0
            
            for drone in temp_swarm.drones:
                assigned_points = [self.global_grid_points[i] for i in drone['assigned_points']]
                if assigned_points:
                    # Use enhanced optimizer to calculate theoretical optimal coverage
                    optimal_waypoints = self.enhanced_optimizer.generate_coverage_optimal_waypoints(
                        drone['start_position'], assigned_points
                    )
                    
                    # Calculate coverage percentage from optimal waypoints
                    covered_points = set()
                    for waypoint in optimal_waypoints:
                        coverage = self.enhanced_optimizer.calculate_coverage_from_position(
                            waypoint, assigned_points
                        )
                        covered_points.update(coverage)
                    
                    coverage_ratio = len(covered_points) / len(assigned_points)
                    total_enhanced_coverage += coverage_ratio
                    
                    # Calculate optimal distance
                    for i in range(1, len(optimal_waypoints)):
                        total_enhanced_distance += self.enhanced_optimizer.euclidean_distance(
                            optimal_waypoints[i-1], optimal_waypoints[i]
                        )
            
            # Enhanced fitness components
            if total_enhanced_distance > 0:
                enhanced_efficiency = (total_enhanced_coverage / total_enhanced_distance) * 100
            else:
                enhanced_efficiency = total_enhanced_coverage * 100
            
            # Add enhanced efficiency to fitness
            fitness += enhanced_efficiency * 2  # Weight enhanced efficiency highly
        
        return fitness
    
    def calculate_enhanced_coverage_metrics(self):
        """Calculate enhanced coverage metrics using the optimizer"""
        if not self.swarm or not hasattr(self, 'enhanced_optimizer'):
            return {}
        
        total_theoretical_coverage = 0
        total_actual_coverage = 0
        total_optimal_distance = 0
        total_actual_distance = 0
        
        for drone in self.swarm.drones:
            assigned_points = [self.global_grid_points[i] for i in drone['assigned_points']]
            if assigned_points:
                # Calculate theoretical optimal
                optimal_waypoints = self.enhanced_optimizer.generate_coverage_optimal_waypoints(
                    drone['start_position'], assigned_points
                )
                
                # Theoretical coverage
                covered_points = set()
                for waypoint in optimal_waypoints:
                    coverage = self.enhanced_optimizer.calculate_coverage_from_position(
                        waypoint, assigned_points
                    )
                    covered_points.update(coverage)
                
                total_theoretical_coverage += len(covered_points)
                total_actual_coverage += len(drone['covered_points'])
                
                # Distance comparison
                for i in range(1, len(optimal_waypoints)):
                    total_optimal_distance += self.enhanced_optimizer.euclidean_distance(
                        optimal_waypoints[i-1], optimal_waypoints[i]
                    )
                
                total_actual_distance += drone['distance_traveled']
        
        return {
            'theoretical_coverage': total_theoretical_coverage,
            'actual_coverage': total_actual_coverage,
            'coverage_efficiency': (total_actual_coverage / max(total_theoretical_coverage, 1)) * 100,
            'optimal_distance': total_optimal_distance,
            'actual_distance': total_actual_distance,
            'distance_efficiency': (total_optimal_distance / max(total_actual_distance, 1)) * 100
        }
    
    def start_optimization(self):
        """Start Bayesian optimization for dynamic assignment"""
        if self.optimization_running:
            return
        
        self.optimization_running = True
        self.optimize_btn.config(state='disabled')
        self.stop_opt_btn.config(state='normal')
        self.optimization_status_var.set("Optimizing with enhanced path algorithm...")
        
        # Reset optimizer
        current_bounds = self.calculate_optimal_grid_bounds()

        print(f"🎯 Optimization bounds for {self.area_width}×{self.area_height}m building:")
        print(f"   Camera: {self.camera_fov}° FOV at {self.altitude}m altitude")
        print(f"   Footprint: {current_bounds['footprint_diameter']:.1f}m diameter")
        print(f"   Optimal grid: {current_bounds['optimal_x']}×{current_bounds['optimal_y']}")
        print(f"   Search bounds: X[{current_bounds['x_min']}-{current_bounds['x_max']}], Y[{current_bounds['y_min']}-{current_bounds['y_max']}]")
        print(f"   Search space: {(current_bounds['x_max']-current_bounds['x_min']+1) * (current_bounds['y_max']-current_bounds['y_min']+1)} combinations")

        self.bayesian_optimizer = BayesianOptimizer(bounds=[
            (current_bounds['x_min'], current_bounds['x_max']),
            (current_bounds['y_min'], current_bounds['y_max'])
        ])
        #self.bayesian_optimizer = BayesianOptimizer(bounds=[(5, 25), (5, 20)])
        self.optimization_history = []
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self.run_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
    
    def stop_optimization(self):
        """Stop optimization process"""
        self.optimization_running = False
        self.optimize_btn.config(state='normal')
        self.stop_opt_btn.config(state='disabled')
        self.optimization_status_var.set("Optimization stopped")
    
    def run_optimization(self):
        """Run the Bayesian optimization loop with improved parameter space"""

        import random

        base_seed = 42
        time_seed = int(time.time()) % 1000
        optimization_seed = base_seed + time_seed

        random.seed(optimization_seed)
        np.random.seed(optimization_seed)


        max_iterations = self.max_iterations_var.get()
        
        print(f"\n🧠 Starting Bayesian Optimization for {max_iterations} iterations")
        print(f"Parameter space: Grid X[5-25], Grid Y[5-20]")
        print(f"Current config: {self.n_drones} drones, {self.start_formation} formation")
        
        for iteration in range(max_iterations):
            if not self.optimization_running:
                break
            
            # Get next point to evaluate
            next_point = self.bayesian_optimizer.suggest_next_point()
            grid_x, grid_y = next_point
            
            print(f"\nIteration {iteration + 1}: Testing Grid {grid_x}x{grid_y}")
            
            # Evaluate fitness
            fitness = self.calculate_swarm_fitness(grid_x, grid_y, 'dynamic_coverage')

            print(f"Using enhanced coverage optimization: Grid {grid_x}x{grid_y}")
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'grid_x': grid_x,
                'grid_y': grid_y,
                'fitness': fitness,
                'is_best': len(self.optimization_history) == 0 or fitness > max(h['fitness'] for h in self.optimization_history)
            })
            
            # Add observation
            self.bayesian_optimizer.add_observation([grid_x, grid_y], fitness)
            
            print(f"Fitness: {fitness:.2f}")
            if len(self.optimization_history) > 1:
                best_so_far = max(h['fitness'] for h in self.optimization_history)
                print(f"Best so far: {best_so_far:.2f}")
            
            # Update status
            self.root.after(0, lambda i=iteration+1: 
                          self.optimization_status_var.set(f"Optimization {i}/{max_iterations} - Current: {fitness:.1f}"))
            
            # Update plots
            self.root.after(0, self.update_optimization_plots)
            
            # Apply best parameters
            if fitness == max(h['fitness'] for h in self.optimization_history):
                self.root.after(0, lambda: self.apply_optimal_parameters(grid_x, grid_y))
                print(f"🎯 New best configuration applied!")
            
            time.sleep(0.1)
        
        # Optimization complete
        if self.optimization_running:
            best_result = max(self.optimization_history, key=lambda x: x['fitness'])
            
            print(f"\n✅ Optimization Complete!")
            print(f"Best configuration: {best_result['grid_x']:.0f}x{best_result['grid_y']:.0f}")
            print(f"Best fitness: {best_result['fitness']:.3f}")
            
            self.root.after(0, lambda: self.optimization_status_var.set(
                f"Complete! Best: ({best_result['grid_x']:.0f},{best_result['grid_y']:.0f}) "
                f"Fitness: {best_result['fitness']:.3f}"))
            
            self.root.after(0, lambda: self.apply_optimal_parameters(
                best_result['grid_x'], best_result['grid_y']))
        
        self.optimization_running = False
        self.root.after(0, lambda: self.optimize_btn.config(state='normal'))
        self.root.after(0, lambda: self.stop_opt_btn.config(state='disabled'))
    
    def apply_optimal_parameters(self, grid_x, grid_y):
        """Apply optimal parameters found by optimization"""
        self.grid_points_x_var.set(int(grid_x))
        self.grid_points_y_var.set(int(grid_y))
        self.on_parameter_change()
    
    def update_optimization_plots(self):
        """Update optimization visualization plots"""
        # Completely clear and recreate the figure
        self.opt_fig.clf()  # Clear the entire figure
        
        # Recreate all subplots fresh
        self.gp_ax = self.opt_fig.add_subplot(2, 2, 1)
        self.acq_ax = self.opt_fig.add_subplot(2, 2, 2)
        self.score_ax = self.opt_fig.add_subplot(2, 2, 3)
        self.param_ax = self.opt_fig.add_subplot(2, 2, 4)
        
        if len(self.bayesian_optimizer.X) > 0:
            X_obs = np.array(self.bayesian_optimizer.X)
            y_obs = np.array(self.bayesian_optimizer.y)
            
            try:
                # Create grid for visualization
                x_range = np.linspace(5, 25, 20)
                y_range = np.linspace(5, 20, 15)
                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                
                # Predict for grid points
                X_test = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                mu, sigma = self.bayesian_optimizer.gp_predict(X_test)
                
                mu_grid = mu.reshape(X_grid.shape)
                
                # 1. GP Mean surface - FRESH PLOT
                cs1 = self.gp_ax.contourf(X_grid, Y_grid, mu_grid, levels=15, cmap='viridis', alpha=0.8)
                self.gp_ax.set_title('GP Mean (Dynamic Assignment Fitness)')
                self.gp_ax.set_xlabel('Grid Points X')
                self.gp_ax.set_ylabel('Grid Points Y')
                
                # Plot observed points
                self.gp_ax.scatter(X_obs[:, 0], X_obs[:, 1], c=y_obs, 
                                cmap='viridis', s=100, edgecolors='white', linewidth=2)
                
                # Highlight best point
                if len(y_obs) > 0:
                    best_idx = np.argmax(y_obs)
                    self.gp_ax.scatter(X_obs[best_idx, 0], X_obs[best_idx, 1], 
                                    marker='*', s=300, color='red', edgecolors='white', linewidth=2)
                
                # Add colorbar
                self.opt_fig.colorbar(cs1, ax=self.gp_ax, shrink=0.8)
                
                # 2. Acquisition function - FRESH PLOT
                acq_values = self.bayesian_optimizer.acquisition_function(X_test)
                acq_grid = acq_values.reshape(X_grid.shape)
                
                cs2 = self.acq_ax.contourf(X_grid, Y_grid, acq_grid, levels=15, cmap='plasma', alpha=0.8)
                self.acq_ax.set_title('Acquisition Function')
                self.acq_ax.set_xlabel('Grid Points X')
                self.acq_ax.set_ylabel('Grid Points Y')
                
                self.acq_ax.scatter(X_obs[:, 0], X_obs[:, 1], c='white', s=50, edgecolors='black')
                
                # Add colorbar
                self.opt_fig.colorbar(cs2, ax=self.acq_ax, shrink=0.8)
                
            except Exception as e:
                self.gp_ax.text(0.5, 0.5, f'GP Error: {str(e)[:30]}...', 
                            transform=self.gp_ax.transAxes, ha='center', va='center')
        
        # 3. Optimization history - FRESH PLOT
        if self.optimization_history:
            iterations = [h['iteration'] for h in self.optimization_history]
            fitness_scores = [h['fitness'] for h in self.optimization_history]
            
            # Fitness progression
            self.score_ax.plot(iterations, fitness_scores, 'b-o', markersize=4, linewidth=2)
            
            # Best scores
            best_scores = []
            current_best = float('-inf')
            for score in fitness_scores:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            self.score_ax.plot(iterations, best_scores, 'r-', linewidth=3, alpha=0.7, label='Best So Far')
            self.score_ax.set_title('Dynamic Assignment Fitness Evolution')
            self.score_ax.set_xlabel('Iteration')
            self.score_ax.set_ylabel('Fitness Score')
            self.score_ax.grid(True, alpha=0.3)
            self.score_ax.legend()
            
            # 4. Parameter exploration - FRESH PLOT
            grid_x_vals = [h['grid_x'] for h in self.optimization_history]
            grid_y_vals = [h['grid_y'] for h in self.optimization_history]
            
            scatter = self.param_ax.scatter(grid_x_vals, grid_y_vals, c=fitness_scores, 
                                        cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Highlight best point
            best_point = max(self.optimization_history, key=lambda x: x['fitness'])
            self.param_ax.scatter(best_point['grid_x'], best_point['grid_y'], 
                                marker='*', s=300, color='red', edgecolors='white', linewidth=2)
            
            self.param_ax.set_title('Parameter Space Exploration')
            self.param_ax.set_xlabel('Grid Points X')
            self.param_ax.set_ylabel('Grid Points Y')
            self.param_ax.grid(True, alpha=0.3)
            self.opt_fig.colorbar(scatter, ax=self.param_ax, shrink=0.8, label='Fitness Score')
        
        # Update canvas
        self.opt_fig.tight_layout()
        self.opt_canvas.draw()
    
    def update_analysis_plots(self):
        """Update dynamic assignment analysis plots"""
        # Clear axes
        for ax in [self.coverage_ax, self.efficiency_ax, self.assignment_ax, self.overlap_ax]:
            ax.clear()
        
        if self.swarm and len(self.swarm.drones) > 0:
            # Coverage analysis per drone
            drone_ids = [f"Drone {d['id']}" for d in self.swarm.drones]
            assigned_counts = [len(d['assigned_points']) for d in self.swarm.drones]
            covered_counts = [len(d['covered_points']) for d in self.swarm.drones]
            
            x_pos = np.arange(len(drone_ids))
            self.coverage_ax.bar(x_pos, assigned_counts, alpha=0.7, label='Assigned Points',
                               color=[d['color'] for d in self.swarm.drones])
            self.coverage_ax.bar(x_pos, covered_counts, alpha=0.9, label='Covered Points',
                               color=[d['color'] for d in self.swarm.drones])
            
            self.coverage_ax.set_title('Dynamic Assignment vs Coverage')
            self.coverage_ax.set_xlabel('Drone')
            self.coverage_ax.set_ylabel('Grid Points')
            self.coverage_ax.set_xticks(x_pos)
            self.coverage_ax.set_xticklabels(drone_ids, rotation=45)
            self.coverage_ax.legend()
            
            # Efficiency analysis
            distances = [d['distance_traveled'] for d in self.swarm.drones]
            efficiencies = [len(d['covered_points']) / max(d['distance_traveled'], 1) for d in self.swarm.drones]
            
            self.efficiency_ax.bar(x_pos, efficiencies, alpha=0.7, 
                                 color=[d['color'] for d in self.swarm.drones])
            self.efficiency_ax.set_title('Coverage Efficiency by Drone')
            self.efficiency_ax.set_xlabel('Drone')
            self.efficiency_ax.set_ylabel('Points per Meter')
            self.efficiency_ax.set_xticks(x_pos)
            self.efficiency_ax.set_xticklabels(drone_ids, rotation=45)
            
            # Assignment distribution
            if len(self.swarm.drones) > 1:
                self.assignment_ax.pie(assigned_counts, labels=drone_ids, autopct='%1.1f%%',
                                     colors=[d['color'] for d in self.swarm.drones])
                self.assignment_ax.set_title('Point Assignment Distribution')
            else:
                self.assignment_ax.text(0.5, 0.5, 'Single Drone\nNo Distribution', 
                                      ha='center', va='center', fontsize=14)
                self.assignment_ax.set_title('Assignment Analysis')
            
            # Overlap analysis
            if len(self.swarm.drones) > 1:
                overlap_matrix = np.zeros((len(self.swarm.drones), len(self.swarm.drones)))
                
                for i, drone1 in enumerate(self.swarm.drones):
                    for j, drone2 in enumerate(self.swarm.drones):
                        if i != j:
                            # Calculate potential coverage overlap
                            overlap_count = 0
                            for point_idx in drone1['assigned_points']:
                                point = self.global_grid_points[point_idx]
                                for other_point_idx in drone2['assigned_points']:
                                    other_point = self.global_grid_points[other_point_idx]
                                    if self.distance(point, other_point) < 20:  # Within potential overlap distance
                                        overlap_count += 1
                            overlap_matrix[i, j] = overlap_count
                
                # Only create colorbar if there's actual data variation
                if np.max(overlap_matrix) > np.min(overlap_matrix):
                    im = self.overlap_ax.imshow(overlap_matrix, cmap='Reds', alpha=0.8)
                    try:
                        plt.colorbar(im, ax=self.overlap_ax, shrink=0.8)
                    except ValueError:
                        # Skip colorbar if there's a layout issue
                        pass
                else:
                    im = self.overlap_ax.imshow(overlap_matrix, cmap='Reds', alpha=0.8)
                
                self.overlap_ax.set_title('Potential Coverage Overlap')
                self.overlap_ax.set_xlabel('Drone')
                self.overlap_ax.set_ylabel('Drone')
                self.overlap_ax.set_xticks(range(len(drone_ids)))
                self.overlap_ax.set_yticks(range(len(drone_ids)))
                self.overlap_ax.set_xticklabels([f"D{i}" for i in range(len(drone_ids))])
                self.overlap_ax.set_yticklabels([f"D{i}" for i in range(len(drone_ids))])
                
                # Add text annotations
                for i in range(len(drone_ids)):
                    for j in range(len(drone_ids)):
                        text = self.overlap_ax.text(j, i, f'{int(overlap_matrix[i, j])}',
                                                  ha="center", va="center", color="black", fontweight='bold')
            else:
                self.overlap_ax.text(0.5, 0.5, 'Single Drone\nNo Overlap', 
                                   ha='center', va='center', fontsize=14)
                self.overlap_ax.set_title('Overlap Analysis')
        
        # Update canvas
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw()
    
    def on_parameter_change(self):
        """Handle parameter changes"""
        # Update grid bounds when building dimensions change
        if hasattr(self, 'area_width') and hasattr(self, 'area_height'):
            old_bounds = getattr(self, 'current_grid_bounds', None)
            new_bounds = self.calculate_optimal_grid_bounds()
            
            # Only update if bounds actually changed (avoid infinite recursion)
            if (old_bounds is None or 
                old_bounds['x_max'] != new_bounds['x_max'] or 
                old_bounds['y_max'] != new_bounds['y_max']):
                self.update_grid_bounds()
        
        if not self.is_running and not self.optimization_running:
            self.initialize_swarm()
    
    def update_swarm_info(self):
        """Update swarm information display"""
        if self.swarm:
            total_waypoints = sum(len(drone['waypoints']) - 1 for drone in self.swarm.drones)  # Exclude start positions
            total_assigned = sum(len(drone['assigned_points']) for drone in self.swarm.drones)
            
            sample_footprint = self.calculate_camera_footprint({'x': 0, 'y': 0})
            footprint_area = self.calculate_footprint_area(sample_footprint)
            
            info_text = f"Formation: {self.start_formation} | Drones: {self.n_drones} | Assigned: {total_assigned} | Waypoints: {total_waypoints} | Path: {self.path_strategy}"
            self.swarm_info_label.config(text=info_text)
    
    def calculate_footprint_area(self, vertices):
        """Calculate polygon area using shoelace formula"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0
    
    def create_drone_status_widgets(self):
        """Create individual drone status widgets"""
        # Clear existing widgets
        for widget in self.drone_status_frame.winfo_children():
            widget.destroy()
        
        # Create status for each drone
        for drone in self.swarm.drones:
            drone_frame = ttk.LabelFrame(self.drone_status_frame, 
                                       text=f"Drone {drone['id']}", padding=5)
            drone_frame.pack(fill=tk.X, pady=2)
            
            # Status variables for this drone
            drone['status_vars'] = {
                'position': tk.StringVar(value="(0.0, 0.0)"),
                'assigned': tk.StringVar(value="0 points"),
                'waypoints': tk.StringVar(value="0 waypoints"),
                'covered': tk.StringVar(value="0 / 0"),
                'distance': tk.StringVar(value="0.0 m"),
                'status': tk.StringVar(value="Ready")
            }
            
            # Create status display
            status_items = [
                ("Position:", drone['status_vars']['position']),
                ("Assigned:", drone['status_vars']['assigned']),
                ("Waypoints:", drone['status_vars']['waypoints']),
                ("Covered:", drone['status_vars']['covered']),
                ("Distance:", drone['status_vars']['distance']),
                ("Status:", drone['status_vars']['status'])
            ]
            
            for label_text, var in status_items:
                item_frame = ttk.Frame(drone_frame)
                item_frame.pack(fill=tk.X)
                ttk.Label(item_frame, text=label_text, width=10).pack(side=tk.LEFT)
                ttk.Label(item_frame, textvariable=var, font=('Arial', 9), 
                         foreground=drone['color']).pack(side=tk.LEFT)
    
    def start_mission(self):
        """Start the swarm mission"""
        if not self.swarm or self.optimization_running:
            return
        
        self.is_running = True
        self.is_paused = False
        self.start_time = time.time() - self.elapsed_time
        
        # Initialize all drones
        for drone in self.swarm.drones:
            if len(drone['waypoints']) > 0:
                drone['current_waypoint_index'] = 0
                drone['position'] = drone['start_position'].copy()
                drone['target_position'] = drone['waypoints'][1].copy() if len(drone['waypoints']) > 1 else drone['position'].copy()
                drone['path'] = [drone['position'].copy()]
                drone['distance_traveled'] = 0
                drone['covered_points'] = set()
                drone['covered_area_cells'] = set()
                drone['is_active'] = True
                drone['mission_complete'] = False
                
                # Reset grid point coverage
                for point_idx in drone['assigned_points']:
                    self.global_grid_points[point_idx]['covered'] = False
        
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        
        # Start animation thread
        self.animation_thread = threading.Thread(target=self.animate_swarm)
        self.animation_thread.daemon = True
        self.animation_thread.start()
    
    def pause_mission(self):
        """Pause/resume the swarm mission"""
        if self.is_running:
            self.is_paused = not self.is_paused
            self.pause_btn.config(text="Resume" if self.is_paused else "Pause")
            if not self.is_paused:
                self.start_time = time.time() - self.elapsed_time
    
    def reset_mission(self):
        """Reset the swarm mission"""
        self.is_running = False
        self.is_paused = False
        self.elapsed_time = 0
        
        if self.swarm:
            for drone in self.swarm.drones:
                drone['current_waypoint_index'] = 0
                drone['position'] = drone['start_position'].copy()
                drone['path'] = []
                drone['distance_traveled'] = 0
                drone['covered_points'] = set()
                drone['covered_area_cells'] = set()
                drone['is_active'] = True
                drone['mission_complete'] = False
                
                # Reset assigned grid points
                for point_idx in drone['assigned_points']:
                    self.global_grid_points[point_idx]['covered'] = False
        
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text='Pause')
        
        self.update_display()
        self.draw()
    
    def animate_swarm(self):
        """Animate the drone swarm"""
        while self.is_running:
            if not self.is_paused:
                now = time.time()
                self.elapsed_time = now - self.start_time
                
                active_drones = 0
                
                # Update each drone
                for drone in self.swarm.drones:
                    if drone['is_active'] and not drone['mission_complete']:
                        active_drones += 1
                        
                        # Move towards target waypoint
                        if drone['current_waypoint_index'] < len(drone['waypoints']) - 1:
                            target = drone['waypoints'][drone['current_waypoint_index'] + 1]
                            distance = self.distance(drone['position'], target)
                            
                            if distance > 0.1:
                                # Calculate movement
                                move_distance = self.drone_speed * 0.05
                                ratio = min(move_distance / distance, 1)
                                
                                new_x = drone['position']['x'] + (target['x'] - drone['position']['x']) * ratio
                                new_y = drone['position']['y'] + (target['y'] - drone['position']['y']) * ratio
                                
                                # Update distance traveled
                                drone['distance_traveled'] += self.distance(drone['position'], {'x': new_x, 'y': new_y})
                                
                                drone['position'] = {'x': new_x, 'y': new_y}
                                drone['path'].append(drone['position'].copy())
                                
                                # Update coverage at current position
                                self.update_drone_coverage(drone)
                            else:
                                # Reached waypoint
                                drone['current_waypoint_index'] += 1
                                drone['position'] = target.copy()
                                self.update_drone_coverage(drone)
                        else:
                            # Mission complete for this drone
                            drone['mission_complete'] = True
                            drone['is_active'] = False
                
                # Update display on main thread
                self.root.after(0, self.update_display)
                self.root.after(0, self.draw)
                self.root.after(0, self.update_analysis_plots)
                
                # Check if all drones completed mission
                if active_drones == 0:
                    self.is_running = False
                    self.root.after(0, lambda: self.start_btn.config(state='normal'))
                    self.root.after(0, lambda: self.pause_btn.config(state='disabled'))
                    break
            
            time.sleep(0.05)
    
    def update_drone_coverage(self, drone):
        """Update coverage for a specific drone"""
        footprint_vertices = self.calculate_camera_footprint(drone['position'])
        
        # Check assigned points for coverage
        for point_idx in drone['assigned_points']:
            point = self.global_grid_points[point_idx]
            if not point['covered']:
                if self.point_in_polygon((point['x'], point['y']), footprint_vertices):
                    point['covered'] = True
                    drone['covered_points'].add(point_idx)
        
        # Update area coverage
        covered_cells = self.get_area_cells_in_footprint(drone['position'])
        drone['covered_area_cells'].update(covered_cells)
    
    def get_area_cells_in_footprint(self, position):
        """Get area cells covered by camera footprint"""
        footprint_vertices = self.calculate_camera_footprint(position)
        covered_cells = set()
        
        bounds = self.get_building_bounds()
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        
        x_steps = int((x_max - x_min) / self.area_resolution)
        y_steps = int((y_max - y_min) / self.area_resolution)
        
        for i in range(x_steps):
            for j in range(y_steps):
                cell_x = x_min + i * self.area_resolution + self.area_resolution/2
                cell_y = y_min + j * self.area_resolution + self.area_resolution/2
                
                if self.point_in_polygon((cell_x, cell_y), footprint_vertices):
                    covered_cells.add((i, j))
        
        return covered_cells
    
    def update_display(self):
        """Update all display elements"""
        if not self.swarm:
            return
        
        # Update main swarm status
        stats = self.swarm.get_swarm_statistics()
        
        time_str = self.format_time(int(self.elapsed_time))
        self.mission_time_var.set(time_str)
        self.active_drones_var.set(f"{stats['active_drones']} / {self.n_drones}")
        self.total_distance_var.set(f"{stats['total_distance']:.1f} m")
        
        # Calculate overall coverage
        total_covered = stats['total_covered_points']
        total_assigned = sum(len(drone['assigned_points']) for drone in self.swarm.drones)
        coverage_percent = (total_covered / max(total_assigned, 1)) * 100
        
        self.swarm_coverage_var.set(f"{coverage_percent:.1f}%")
        self.swarm_progress_var.set(coverage_percent)
        
        # Calculate overlap ratio
        all_covered_points = set()
        overlap_points = set()
        for drone in self.swarm.drones:
            for point_idx in drone['covered_points']:
                if point_idx in all_covered_points:
                    overlap_points.add(point_idx)
                all_covered_points.add(point_idx)
        
        overlap_ratio = (len(overlap_points) / max(len(all_covered_points), 1)) * 100
        self.overlap_ratio_var.set(f"{overlap_ratio:.1f}%")
        
        # Calculate efficiency
        efficiency = total_covered / max(stats['total_distance'], 1)
        self.efficiency_var.set(f"{efficiency:.3f}")
        
        # Update individual drone status
        for drone in self.swarm.drones:
            if 'status_vars' in drone:
                pos_text = f"({drone['position']['x']:.1f}, {drone['position']['y']:.1f})"
                drone['status_vars']['position'].set(pos_text)
                
                assigned_text = f"{len(drone['assigned_points'])} points"
                drone['status_vars']['assigned'].set(assigned_text)
                
                waypoint_text = f"{len(drone['waypoints']) - 1} waypoints"  # Exclude start position
                drone['status_vars']['waypoints'].set(waypoint_text)
                
                covered_text = f"{len(drone['covered_points'])} / {len(drone['assigned_points'])}"
                drone['status_vars']['covered'].set(covered_text)
                
                drone['status_vars']['distance'].set(f"{drone['distance_traveled']:.1f} m")
                
                if drone['mission_complete']:
                    status_text = "Complete"
                elif drone['is_active']:
                    status_text = "Active"
                else:
                    status_text = "Inactive"
                drone['status_vars']['status'].set(status_text)

        if hasattr(self, 'enhanced_optimizer'):
            enhanced_metrics = self.calculate_enhanced_coverage_metrics()
            if enhanced_metrics:
                # Update efficiency display with enhanced metrics
                enhanced_eff = enhanced_metrics.get('coverage_efficiency', 0)
                distance_eff = enhanced_metrics.get('distance_efficiency', 0)
                
                # Modify the efficiency display to show enhanced metrics
                combined_efficiency = (enhanced_eff + distance_eff) / 200  # Normalize to 0-1
                self.efficiency_var.set(f"{combined_efficiency:.3f}")
    
    def format_time(self, seconds):
        """Format time as MM:SS"""
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def draw(self):
        """Draw the dynamic swarm simulation with adaptive plot size"""
        self.ax.clear()
        
        # Get building bounds and set adaptive plot limits
        bounds = self.get_building_bounds()
        
        # Calculate appropriate margins based on building size
        margin_x = max(10, bounds['width'] * 0.15)  # 15% margin or min 10m
        margin_y = max(10, bounds['height'] * 0.15)  # 15% margin or min 10m
        
        # Set plot limits to fit the building with appropriate margins
        self.ax.set_xlim(bounds['x_min'] - margin_x, bounds['x_max'] + margin_x)
        self.ax.set_ylim(bounds['y_min'] - margin_y, bounds['y_max'] + margin_y)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Distance (m)')
        
        # Draw building boundary with different shapes
        if self.area_shape == 'circle':
            # Draw circle
            radius = min(self.area_width, self.area_height) / 2
            circle = Circle((0, 0), radius=radius, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(circle)
        elif self.area_shape == 'triangle':
            # Draw equilateral triangle centered at origin
            side_length = self.area_width
            height = side_length * math.sqrt(3) / 2
            triangle_points = [
                (0, 2*height/3),                    # Top vertex
                (-side_length/2, -height/3),        # Bottom left
                (side_length/2, -height/3)          # Bottom right
            ]
            triangle = Polygon(triangle_points, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(triangle)
        elif self.area_shape == 'hexagon':
            # Draw regular hexagon
            radius = min(self.area_width, self.area_height) / 2
            hex_points = []
            for i in range(6):
                angle = i * math.pi / 3
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                hex_points.append((x, y))
            hexagon = Polygon(hex_points, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(hexagon)
        elif self.area_shape == 'pentagon':
            # Draw regular pentagon
            radius = min(self.area_width, self.area_height) / 2
            pent_points = []
            for i in range(5):
                angle = 2 * math.pi * i / 5 - math.pi/2  # Start from top
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pent_points.append((x, y))
            pentagon = Polygon(pent_points, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(pentagon)
        elif self.area_shape == 'octagon':
            # Draw regular octagon
            radius = min(self.area_width, self.area_height) / 2
            oct_points = []
            for i in range(8):
                angle = 2 * math.pi * i / 8
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                oct_points.append((x, y))
            octagon = Polygon(oct_points, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(octagon)
        elif self.area_shape == 'diamond':
            # Draw diamond (rotated square)
            size = min(self.area_width, self.area_height) / 2
            diamond_points = [
                (0, size),      # Top
                (size, 0),      # Right
                (0, -size),     # Bottom
                (-size, 0)      # Left
            ]
            diamond = Polygon(diamond_points, fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(diamond)

        else:
            # Draw rectangle/square
            boundary_rect = Rectangle((bounds['x_min'], bounds['y_min']), 
                                    bounds['width'], bounds['height'],
                                    fill=False, edgecolor='black', linewidth=3)
            self.ax.add_patch(boundary_rect)
            self.ax.add_patch(boundary_rect)
        
        if not self.swarm:
            self.ax.set_title("Initializing Dynamic Swarm...")
            self.canvas.draw()
            return
        
        # Draw covered area cells with size-appropriate resolution
        all_covered_cells = set()
        for drone in self.swarm.drones:
            all_covered_cells.update(drone['covered_area_cells'])
        
        # Adjust cell visualization based on building size
        cell_size = max(1, min(self.area_resolution, bounds['width'] / 50))
        
        for cell in all_covered_cells:
            i, j = cell
            x = bounds['x_min'] + i * self.area_resolution
            y = bounds['y_min'] + j * self.area_resolution
            coverage_rect = Rectangle((x, y), cell_size, cell_size,
                                    fill=True, facecolor='lightgreen', alpha=0.3, edgecolor='none')
            self.ax.add_patch(coverage_rect)
        
        # Draw grid points with size-appropriate markers
        grid_marker_size = max(4, min(8, bounds['width'] / 20))  # Adaptive marker size
        
        for point in self.global_grid_points:
            if point['assigned_drone'] >= 0:
                # Assigned point - use drone color
                drone_color = self.swarm.drones[point['assigned_drone']]['color']
                marker_color = 'green' if point['covered'] else drone_color
                marker_size = grid_marker_size + 2 if point['covered'] else grid_marker_size
                alpha = 1.0 if point['covered'] else 0.7
            else:
                # Unassigned point
                marker_color = 'gray'
                marker_size = grid_marker_size - 1
                alpha = 0.5
            
            self.ax.plot(point['x'], point['y'], 'o', color=marker_color, 
                        markersize=marker_size, alpha=alpha)
        
        # Draw drone-specific elements
        for drone in self.swarm.drones:
            # Draw assignment connections (lines from drone to assigned points)
            drone_pos = drone['position']
            for point_idx in drone['assigned_points']:
                point = self.global_grid_points[point_idx]
                self.ax.plot([drone_pos['x'], point['x']], [drone_pos['y'], point['y']], 
                           color=drone['color'], alpha=0.2, linewidth=1, linestyle=':')
            
            # Draw waypoints and planned path
            if len(drone['waypoints']) > 1:
                waypoint_x = [wp['x'] for wp in drone['waypoints']]
                waypoint_y = [wp['y'] for wp in drone['waypoints']]
                self.ax.plot(waypoint_x, waypoint_y, color=drone['color'], alpha=0.6, 
                           linestyle='--', linewidth=2, label=f'Drone {drone["id"]} Plan')
                self.ax.plot(waypoint_x, waypoint_y, 'o', color=drone['color'], 
                           markersize=6, alpha=0.6)
            
            # Draw completed path
            if len(drone['path']) > 1:
                path_x = [p['x'] for p in drone['path']]
                path_y = [p['y'] for p in drone['path']]
                self.ax.plot(path_x, path_y, color=drone['color'], linewidth=3, alpha=0.8)
            
            # Draw starting position
            start_circle = Circle((drone['start_position']['x'], drone['start_position']['y']),
                                radius=1.5, fill=True, facecolor=drone['color'], alpha=0.3,
                                edgecolor=drone['color'], linewidth=1)
            self.ax.add_patch(start_circle)
            
            # Draw drone current position
            drone_circle = Circle((drone['position']['x'], drone['position']['y']),
                                radius=2, fill=True, facecolor=drone['color'], 
                                edgecolor='white', linewidth=2)
            self.ax.add_patch(drone_circle)
            
            # Add drone ID label (simpler version)
            drone_label = f"D{drone['id']}"
            self.ax.text(drone['position']['x'], drone['position']['y'] + 4, 
                        drone_label, ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=drone['color'])
            
            # Draw current mission progress indicators during simulation
            if drone['is_active'] and self.is_running:
                # Highlight current target waypoint with yellow outline
                if len(drone['waypoints']) > 1:
                    current_wp_idx = min(drone['current_waypoint_index'] + 1, len(drone['waypoints']) - 1)
                    if current_wp_idx < len(drone['waypoints']):
                        target_wp = drone['waypoints'][current_wp_idx]
                        
                        # Draw yellow circle around current target
                        target_circle = Circle((target_wp['x'], target_wp['y']), radius=4, 
                                             fill=False, edgecolor='yellow', linewidth=4, alpha=0.9)
                        self.ax.add_patch(target_circle)
                
                # Draw current camera footprint
                footprint_vertices = self.calculate_camera_footprint(drone['position'])
                footprint_polygon = Polygon(footprint_vertices, fill=True, 
                                          facecolor=drone['color'], alpha=0.2,
                                          edgecolor=drone['color'], linewidth=1)
                self.ax.add_patch(footprint_polygon)
        
        # Draw current mission progress indicators during simulation (on top of waypoints)
        if self.is_running:
            for drone in self.swarm.drones:
                if drone['is_active'] and len(drone['waypoints']) > 1:
                    # Highlight current target waypoint with yellow outline
                    current_wp_idx = min(drone['current_waypoint_index'] + 1, len(drone['waypoints']) - 1)
                    if current_wp_idx < len(drone['waypoints']):
                        target_wp = drone['waypoints'][current_wp_idx]
                        
                        # Draw yellow circle around current target (on top)
                        target_circle = Circle((target_wp['x'], target_wp['y']), radius=5, 
                                             fill=False, edgecolor='yellow', linewidth=4, alpha=0.9, zorder=20)
                        self.ax.add_patch(target_circle)
                    
                    # Draw current camera footprint
                    footprint_vertices = self.calculate_camera_footprint(drone['position'])
                    footprint_polygon = Polygon(footprint_vertices, fill=True, 
                                              facecolor=drone['color'], alpha=0.2,
                                              edgecolor=drone['color'], linewidth=1, zorder=1)
                    self.ax.add_patch(footprint_polygon)
        legend_elements = []
        for drone in self.swarm.drones:
            assigned_count = len(drone['assigned_points'])
            covered_count = len(drone['covered_points'])
            waypoint_count = len(drone['waypoints']) - 1  # Exclude start position
            
            legend_elements.append(plt.Line2D([0], [0], color=drone['color'], 
                                            linewidth=3, 
                                            label=f'Drone {drone["id"]} ({covered_count}/{assigned_count}pts, {waypoint_count}wp)'))
        
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='green', linestyle='None', label='Covered Points'),
            plt.Line2D([0], [0], marker='o', color='gray', linestyle='None', label='Unassigned Points'),
        ])
        
        # Add title with dynamic assignment info including effective grid size
        if self.swarm:
            stats = self.swarm.get_swarm_statistics()
            total_assigned = sum(len(drone['assigned_points']) for drone in self.swarm.drones)
            total_waypoints = sum(len(drone['waypoints']) - 1 for drone in self.swarm.drones)
            coverage_pct = (stats['total_covered_points'] / max(total_assigned, 1)) * 100
            
            # Show effective grid size if different from set parameters
            if hasattr(self, 'effective_grid_x') and hasattr(self, 'effective_grid_y'):
                if self.effective_grid_x != self.grid_points_x or self.effective_grid_y != self.grid_points_y:
                    grid_info = f"Grid: {self.effective_grid_x}x{self.effective_grid_y} (adapted from {self.grid_points_x}x{self.grid_points_y})"
                else:
                    grid_info = f"Grid: {self.grid_points_x}x{self.grid_points_y}"
            else:
                grid_info = f"Grid: {self.grid_points_x}x{self.grid_points_y}"
            
            title = f"Dynamic Assignment - {self.n_drones} Drones\n{grid_info} | Assigned: {total_assigned} | Waypoints: {total_waypoints} | Coverage: {coverage_pct:.1f}%"
        else:
            title = f"Dynamic Drone Swarm System"
        
        self.ax.set_title(title)
        
        try:
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        except:
            pass
        
        # Update canvas
        try:
            self.canvas.draw()
        except:
            pass
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.patches import Rectangle, Circle, Polygon
        import numpy as np
        from scipy.stats import norm
        from scipy.optimize import minimize
        
        print("🚁 Starting Dynamic Region Drone Swarm System...")
        print("\nKey Features:")
        print("- Dynamic region assignment during waypoint generation")
        print("- No predefined regions - drones can explore multiple areas")
        print("- Minimized overlap through intelligent assignment strategies")
        print("- Energy-efficient coordination with overlap penalties")
        print("- Real-time reassignment capabilities")
        
        print("\n🚁 Realistic Starting Formations:")
        print("- Corner: Start from building corners (most common for inspections)")
        print("- Edge: Distribute along building edges")  
        print("- Single Point: All drones from one launch point (small operations)")
        print("- Custom Points: Strategic positions for optimal coverage")
        print("- Opposite Corners: Maximum spread for large buildings")
        print("- Perimeter: Even distribution around building perimeter")
        print("- Center: Single drone at (0,0) or line formation at center")
        print("\n🛤️ Path Optimization Strategies:")
        print("- Smart POV: Camera-centric optimization focusing on coverage efficiency")
        print("- Coverage: Maximum coverage with minimal waypoints")
        print("- Hybrid: Balanced approach considering coverage and travel distance")
        print("- Nearest: Simple nearest neighbor algorithm")
        print("- TSP Optimized: Traveling salesman optimization with 2-opt improvement")
        
        print("\n🧠 Dynamic Assignment Strategies:")
        print("- Dynamic Coverage: Multi-objective optimization with distance/coverage/overlap")
        print("- Competitive Assignment: Auction-based point allocation")
        print("- Cooperative Planning: Workload balancing with efficiency matrix")
        
        print("\n🔧 Advanced Features:")
        print("- Customizable overlap penalty and distance weighting")
        print("- Real-time point reassignment during mission")
        print("- Bayesian optimization for grid and strategy parameters")
        print("- Comprehensive analysis of assignment efficiency and overlap")
        
        print("\nUsage:")
        print("1. Configure drones and assignment parameters")
        print("2. Choose assignment strategy (dynamic_coverage recommended)")
        print("3. Adjust overlap penalty and weights for your scenario")
        print("4. Use '🔄 Reassign Points' to dynamically redistribute")
        print("6. Choose realistic starting formation for your inspection scenario")
        print("7. Monitor assignment efficiency in Analysis tab")
        print("8. Start mission to see coordinated coverage from realistic launch points")
        
        app = LiveDroneSwarmSimulation()
        app.run()
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nTo install required packages, run:")
        print("pip install matplotlib numpy scipy")
