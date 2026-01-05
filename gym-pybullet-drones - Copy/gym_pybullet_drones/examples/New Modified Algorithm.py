
import os
import time
import argparse
import math
import random
import numpy as np
import cv2
import pybullet as p
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Default simulation parameters
DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 250
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# GA Parameters
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
GA_MUTATION_RATE = 0.2
GA_CROSSOVER_RATE = 0.8
GA_ELITE_SIZE = 2

# A* Parameters
GRID_RESOLUTION = 0.5  # Grid cell size for A* pathfinding
DIAGONAL_COST = 1.414  # Cost of diagonal movement (√2)
STRAIGHT_COST = 1.0    # Cost of straight movement

#############################################################
# A* Algorithm Implementation
#############################################################

class Node:
    """A node class for A* pathfinding"""
    def __init__(self, position, parent=None):
        self.position = position  # (x, y) tuple
        self.parent = parent
        
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current to goal)
        self.f = 0  # Total cost (g + h)
        
    def __eq__(self, other):
        return self.position == other.position
        
    def __lt__(self, other):
        return self.f < other.f
        
    def __hash__(self):
        return hash(self.position)

def heuristic(point1, point2):
    """Calculate the Euclidean distance heuristic"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_neighbors(node, grid, allow_diagonal=True):
    """Get neighboring cells of a node in the grid"""
    neighbors = []
    # Define possible movement actions (8-directional)
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
    
    if allow_diagonal:
        actions += [(1, 1), (1, -1), (-1, -1), (-1, 1)]  # Diagonals
    
    for action in actions:
        # Get neighbor position
        neighbor_pos = (node.position[0] + action[0], node.position[1] + action[1])
        
        # Check if within grid bounds
        if (0 <= neighbor_pos[0] < grid.shape[1] and 
            0 <= neighbor_pos[1] < grid.shape[0]):
            
            # Check if obstacle-free (255 is free space)
            if grid[neighbor_pos[1], neighbor_pos[0]] >= 200:  # Threshold for passable
                # Calculate movement cost
                if action in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    move_cost = STRAIGHT_COST
                else:
                    move_cost = DIAGONAL_COST
                    
                # Create new node
                neighbor = Node(neighbor_pos, node)
                neighbor.g = node.g + move_cost
                neighbors.append(neighbor)
    
    return neighbors

def a_star_search(start_pos, goal_pos, grid):
    """
    A* pathfinding algorithm to find an obstacle-avoiding path
    
    Args:
        start_pos: Starting position (x, y) in grid coordinates
        goal_pos: Goal position (x, y) in grid coordinates
        grid: Binary occupancy grid where 0 is obstacle, 255 is free space
        
    Returns:
        List of path waypoints in grid coordinates
    """
    # Convert positions to integer grid coordinates if they aren't already
    start_pos = (int(start_pos[0]), int(start_pos[1]))
    goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
    
    # Create start and end nodes
    start_node = Node(start_pos)
    end_node = Node(goal_pos)
    
    # Initialize open and closed sets
    open_set = []
    closed_set = set()
    
    # Add start node to open set
    start_node.g = start_node.h = start_node.f = 0
    heapq.heappush(open_set, (start_node.f, id(start_node), start_node))
    
    # Loop until find the path or open set is empty
    while open_set:
        # Get node with lowest f value
        _, _, current_node = heapq.heappop(open_set)
        
        # Add to closed set
        closed_set.add(current_node.position)
        
        # Check if goal reached
        if current_node.position == end_node.position:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path
        
        # Generate neighbors
        for neighbor in get_neighbors(current_node, grid):
            # Skip if in closed set
            if neighbor.position in closed_set:
                continue
                
            # Calculate f, g, h values
            neighbor.g = current_node.g + heuristic(current_node.position, neighbor.position)
            neighbor.h = heuristic(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h
            
            # Check if neighbor already in open set with better g score
            already_in_open = False
            for i, (_, _, open_node) in enumerate(open_set):
                if neighbor.position == open_node.position:
                    already_in_open = True
                    if neighbor.g < open_node.g:
                        # Replace with better path
                        open_set[i] = (neighbor.f, id(neighbor), neighbor)
                        heapq.heapify(open_set)
                    break
                    
            # Add to open set if not already there
            if not already_in_open:
                heapq.heappush(open_set, (neighbor.f, id(neighbor), neighbor))
    
    # No path found
    return None

def convert_world_to_grid(point, origin, resolution):
    """Convert world coordinates to grid coordinates"""
    grid_x = int((point[0] - origin[0]) / resolution)
    grid_y = int((point[1] - origin[1]) / resolution)
    return (grid_x, grid_y)

def convert_grid_to_world(grid_point, origin, resolution):
    """Convert grid coordinates to world coordinates"""
    world_x = grid_point[0] * resolution + origin[0]
    world_y = grid_point[1] * resolution + origin[1]
    return (world_x, world_y)

def divide_area(total_size, num_drones):
    """
    Divide a square area into equal sections for multiple drones
    
    Args:
        total_size (float): Size of the square area to divide
        num_drones (int): Number of drones
        
    Returns:
        list: List of dictionaries with region parameters for each drone
    """
    regions = []
    
    # For 4 drones, divide into quadrants
    if num_drones == 4:
        half_size = total_size / 2
        quarter_size = total_size / 4
        
        # Define the four quadrants (top-left, top-right, bottom-left, bottom-right)
        regions = [
            {
                'center': [-quarter_size, quarter_size],  # Top-left quadrant
                'size': half_size,
                'offset': [-half_size, quarter_size]
            },
            {
                'center': [quarter_size, quarter_size],   # Top-right quadrant
                'size': half_size,
                'offset': [quarter_size, quarter_size]
            },
            {
                'center': [-quarter_size, -quarter_size], # Bottom-left quadrant
                'size': half_size,
                'offset': [-half_size, -half_size]
            },
            {
                'center': [quarter_size, -quarter_size],  # Bottom-right quadrant
                'size': half_size,
                'offset': [quarter_size, -half_size]
            }
        ]
    
    # For 2 drones, divide into left and right halves
    elif num_drones == 2:
        half_size = total_size / 2
        
        regions = [
            {
                'center': [-quarter_size, 0],  # Left half
                'size': half_size,
                'offset': [-half_size, -half_size/2]
            },
            {
                'center': [quarter_size, 0],   # Right half
                'size': half_size,
                'offset': [0, -half_size/2]
            }
        ]
    
    # For other numbers of drones (1, 3, etc.), use a grid-based approach
    else:
        # Calculate a grid size based on the number of drones
        grid_size = int(np.ceil(np.sqrt(num_drones)))
        cell_size = total_size / grid_size
        half_cell = cell_size / 2
        
        # Create a grid of positions
        for i in range(min(num_drones, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            
            regions.append({
                'center': [col * cell_size + half_cell - total_size/2, 
                          row * cell_size + half_cell - total_size/2],
                'size': cell_size,
                'offset': [col * cell_size - total_size/2, 
                          row * cell_size - total_size/2]
            })
    
    return regions



def create_obstacles(pyb_client, waypoints, num_obstacles=10, size=0.01):
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

def create_grid_from_coverage_map(coverage_map, obstacles=None):
    """
    Create a grid for A* from the coverage map and obstacles
    
    Args:
        coverage_map: The coverage map
        obstacles: List of obstacle positions in world coordinates
        
    Returns:
        Grid suitable for A* navigation (255 for free, 0 for obstacles)
    """
    # Start with a copy of the coverage map
    grid = coverage_map.copy()
    
    # Mark all cells as free (if not already)
    grid[grid < 100] = 200  # Set uncovered areas to 200 (passable but uncovered)
    grid[grid >= 100] = 255  # Set covered areas to 255 (passable and covered)
    
    # Add obstacles to the grid if provided
    if obstacles:
        for obstacle in obstacles:
            # Convert obstacle position to grid coordinates
            x, y = convert_world_to_grid(obstacle[:2], [-7.5, -7.5], GRID_RESOLUTION)
            
            # Create obstacle zone (safety margin around obstacle)
            radius = int(0.3 / GRID_RESOLUTION)  # 0.3m safety margin
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:  # Circular obstacle
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                            grid[ny, nx] = 0  # Mark as obstacle
    
    return grid

def find_path_with_astar(start_pos, goal_pos, coverage_map, obstacles, origin=[-7.5, -7.5], resolution=GRID_RESOLUTION):
    """
    Find a path from start to goal using A* that avoids obstacles
    
    Args:
        start_pos: Starting position [x, y, z] in world coordinates
        goal_pos: Goal position [x, y, z] in world coordinates
        coverage_map: Coverage map used to identify free/covered areas
        obstacles: List of obstacle positions
        origin: Origin of the coverage map in world coordinates
        resolution: Grid resolution
        
    Returns:
        List of waypoints in world coordinates [x, y, z]
    """
    # Create grid for A*
    grid = create_grid_from_coverage_map(coverage_map, obstacles)
    
    # Convert world positions to grid coordinates
    start_grid = convert_world_to_grid(start_pos[:2], origin, resolution)
    goal_grid = convert_world_to_grid(goal_pos[:2], origin, resolution)
    
    # Check if start or goal is in obstacle
    if (0 <= start_grid[0] < grid.shape[1] and 0 <= start_grid[1] < grid.shape[0] and 
        0 <= goal_grid[0] < grid.shape[1] and 0 <= goal_grid[1] < grid.shape[0]):
        if grid[start_grid[1], start_grid[0]] == 0 or grid[goal_grid[1], goal_grid[0]] == 0:
            # Start or goal in obstacle - no safe path possible
            return None
    else:
        # Start or goal outside grid bounds
        return None
    
    # Run A* search
    path_grid = a_star_search(start_grid, goal_grid, grid)
    
    if path_grid is None:
        return None
    
    # Convert path to world coordinates with fixed altitude
    altitude = start_pos[2]
    path_world = []
    for point in path_grid:
        world_x, world_y = convert_grid_to_world(point, origin, resolution)
        path_world.append([world_x, world_y, altitude])
    
    return np.array(path_world)

#############################################################
# Genetic Algorithm Implementation
#############################################################

class GACoveragePlanner:
    """Genetic Algorithm for optimizing coverage paths"""
    
    def __init__(self, area_size, num_drones, grid_resolution, region_division=True):
        """
        Initialize GA planner
        
        Args:
            area_size: Size of the square area to cover
            num_drones: Number of drones
            grid_resolution: Resolution of the environment grid
            region_division: Whether to divide area into regions
        """
        self.area_size = area_size
        self.num_drones = num_drones
        self.grid_resolution = grid_resolution
        self.region_division = region_division
        
        # Define grid dimensions
        self.grid_cells = int(area_size / grid_resolution)
        
        # Get region divisions
        if region_division:
            self.regions = divide_area(area_size, num_drones)
        else:
            self.regions = None
    
    def generate_individual(self):
        """
        Generate a single individual (candidate solution)
        
        Returns:
            List of waypoint sets for each drone
        """
        if self.region_division:
            # Generate waypoints for each drone in its own region
            individual = []
            for i in range(self.num_drones):
                waypoints = self._generate_region_waypoints(self.regions[i])
                individual.append(waypoints)
        else:
            # Generate waypoints for each drone across the whole area
            individual = []
            for i in range(self.num_drones):
                waypoints = self._generate_random_waypoints(10)  # 10 waypoints per drone
                individual.append(waypoints)
        
        return individual
    
    def _generate_region_waypoints(self, region, num_points=8):
        """Generate waypoints for a specific region"""
        offset_x, offset_y = region['offset']
        size = region['size']
        
        # Define region bounds
        min_x = offset_x
        max_x = offset_x + size
        min_y = offset_y
        max_y = offset_y + size
        
        # Generate random waypoints within region
        waypoints = []
        for _ in range(num_points):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            z = 1.5  # Fixed altitude
            waypoints.append([x, y, z])
        
        return np.array(waypoints)
    
    def _generate_random_waypoints(self, num_points):
        """Generate random waypoints across the entire area"""
        half_size = self.area_size / 2
        
        waypoints = []
        for _ in range(num_points):
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            z = 1.5  # Fixed altitude
            waypoints.append([x, y, z])
        
        return np.array(waypoints)
    
    def initialize_population(self, size=GA_POPULATION_SIZE):
        """Initialize a population of candidate solutions"""
        return [self.generate_individual() for _ in range(size)]
    
    def evaluate_fitness(self, individual, coverage_map, obstacles=None):
        """
        Evaluate the fitness of an individual
        
        Args:
            individual: List of waypoint sets for each drone
            coverage_map: Current coverage map
            obstacles: List of obstacle positions
            
        Returns:
            Fitness score (higher is better)
        """
        # Create a copy of the coverage map for simulation
        simulation_map = coverage_map.copy()
        
        # Create a grid for A* pathfinding
        grid = create_grid_from_coverage_map(coverage_map, obstacles)
        
        # Simulate coverage for all drones
        total_path_length = 0
        total_coverage_gain = 0
        
        for drone_waypoints in individual:
            # Calculate coverage gain and path length for this drone
            path_length, coverage_gain = self._simulate_drone_coverage(
                drone_waypoints, simulation_map, grid, obstacles
            )
            
            total_path_length += path_length
            total_coverage_gain += coverage_gain
        
        # Calculate fitness (balance between coverage gain and path efficiency)
        if total_path_length == 0:
            return 0  # Invalid path
        
        # Fitness is coverage gain divided by path length (efficiency)
        # We multiply by 1000 to get more meaningful numbers
        fitness = (total_coverage_gain / total_path_length) * 1000
        
        return fitness
    
    def _simulate_drone_coverage(self, waypoints, coverage_map, grid, obstacles):
        """
        Simulate a drone following waypoints and calculate coverage metrics
        
        Args:
            waypoints: Array of waypoints for one drone
            coverage_map: Coverage map to update
            grid: Grid for A* pathfinding
            obstacles: List of obstacle positions
            
        Returns:
            (path_length, coverage_gain) tuple
        """
        if len(waypoints) < 2:
            return 0, 0
        
        # Track path length and coverage gain
        path_length = 0
        initial_coverage = np.sum(coverage_map > 0)
        
        # Calculate footprint dimensions
        altitude = waypoints[0][2]
        footprint_size = int(altitude * 0.3 / 0.1)  # Simple model: coverage width = 0.3 * altitude
        
        # Simulate drone moving through waypoints
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i+1]
            
            # Add Euclidean distance to path length
            dist = np.linalg.norm(start[:2] - end[:2])
            path_length += dist
            
            # Update coverage map along the path
            direction = end[:2] - start[:2]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
                # Discretize the path
                step_size = 0.5  # meters
                steps = int(dist / step_size) + 1
                
                for step in range(steps):
                    t = step / steps if steps > 1 else 0
                    pos = start[:2] + t * (end[:2] - start[:2])
                    
                    # Convert to grid coordinates
                    grid_x = int((pos[0] + self.area_size/2) / 0.1)
                    grid_y = int((pos[1] + self.area_size/2) / 0.1)
                    
                    # Update coverage with drone footprint
                    x_min = max(0, grid_x - footprint_size//2)
                    x_max = min(coverage_map.shape[1]-1, grid_x + footprint_size//2)
                    y_min = max(0, grid_y - footprint_size//2)
                    y_max = min(coverage_map.shape[0]-1, grid_y + footprint_size//2)
                    
                    coverage_map[y_min:y_max+1, x_min:x_max+1] = 255
        
        # Calculate total coverage gain
        final_coverage = np.sum(coverage_map > 0)
        coverage_gain = final_coverage - initial_coverage
        
        return path_length, coverage_gain
    
    def select_parents(self, population, fitness_scores, num_parents=2):
        """
        Select parents for crossover using tournament selection
        
        Args:
            population: Current population
            fitness_scores: List of fitness scores for the population
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent individuals
        """
        tournament_size = 3
        selected_parents = []
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner (highest fitness)
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx])
        
        return selected_parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Args:
            parent1, parent2: Parent individuals
            
        Returns:
            Two child individuals
        """
        if random.random() > GA_CROSSOVER_RATE:
            return parent1, parent2  # No crossover
        
        child1 = []
        child2 = []
        
        # Perform crossover for each drone's waypoints
        for i in range(self.num_drones):
            if random.random() < 0.5:  # 50% chance to swap
                child1.append(parent2[i].copy())
                child2.append(parent1[i].copy())
            else:
                child1.append(parent1[i].copy())
                child2.append(parent2[i].copy())
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate an individual
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        if random.random() > GA_MUTATION_RATE:
            return individual  # No mutation
        
        # Select a random drone to mutate
        drone_idx = random.randint(0, self.num_drones - 1)
        waypoints = individual[drone_idx]
        
        if len(waypoints) == 0:
            return individual  # Can't mutate empty waypoints
        
        # Select random mutation type
        mutation_type = random.choice(['add', 'remove', 'modify'])
        
        if mutation_type == 'add' and len(waypoints) < 20:
            # Add a new waypoint
            if self.region_division:
                region = self.regions[drone_idx]
                offset_x, offset_y = region['offset']
                size = region['size']
                
                # Generate within region
                x = random.uniform(offset_x, offset_x + size)
                y = random.uniform(offset_y, offset_y + size)
            else:
                # Generate anywhere
                half_size = self.area_size / 2
                x = random.uniform(-half_size, half_size)
                y = random.uniform(-half_size, half_size)
                
            z = waypoints[0][2]  # Use same altitude
            
            # Insert at random position
            insert_pos = random.randint(0, len(waypoints))
            new_waypoints = np.vstack((waypoints[:insert_pos], 
                                      [[x, y, z]], 
                                      waypoints[insert_pos:]))
            individual[drone_idx] = new_waypoints
            
        elif mutation_type == 'remove' and len(waypoints) > 2:
            # Remove a random waypoint
            remove_idx = random.randint(0, len(waypoints) - 1)
            individual[drone_idx] = np.delete(waypoints, remove_idx, axis=0)
            
        elif mutation_type == 'modify':
            # Modify a random waypoint
            modify_idx = random.randint(0, len(waypoints) - 1)
            
            # Small perturbation
            dx = random.uniform(-1.5, 1.5)
            dy = random.uniform(-1.5, 1.5)
            
            # Apply perturbation with bounds checking
            if self.region_division:
                region = self.regions[drone_idx]
                offset_x, offset_y = region['offset']
                size = region['size']
                
                x = max(offset_x, min(offset_x + size, waypoints[modify_idx][0] + dx))
                y = max(offset_y, min(offset_y + size, waypoints[modify_idx][1] + dy))
            else:
                half_size = self.area_size / 2
                x = max(-half_size, min(half_size, waypoints[modify_idx][0] + dx))
                y = max(-half_size, min(half_size, waypoints[modify_idx][1] + dy))
                
            individual[drone_idx][modify_idx][0] = x
            individual[drone_idx][modify_idx][1] = y
        
        return individual
    
    def evolve(self, population, coverage_map, obstacles=None, num_generations=GA_GENERATIONS):
        """
        Evolve the population to find optimal coverage paths
        
        Args:
            population: Initial population
            coverage_map: Current coverage map
            obstacles: List of obstacle positions
            num_generations: Number of generations to evolve
            
        Returns:
            (best_individual, best_fitness) tuple
        """
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(num_generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self.evaluate_fitness(ind, coverage_map, obstacles) 
                              for ind in population]
            
            # Find best individual
            current_best_idx = np.argmax(fitness_scores)
            current_best = population[current_best_idx]
            current_best_fitness = fitness_scores[current_best_idx]
            
            # Update overall best
            if current_best_fitness > best_fitness:
                best_individual = current_best
                best_fitness = current_best_fitness
                
            print(f"Generation {generation+1}/{num_generations}, Best Fitness: {best_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individuals
            sorted_idx = np.argsort(fitness_scores)[::-1]
            for i in range(GA_ELITE_SIZE):
                new_population.append(population[sorted_idx[i]])
            
            # Fill the rest through selection, crossover, and mutation
            while len(new_population) < len(population):
                # Select parents
                parents = self.select_parents(population, fitness_scores)
                
                # Crossover
                child1, child2 = self.crossover(parents[0], parents[1])
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < len(population):
                    new_population.append(child2)
            
            # Replace old population
            population = new_population
        
        return best_individual, best_fitness

#############################################################
# Helper Classes
#############################################################

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

class PotentialFieldController:
    def __init__(self, drone_pos, waypoint, obstacles, max_speed=2.0, max_force=1.0):
        self.max_speed = max_speed
        self.max_force = max_force
        self.attraction_gain = 1.0
        self.repulsion_gain = 100.0
        self.repulsion_range = 0.3
        self.waypoint_threshold = 1
        self.slowdown_radius = 1.0
        
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
    def __init__(self, area_size=14.5, resolution=0.1, 
                 focal_length=4e-3, sensor_width=6.17e-3, sensor_height=4.55e-3):
        """
        Args:
            area_size: Size of the area to track
            resolution: Grid resolution
            focal_length: Camera focal length (m)
            sensor_width/height: Physical sensor dimensions (m)
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
        """Update coverage map with drone positions"""
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
        """Calculate percentage of area covered"""
        covered = np.sum(self.coverage_map > 0)
        return (covered / self.coverage_map.size) * 100
        
    def visualize(self, window_name="Coverage Map", scale=1, positions=None, altitude=None):
        """Visualize the coverage map and drone positions"""
        rotated_map = cv2.rotate(self.coverage_map.copy(), cv2.ROTATE_180)
        flip_map = cv2.flip(rotated_map, 1)  # Flip vertically
        display_img = cv2.resize(flip_map, 
                               (self.grid_size*scale, self.grid_size*scale),
                               interpolation=cv2.INTER_NEAREST)
        
        # Convert to color image if adding drone rectangles
        if positions is not None and altitude is not None:
            # Convert to color image if it's grayscale
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                
            # Calculate camera footprint
            ground_w, ground_h = self.calculate_footprint(altitude)
            pixel_w = int(ground_w / self.resolution) * scale
            pixel_h = int(ground_h / self.resolution) * scale
            
            # Draw rectangles for each drone's footprint
            for pos in positions:
                # Convert world coordinates to pixel coordinates
                x_center = int((pos[0] + self.area_offset) / self.resolution) * scale
                y_center = int((pos[1] + self.area_offset) / self.resolution) * scale
                
                # Calculate rectangle corners
                x1 = max(0, x_center - pixel_w//2)
                y1 = max(0, y_center - pixel_h//2)
                x2 = min(display_img.shape[1]-1, x_center + pixel_w//2)
                y2 = min(display_img.shape[0]-1, y_center + pixel_h//2)
                
                # Draw rectangle
                cv2.rectangle(display_img, 
                            (x1, y1), 
                            (x2, y2),
                            (0, 255, 0), 
                            1)  # Green rectangle with thickness 1
                
        cv2.imshow(window_name, display_img)


def get_nadir_camera_image(positions, drone_quat, fov=60, resolution=(320, 240), near=0.1, far=100):
    """
    Get a simulated camera image from drone's nadir view
    
    Args:
        positions: Drone position [x, y, z]
        drone_quat: Drone orientation quaternion
        fov: Field of view in degrees
        resolution: Image resolution
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        RGB image array
    """
    width, height = resolution
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=positions,
        distance=0.001,   # close to drone
        yaw=0,
        pitch=-90,        # look directly downward
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
    """
    Detect cracks in an image
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Modified image with detected cracks highlighted
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] != 3:
        return image  # Not a valid image format

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define threshold for dark areas (potential cracks)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    # Create mask for dark areas
    mask_black = cv2.inRange(image, lower_black, upper_black)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel)

    # Find contours in mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    # Highlight and label cracks
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "Crack", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output

def create_cracks_on_floor(pyb_client, floor_position, floor_size, num_cracks=5):
    """
    Create visual cracks on the floor for simulated inspection
    
    Args:
        pyb_client: PyBullet physics client ID
        floor_position: Position of the floor [x, y, z]
        floor_size: Size of the floor [width, length]
        num_cracks: Number of cracks to create
        
    Returns:
        List of crack visual IDs
    """
    crack_ids = []
    
    # Extract floor parameters
    width, length = floor_size
    half_width = width / 2
    half_length = length / 2
    
    for _ in range(num_cracks):
        # Random position on floor
        x_pos = random.uniform(-half_width + 1, half_width - 1) + floor_position[0]
        y_pos = random.uniform(-half_length + 1, half_length - 1) + floor_position[1]
        
        # Random size and orientation for crack
        crack_length = random.uniform(0.5, 2.0)
        crack_width = random.uniform(0.05, 0.15)
        rotation = random.uniform(0, 2*math.pi)
        
        # Create crack visual (no collision, just visual)
        crack_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[crack_length/2, crack_width/2, 0.01],
            rgbaColor=[0.1, 0.1, 0.1, 1],  # Dark color for cracks
            physicsClientId=pyb_client
        )
        
        # Create crack with rotation
        q = p.getQuaternionFromEuler([0, 0, rotation])
        crack_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=crack_visual,
            basePosition=[x_pos, y_pos, floor_position[2] + 0.01],  # Slightly above floor
            baseOrientation=q,
            physicsClientId=pyb_client
        )
        
        crack_ids.append(crack_body)
    
    return crack_ids

def create_simplified_building(pyb_client, position=[0, 0, 0], size=[7.5, 7.5, 6], wall_thickness=0.2):
    """
    Create a simplified building structure using primitive shapes for better performance
    
    Args:
        pyb_client: PyBullet physics client ID
        position: Center position of the building [x, y, z]
        size: Size of the building [width, length, height]
        wall_thickness: Thickness of the walls
        
    Returns:
        List of building part IDs
    """
    building_ids = []
    
    # Extract building parameters
    width, length, height = size
    half_width = width / 2
    half_length = length / 2
    
    # Define wall parameters
    wall_color = [0.8, 0.8, 0.8, 1]  # Light gray
    floor_color = [0.9, 0.9, 0.9, 1]  # Almost white
    
    # Create floor
    floor_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_width, half_length, wall_thickness/2],
        physicsClientId=pyb_client
    )
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_width, half_length, wall_thickness/2],
        rgbaColor=floor_color,
        physicsClientId=pyb_client
    )
    floor_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[position[0], position[1], position[2]],
        physicsClientId=pyb_client
    )
    building_ids.append(floor_body)
    
    # Create walls
    # Wall 1 (North)
    wall_north = create_wall(
        pyb_client,
        [position[0], position[1] + half_length - wall_thickness/2, position[2] + height/2],
        [width, wall_thickness, height],
        wall_color
    )
    building_ids.append(wall_north)
    
    # Wall 2 (South)
    wall_south = create_wall(
        pyb_client,
        [position[0], position[1] - half_length + wall_thickness/2, position[2] + height/2],
        [width, wall_thickness, height],
        wall_color
    )
    building_ids.append(wall_south)
    
    # Wall 3 (East)
    wall_east = create_wall(
        pyb_client,
        [position[0] + half_width - wall_thickness/2, position[1], position[2] + height/2],
        [wall_thickness, length, height],
        wall_color
    )
    building_ids.append(wall_east)
    
    # Wall 4 (West)
    wall_west = create_wall(
        pyb_client,
        [position[0] - half_width + wall_thickness/2, position[1], position[2] + height/2],
        [wall_thickness, length, height],
        wall_color
    )
    building_ids.append(wall_west)
    
    # Add some random internal walls for obstacles
    num_internal_walls = 4
    for _ in range(num_internal_walls):
        # Random position within building
        x_pos = random.uniform(-half_width + 2, half_width - 2) + position[0]
        y_pos = random.uniform(-half_length + 2, half_length - 2) + position[1]
        
        # Random orientation (0 = north-south, 1 = east-west)
        orientation = random.randint(0, 1)
        
        # Random length (but smaller than building)
        if orientation == 0:  # north-south wall
            wall_size = [wall_thickness, random.uniform(2, 5), height * 0.8]
        else:  # east-west wall
            wall_size = [random.uniform(2, 5), wall_thickness, height * 0.8]
        
        # Create internal wall
        internal_wall = create_wall(
            pyb_client,
            [x_pos, y_pos, position[2] + wall_size[2]/2],
            wall_size,
            [0.7, 0.7, 0.7, 1]  # Slightly darker
        )
        building_ids.append(internal_wall)
    
    return building_ids

def create_wall(pyb_client, position, size, color):
    """Helper function to create a wall"""
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[size[0]/2, size[1]/2, size[2]/2],
        physicsClientId=pyb_client
    )
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size[0]/2, size[1]/2, size[2]/2],
        rgbaColor=color,
        physicsClientId=pyb_client
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=pyb_client
    )
    return body

def add_building_to_simulation(pyb_client, building_size=7.5):
    """
    Add a building to the simulation with simplified geometry for better performance
    
    Args:
        pyb_client: PyBullet physics client ID
        building_size: Size of the building
        
    Returns:
        List of building object IDs
    """
    # Create building centered at origin with 6m height
    building_ids = create_simplified_building(
        pyb_client,
        position=[0, 0, 0],
        size=[building_size, building_size, 6],
        wall_thickness=0.2
    )
    
    # Add cracks on floor for inspection tasks
    crack_ids = create_cracks_on_floor(
        pyb_client,
        floor_position=[0, 0, 0.1],  # Slightly above ground
        floor_size=[building_size, building_size],
        num_cracks=8
    )
    
    return building_ids + crack_ids

def optimize_paths_with_ga(num_drones, area_size, regions, coverage_map, obstacles, altitude=1.5):
    """
    Optimize coverage paths using Genetic Algorithm
    
    Args:
        num_drones: Number of drones
        area_size: Size of the area to cover
        regions: List of region dictionaries
        coverage_map: Current coverage map
        obstacles: List of obstacle positions
        altitude: Flight altitude
        
    Returns:
        List of optimized waypoint paths for each drone
    """
    # Initialize GA planner
    ga_planner = GACoveragePlanner(
        area_size=area_size,
        num_drones=num_drones,
        grid_resolution=GRID_RESOLUTION,
        region_division=True
    )
    
    # Initialize population
    population = ga_planner.initialize_population(size=GA_POPULATION_SIZE)
    
    # Evolve population
    best_individual, best_fitness = ga_planner.evolve(
        population=population,
        coverage_map=coverage_map,
        obstacles=obstacles,
        num_generations=GA_GENERATIONS
    )
    
    # Convert optimized waypoints to A* paths
    optimized_paths = []
    
    for i in range(num_drones):
        region_waypoints = best_individual[i]
        if len(region_waypoints) < 2:
            # If not enough waypoints, generate a simple path within region
            center_x, center_y = regions[i]['center']
            path = np.array([[center_x, center_y, altitude]])
            optimized_paths.append(path)
            continue
            
        # Create a detailed path through the waypoints using A*
        detailed_path = []
        
        # First waypoint
        detailed_path.append(region_waypoints[0])
        
        # Connect subsequent waypoints with A*
        for j in range(1, len(region_waypoints)):
            start = region_waypoints[j-1]
            goal = region_waypoints[j]
            
            # Find A* path between waypoints
            astar_path = find_path_with_astar(
                start_pos=start,
                goal_pos=goal,
                coverage_map=coverage_map,
                obstacles=obstacles,
                origin=[-area_size/2, -area_size/2],
                resolution=GRID_RESOLUTION
            )
            
            # If A* fails, use direct path
            if astar_path is None or len(astar_path) == 0:
                detailed_path.append(goal)
            else:
                # Skip first point as it duplicates the previous endpoint
                if len(astar_path) > 1:
                    detailed_path.extend(astar_path[1:])
                else:
                    detailed_path.append(astar_path[0])
        
        optimized_paths.append(np.array(detailed_path))
    
    return optimized_paths

#############################################################
# Main Simulation Function
#############################################################

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
        use_building=True
        ):
    """
    Main simulation function
    
    Args:
        drone: Drone model to use
        gui: Whether to use PyBullet GUI
        record_video: Whether to record video
        plot: Whether to plot results
        user_debug_gui: Whether to use debug GUI
        obstacles: Whether to add obstacles
        simulation_freq_hz: Simulation frequency in Hz
        control_freq_hz: Control frequency in Hz
        duration_sec: Duration of simulation in seconds
        output_folder: Folder to save output
        colab: Whether running in Colab
        use_building: Whether to include a building for inspection
    """
    # Simulation parameters
    NUM_DRONES = 4
    BUILDING_SIZE = 15
    ALTITUDE = 1.5
    COVERAGE_ORIGIN = [-BUILDING_SIZE/2, -BUILDING_SIZE/2]
    NUM_OBSTACLES = 8

    # Divide the area into regions for drones
    regions = divide_area(BUILDING_SIZE, NUM_DRONES)

    # Set different starting positions for each drone in their respective regions
    INIT_XYZS = np.array([
        [regions[i]['center'][0], regions[i]['center'][1], 0.5] for i in range(NUM_DRONES)
    ])

    # Create the environment
    env = VelocityAviary(
        drone_model=drone,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=False,  # We'll add our own obstacles
        user_debug_gui=user_debug_gui
    )

    # Get PyBullet client and drone IDs
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    # Define drone marker colors
    DRONE_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    # Configure PyBullet GUI camera
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=0,
            cameraPitch=-89.9,  # Angled view to see building better
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=PYB_CLIENT
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        
        # Higher quality rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # Create logger
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=NUM_DRONES,
        output_folder=output_folder,
        colab=colab
    )

    # Add building if enabled
    building_ids = []
    if use_building:
        print("Creating building structure...")
        building_ids = add_building_to_simulation(PYB_CLIENT, building_size=BUILDING_SIZE)
        
        # Prevent obstacles from being placed inside building
        obstacles = False
        
        # Adjust altitude to be inside building
        ALTITUDE = 1.5

    # Create obstacles if building not used
    obstacle_ids, obstacle_positions = [], []
    if obstacles and not use_building:
        obstacle_ids, obstacle_positions = create_obstacles(
            PYB_CLIENT, 
            num_obstacles=NUM_OBSTACLES,
            area_size=BUILDING_SIZE
        )
    elif use_building:
        # If building is used, use wall positions as obstacles for path planning
        # This is a simplified representation of the building for the algorithms
        for building_part in building_ids:
            pos, _ = p.getBasePositionAndOrientation(building_part, physicsClientId=PYB_CLIENT)
            if pos[2] > 0.5:  # Only include walls, not floor
                obstacle_positions.append(pos)

    # Initialize coverage tracker
    coverage = NadirCoverageTracker(
        area_size=BUILDING_SIZE,
        resolution=0.1,
        focal_length=4.3e-3,
        sensor_width=6.17e-3,
        sensor_height=4.55e-3
    )
    
    # Wait for a few steps to initialize the simulation properly
    for _ in range(10):
        action = np.zeros((NUM_DRONES, 4))
        env.step(action)
    
    print("Initializing coverage paths with GA and A*...")
    
    # Get current drone positions
    obs, _, _, _, _ = env.step(np.zeros((NUM_DRONES, 4)))
    positions = [obs[j][0:3] for j in range(NUM_DRONES)]
    
    # Optimize paths using GA and A*
    optimized_paths = optimize_paths_with_ga(
        num_drones=NUM_DRONES,
        area_size=BUILDING_SIZE,
        regions=regions,
        coverage_map=coverage.coverage_map,
        obstacles=obstacle_positions,
        altitude=ALTITUDE
    )
    
    print("Path optimization complete!")
    
    # Initialize controllers for each drone
    pf_controllers = [
        PotentialFieldController(
            positions[i],
            optimized_paths[i][0] if len(optimized_paths[i]) > 0 else positions[i],
            obstacle_positions,
            max_speed=1.2,  # Reduced speed for indoor navigation
            max_force=0.5
        ) for i in range(NUM_DRONES)
    ]
    
    # Initialize waypoint counters and current waypoints
    wp_counters = [0 for _ in range(NUM_DRONES)]
    current_waypoints = [
        optimized_paths[i][0] if len(optimized_paths[i]) > 0 else positions[i] 
        for i in range(NUM_DRONES)
    ]
    
    # Initialize trails for visualization
    drone_trails = [[] for _ in range(NUM_DRONES)]
    TRAIL_LENGTH = 20  # Trail length for visualization
    
    # Draw the optimized paths
    if gui:
        for i in range(NUM_DRONES):
            path = optimized_paths[i]
            if len(path) > 1:
                for j in range(len(path) - 1):
                    p.addUserDebugLine(
                        path[j],
                        path[j+1],
                        lineColorRGB=DRONE_COLORS[i],
                        lineWidth=2,
                        lifeTime=0,
                        physicsClientId=PYB_CLIENT
                    )
    
    # Main simulation loop
    action = np.zeros((NUM_DRONES, 4))
    START = time.time()
    
    # Create windows for drone camera feeds
    if gui:
        for i in range(NUM_DRONES):
            cv2.namedWindow(f"Drone {i} Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Drone {i} Camera", 320, 240)
    
    # Initialize detected cracks positions
    detected_cracks = []
    
    try:
        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            positions = [obs[j][0:3] for j in range(NUM_DRONES)]
            
            # Add custom drone markers
            if gui:
                add_drone_markers(PYB_CLIENT, positions, DRONE_COLORS, marker_size=0.4)
                
                # Update drone trails
                if i % 5 == 0:
                    for j in range(NUM_DRONES):
                        # Add current position to trail
                        drone_trails[j].append(positions[j])
                        # Limit trail length
                        if len(drone_trails[j]) > TRAIL_LENGTH:
                            drone_trails[j].pop(0)
                        
                        # Draw trail
                        if len(drone_trails[j]) > 1:
                            for t in range(len(drone_trails[j]) - 1):
                                p.addUserDebugLine(
                                    drone_trails[j][t],
                                    drone_trails[j][t+1],
                                    lineColorRGB=DRONE_COLORS[j],
                                    lineWidth=1,
                                    lifeTime=0.1,
                                    physicsClientId=PYB_CLIENT
                                )
            
            # Update coverage map
            if i % 5 == 0:
                coverage.update(positions, ALTITUDE)
                
                # Get drone camera images and detect cracks
                if gui and i % 20 == 0:
                    for j in range(NUM_DRONES):
                        # Get camera image
                        cam_img = get_nadir_camera_image(
                            positions[j], 
                            None,  # No quaternion needed
                            fov=60, 
                            resolution=(320, 240)
                        )
                        
                        # Convert to BGR for OpenCV processing
                        img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                        
                        # Detect cracks
                        output_img = detect_crack(img_bgr)
                        
                        # Display the image
                        cv2.imshow(f"Drone {j} Camera", output_img)
                        cv2.waitKey(1)
                
                # Visualize coverage map
                if i % 50 == 0:
                    coverage.visualize(positions=positions, altitude=ALTITUDE)
                    coverage_percent = coverage.get_coverage_percentage()
                    print(f"Coverage: {coverage_percent:.2f}%")
                    
                    # If coverage is almost complete, end simulation
                    if coverage_percent > 95.0:  # Reduced threshold for building scenario
                        print("Coverage goal achieved! Ending simulation.")
                        break
            
            # Calculate simulation time
            elapsed_time = time.time() - START
            
            # Update each drone's path following
            for j in range(NUM_DRONES):
                current_pos = positions[j]
                
                # Check if we've reached the current waypoint
                dist_to_waypoint = np.linalg.norm(current_pos[:2] - current_waypoints[j][:2])
                
                if dist_to_waypoint < 0.3:  # Threshold for reaching waypoint
                    wp_counters[j] += 1
                    
                    # Check if we've reached the end of the path
                    if wp_counters[j] < len(optimized_paths[j]):
                        current_waypoints[j] = optimized_paths[j][wp_counters[j]]
                        print(f"Drone {j} reached waypoint {wp_counters[j]-1}, moving to next")
                    else:
                        # Path completed, hover at last position
                        current_waypoints[j] = current_pos.copy()
                        current_waypoints[j][2] = ALTITUDE  # Maintain altitude
                        print(f"Drone {j} completed all waypoints")
                
                # Obstacle detection using ray casting for more accurate collision avoidance
                sensor_range = 1.5  # Detection range in meters
                num_rays = 8  # Number of rays to cast
                ray_detected_obstacle = False
                
                for ray_idx in range(num_rays):
                    # Calculate ray direction
                    angle = 2 * math.pi * ray_idx / num_rays
                    ray_dir = [math.cos(angle), math.sin(angle), 0]
                    ray_end = [
                        current_pos[0] + ray_dir[0] * sensor_range,
                        current_pos[1] + ray_dir[1] * sensor_range,
                        current_pos[2] + ray_dir[2] * sensor_range
                    ]
                    
                    # Cast ray
                    ray_result = p.rayTest(
                        current_pos,
                        ray_end,
                        physicsClientId=PYB_CLIENT
                    )
                    
                    # If ray hit something, adjust waypoint
                    if ray_result[0][0] != -1:
                        hit_fraction = ray_result[0][2]
                        hit_distance = hit_fraction * sensor_range
                        
                        if hit_distance < 0.5:  # Close obstacle
                            ray_detected_obstacle = True
                            
                            # Compute avoidance direction (away from obstacle)
                            avoidance_dir = [-ray_dir[0], -ray_dir[1], 0]
                            
                            # Temporary waypoint for obstacle avoidance
                            temp_waypoint = [
                                current_pos[0] + avoidance_dir[0] * 0.5,
                                current_pos[1] + avoidance_dir[1] * 0.5,
                                ALTITUDE
                            ]
                            
                            # Use temporary waypoint
                            current_waypoints[j] = temp_waypoint
                            
                            # Draw debug ray
                            if gui:
                                hit_pos = [
                                    current_pos[0] + ray_dir[0] * hit_distance,
                                    current_pos[1] + ray_dir[1] * hit_distance,
                                    current_pos[2] + ray_dir[2] * hit_distance
                                ]
                                p.addUserDebugLine(
                                    current_pos,
                                    hit_pos,
                                    lineColorRGB=[1, 0, 0],  # Red for hits
                                    lineWidth=2,
                                    lifeTime=0.1,
                                    physicsClientId=PYB_CLIENT
                                )
                            
                            break
                
                # Check if path is still valid (periodically)
                if i % 100 == 0 and not ray_detected_obstacle and wp_counters[j] < len(optimized_paths[j]) - 1:
                    # Check if we can see the next waypoint
                    next_wp = optimized_paths[j][wp_counters[j] + 1]
                    ray_result = p.rayTest(
                        current_pos,
                        next_wp,
                        physicsClientId=PYB_CLIENT
                    )
                    
                    # If path is blocked by an obstacle, replan
                    if ray_result[0][0] != -1:
                        print(f"Drone {j} path is blocked, replanning...")
                        
                        # Use A* to find a new path to the next waypoint
                        new_path = find_path_with_astar(
                            current_pos,
                            next_wp,
                            coverage.coverage_map,
                            obstacle_positions,
                            origin=COVERAGE_ORIGIN,
                            resolution=GRID_RESOLUTION
                        )
                        
                        if new_path is not None and len(new_path) > 1:
                            # Replace path segment with new path
                            optimized_paths[j] = np.vstack((
                                optimized_paths[j][:wp_counters[j]],
                                new_path[1:],
                                optimized_paths[j][wp_counters[j]+1:]
                            ))
                            
                            # Update current waypoint
                            current_waypoints[j] = new_path[1]
                            print(f"Drone {j} found a new path!")
                
                # Compute force using potential field controller
                pf_force, _ = pf_controllers[j].compute_force(
                    current_pos,
                    current_waypoints[j],
                    obstacle_positions,
                    ALTITUDE
                )
                
                # Apply control action
                action[j, 0:3] = pf_force
                action[j, 3] = 0.95  # Slightly reduced RPM scaling for indoor flight
                
                # Draw debug lines for forces and targets
                if gui and i % 10 == 0:
                    # Draw force vector
                    p.addUserDebugLine(
                        current_pos,
                        current_pos + pf_force,
                        lineColorRGB=[1, 0, 0],
                        lineWidth=2,
                        lifeTime=0.1,
                        physicsClientId=PYB_CLIENT
                    )
                    
                    # Draw line to target
                    p.addUserDebugLine(
                        current_pos,
                        current_waypoints[j],
                        lineColorRGB=[0, 1, 0],
                        lineWidth=1,
                        lifeTime=0.1,
                        physicsClientId=PYB_CLIENT
                    )
                
                # Display debug info
                if gui and i % 50 == 0:
                    text_position = np.array([-18, 13 - (j*1), 0.5])
                    text = f"Drone {j}: Pos={np.round(current_pos, 2)}, WP={wp_counters[j]}/{len(optimized_paths[j])}"
                    p.addUserDebugText(
                        text,
                        text_position,
                        textColorRGB=DRONE_COLORS[j],
                        textSize=1,
                        lifeTime=0.5,
                        physicsClientId=PYB_CLIENT
                    )
            
            # Sync simulation with real time if GUI enabled
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
                
                # Reset camera view periodically
                if i % 500 == 0:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=15,
                        cameraYaw=0,
                        cameraPitch=-70,  # Angled view to see building
                        cameraTargetPosition=[0, 0, 0],
                        physicsClientId=PYB_CLIENT
                    )
            
            # Log data
            for j in range(NUM_DRONES):
                logger.log(
                    drone=j,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[j],
                    control=np.hstack([action[j, 0:3], np.zeros(9)])
                )
            
            # Render
            env.render()
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Clean up
        env.close()
        cv2.destroyAllWindows()
        
        # Print final coverage
        print(f"Final Coverage: {coverage.get_coverage_percentage():.2f}%")
    
    # Save and plot results
    logger.save_as_csv("ga_astar_coverage")
    if plot:
        logger.plot()

#############################################################
# Main Entry Point
#############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone Coverage Planning with GA and A*')
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
    parser.add_argument('--use_building',       default=True, type=str2bool,      help='Whether to use the building simulation (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))