#!/usr/bin/env python3
"""
URDF Building Environment Visualizer
====================================

This script loads and visualizes the rectangular building URDF file,
allowing you to inspect the 3D structure, obstacles, and spatial relationships.

Features:
- Interactive 3D visualization
- Camera controls
- Object highlighting
- Measurement tools
- Export screenshots

Usage:
    python urdf_visualizer.py [path_to_urdf_file]
"""

import pybullet as p
import pybullet_data
import time
import argparse
import os
import sys
import numpy as np
from datetime import datetime

class URDFVisualizer:
    def __init__(self, urdf_path, gui=True, floor_texture=None, wall_texture=None):
        """
        Initialize the URDF visualizer
        
        Args:
            urdf_path (str): Path to the URDF file
            gui (bool): Whether to use GUI mode
            floor_texture (str): Path to floor texture image
            wall_texture (str): Path to wall texture image
        """
        self.urdf_path = urdf_path
        self.gui = gui
        self.physics_client = None
        self.building_id = None
        self.camera_distance = 30
        self.camera_yaw = 0
        self.camera_pitch = -45
        self.camera_target = [0, 0, 1]
        
        # Texture paths
        self.floor_texture_path = floor_texture
        self.wall_texture_path = wall_texture
        
        # Color schemes for different object types
        self.colors = {
            'walls': [0.9, 0.9, 0.9, 1.0],      # Light gray
            'floor': [1.0, 1.0, 1.0, 1.0],      # White
            'pillars': [0.6, 0.6, 0.6, 1.0],    # Gray
            'emergency': [0.9, 0.2, 0.2, 1.0],   # Red
            'cracks': [0.2, 0.1, 0.1, 1.0]      # Dark brown
        }
        
    def initialize_physics(self):
        """Initialize PyBullet physics engine"""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            print("🎮 GUI mode enabled - Use mouse to navigate:")
            print("   • Left click + drag: Rotate view")
            print("   • Right click + drag: Zoom")
            print("   • Middle click + drag: Pan")
        else:
            self.physics_client = p.connect(p.DIRECT)
            print("🖥️ Headless mode - generating images only")
        
        # Set additional search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure physics
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        
    def load_urdf(self):
        """Load the URDF file into the simulation"""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        print(f"📄 Loading URDF file: {self.urdf_path}")
        
        try:
            # Load the building URDF
            self.building_id = p.loadURDF(
                self.urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.physics_client
            )
            
            print(f"✅ Successfully loaded building (ID: {self.building_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading URDF: {e}")
            return False
    
    def analyze_structure(self):
        """Analyze and print information about the loaded structure"""
        if self.building_id is None:
            print("❌ No building loaded")
            return
        
        print("\n🏗️ Building Structure Analysis:")
        print("=" * 50)
        
        # Get number of links
        num_joints = p.getNumJoints(self.building_id, physicsClientId=self.physics_client)
        print(f"Number of components (joints): {num_joints}")
        
        # Analyze each component
        components = {
            'walls': [],
            'pillars': [],
            'emergency_stations': [],
            'other': []
        }
        
        # Base link (floor)
        print(f"\n📋 Base Component (Floor):")
        base_info = p.getBasePositionAndOrientation(self.building_id, physicsClientId=self.physics_client)
        print(f"   Position: {base_info[0]}")
        print(f"   Orientation: {base_info[1]}")
        
        # Joint/link information
        print(f"\n📋 Individual Components:")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.building_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            print(f"   {i+1:2d}. {link_name}")
            
            # Categorize components
            if 'wall' in link_name.lower():
                components['walls'].append((i, link_name))
            elif 'pillar' in link_name.lower():
                components['pillars'].append((i, link_name))
            elif 'emergency' in link_name.lower():
                components['emergency_stations'].append((i, link_name))
            else:
                components['other'].append((i, link_name))
        
        # Print categorized summary
        print(f"\n📊 Component Summary:")
        print(f"   Walls: {len(components['walls'])}")
        print(f"   Pillars: {len(components['pillars'])}")
        print(f"   Emergency Stations: {len(components['emergency_stations'])}")
        print(f"   Other Components: {len(components['other'])}")
        
        return components
    
    def setup_camera(self):
        """Setup optimal camera view for the building"""
        if not self.gui:
            return
        
        # Calculate building bounds for optimal camera positioning
        # For 50x20m building, set camera appropriately
        building_width = 50
        building_height = 20
        max_dimension = max(building_width, building_height)
        
        # Set camera distance based on building size
        self.camera_distance = max_dimension * 0.8
        self.camera_target = [0, 0, 1.5]  # Center of building at mid-height
        
        # Set initial camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target,
            physicsClientId=self.physics_client
        )
        
        print(f"📷 Camera positioned at distance {self.camera_distance:.1f}m")
    
    def add_measurement_tools(self):
        """Add visual measurement tools and coordinate system"""
        if self.building_id is None:
            return
        
        # Add coordinate system at origin
        axis_length = 5.0
        
        # X-axis (Red)
        p.addUserDebugLine(
            [0, 0, 0], [axis_length, 0, 0],
            lineColorRGB=[1, 0, 0], lineWidth=3,
            physicsClientId=self.physics_client
        )
        p.addUserDebugText(
            "X", [axis_length + 1, 0, 0],
            textColorRGB=[1, 0, 0], textSize=2,
            physicsClientId=self.physics_client
        )
        
        # Y-axis (Green)
        p.addUserDebugLine(
            [0, 0, 0], [0, axis_length, 0],
            lineColorRGB=[0, 1, 0], lineWidth=3,
            physicsClientId=self.physics_client
        )
        p.addUserDebugText(
            "Y", [0, axis_length + 1, 0],
            textColorRGB=[0, 1, 0], textSize=2,
            physicsClientId=self.physics_client
        )
        
        # Z-axis (Blue)
        p.addUserDebugLine(
            [0, 0, 0], [0, 0, axis_length],
            lineColorRGB=[0, 0, 1], lineWidth=3,
            physicsClientId=self.physics_client
        )
        p.addUserDebugText(
            "Z", [0, 0, axis_length + 1],
            textColorRGB=[0, 0, 1], textSize=2,
            physicsClientId=self.physics_client
        )
        
        # Add building dimension annotations
        self.add_dimension_annotations()
        
    def add_dimension_annotations(self):
        """Add dimension annotations to the building"""
        # Building dimensions: 50m x 20m
        
        # Length annotation (50m)
        """p.addUserDebugLine(
            [-25, -12, 0.1], [25, -12, 0.1],
            lineColorRGB=[0, 0, 0], lineWidth=2,
            physicsClientId=self.physics_client
        )"""
        """p.addUserDebugText(
            "50m", [0, -14, 0.5],
            textColorRGB=[0, 0, 0], textSize=1.5,
            physicsClientId=self.physics_client
        )"""
        
        """# Width annotation (20m)
        p.addUserDebugLine(
            [-27, -10, 0.1], [-27, 10, 0.1],
            lineColorRGB=[0, 0, 0], lineWidth=2,
            physicsClientId=self.physics_client
        )"""
        """p.addUserDebugText(
            "20m", [-30, 0, 0.5],
            textColorRGB=[0, 0, 0], textSize=1.5,
            physicsClientId=self.physics_client
        )"""
        
        """# Height annotation (3m walls)
        p.addUserDebugLine(
            [-27, -10, 0], [-27, -10, 3],
            lineColorRGB=[0, 0, 0], lineWidth=2,
            physicsClientId=self.physics_client
        )"""
        """p.addUserDebugText(
            "3m", [-30, -10, 1.5],
            textColorRGB=[0, 0, 0], textSize=1.5,
            physicsClientId=self.physics_client
        )
    """
    def add_labels(self, components):
        """Add labels to building components"""
        if self.building_id is None or not components:
            return
        
        # Define label positions for different component types
        label_positions = {
            'north_wall': [0, 11, 2],
            'south_wall': [0, -11, 2],
            'east_wall': [26, 0, 2],
            'west_wall': [-26, 0, 2],
            'pillar_1': [-10, 6, 2],
            'pillar_2': [10, -4.5, 2],
            'emergency_station_1': [-5, 6.5, 1],
            'emergency_station_2': [-5, -4.5, 1],
            'emergency_station_3': [10, 6.5, 1],
            'emergency_station_4': [18, -4.5, 1]
        }
        
        # Add labels for major components
        for joint_id, link_name in components['walls']:
            if link_name in label_positions:
                pos = label_positions[link_name]
                p.addUserDebugText(
                    link_name.replace('_', ' ').title(),
                    pos, textColorRGB=[0.2, 0.2, 0.2],
                    textSize=1.2, physicsClientId=self.physics_client
                )
        
        for joint_id, link_name in components['pillars']:
            if link_name in label_positions:
                pos = label_positions[link_name]
                p.addUserDebugText(
                    link_name.replace('_', ' ').title(),
                    pos, textColorRGB=[0.4, 0.4, 0.4],
                    textSize=1.0, physicsClientId=self.physics_client
                )
        
        for joint_id, link_name in components['emergency_stations']:
            if link_name in label_positions:
                pos = label_positions[link_name]
                p.addUserDebugText(
                    "Emergency Station",
                    pos, textColorRGB=[0.8, 0.2, 0.2],
                    textSize=1.0, physicsClientId=self.physics_client
                )
    
    def apply_textures(self):
        """Apply textures to the building components"""
        if self.building_id is None:
            print("❌ No building loaded, cannot apply textures")
            return
        
        print("🎨 Applying textures to building...")
        
        # Validate texture files
        texture_status = self._validate_texture_files()
        
        # Apply floor texture
        floor_success = False
        if self.floor_texture_path and texture_status['floor']:
            floor_success = self._apply_floor_texture()
        
        # Apply wall textures
        wall_success = False
        if self.wall_texture_path and texture_status['wall']:
            wall_success = self._apply_wall_textures()
        
        # Report results
        if floor_success and wall_success:
            print("✅ All textures applied successfully!")
        elif floor_success:
            print("✅ Floor texture applied, ⚠️ wall texture failed")
        elif wall_success:
            print("⚠️ Floor texture failed, ✅ wall texture applied")
        elif self.floor_texture_path or self.wall_texture_path:
            print("❌ Texture application failed")
        else:
            print("ℹ️ No textures specified - using default materials")
    
    def _validate_texture_files(self):
        """Validate that texture files exist and are readable"""
        results = {'floor': False, 'wall': False}
        
        if self.floor_texture_path:
            if os.path.exists(self.floor_texture_path):
                size = os.path.getsize(self.floor_texture_path)
                print(f"✅ Floor texture found: {os.path.basename(self.floor_texture_path)} ({size} bytes)")
                results['floor'] = True
            else:
                print(f"❌ Floor texture not found: {self.floor_texture_path}")
        
        if self.wall_texture_path:
            if os.path.exists(self.wall_texture_path):
                size = os.path.getsize(self.wall_texture_path)
                print(f"✅ Wall texture found: {os.path.basename(self.wall_texture_path)} ({size} bytes)")
                results['wall'] = True
            else:
                print(f"❌ Wall texture not found: {self.wall_texture_path}")
        
        return results
    
    def _apply_floor_texture(self):
        """Apply texture to the floor (base link)"""
        try:
            # Load the floor texture
            texture_id = p.loadTexture(self.floor_texture_path, physicsClientId=self.physics_client)
            
            # Apply to base link (floor) - linkIndex=-1 for base link
            p.changeVisualShape(
                self.building_id, 
                linkIndex=-1,  # base_link is the floor
                textureUniqueId=texture_id,
                rgbaColor=[1, 1, 1, 1],  # White tint for clear texture
                physicsClientId=self.physics_client
            )
            
            print(f"✅ Floor texture applied: {os.path.basename(self.floor_texture_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply floor texture: {e}")
            return False
    
    def _apply_wall_textures(self):
        """Apply texture to all wall components"""
        try:
            # Load wall texture
            wall_texture_id = p.loadTexture(self.wall_texture_path, physicsClientId=self.physics_client)
            print(f"📄 Wall texture loaded: {os.path.basename(self.wall_texture_path)}")
            
            # Method 1: Try expected link indices based on URDF structure
            success_count = self._apply_wall_textures_by_index(wall_texture_id)
            
            # Method 2: If Method 1 fails, try scanning all links
            if success_count == 0:
                print("🔍 Primary method failed, scanning all links...")
                success_count = self._apply_wall_textures_by_scanning(wall_texture_id)
            
            if success_count > 0:
                print(f"✅ Wall texture applied to {success_count} walls")
                return True
            else:
                print("❌ Failed to apply wall texture to any walls")
                return False
            
        except Exception as e:
            print(f"❌ Failed to load wall texture: {e}")
            return False
    
    def _apply_wall_textures_by_index(self, wall_texture_id):
        """Apply wall texture using expected link indices"""
        # Expected wall links based on your URDF structure
        wall_links = {
            0: "north_wall",
            1: "south_wall", 
            2: "east_wall",
            3: "west_wall"
        }
        
        success_count = 0
        
        for link_idx, wall_name in wall_links.items():
            try:
                p.changeVisualShape(
                    self.building_id,
                    linkIndex=link_idx,
                    textureUniqueId=wall_texture_id,
                    rgbaColor=[1, 1, 1, 1],
                    physicsClientId=self.physics_client
                )
                success_count += 1
                print(f"  ✅ Applied to {wall_name} (link {link_idx})")
                
            except Exception as e:
                print(f"  ❌ Failed to apply to {wall_name} (link {link_idx}): {e}")
        
        return success_count
    
    def _apply_wall_textures_by_scanning(self, wall_texture_id):
        """Apply wall texture by scanning all links for walls"""
        try:
            num_joints = p.getNumJoints(self.building_id, physicsClientId=self.physics_client)
            success_count = 0
            
            for i in range(num_joints):
                try:
                    # Get link info
                    joint_info = p.getJointInfo(self.building_id, i, physicsClientId=self.physics_client)
                    link_name = joint_info[12].decode() if joint_info[12] else f"link_{i}"
                    
                    # Apply texture if link name contains "wall"
                    if "wall" in link_name.lower():
                        p.changeVisualShape(
                            self.building_id,
                            linkIndex=i,
                            textureUniqueId=wall_texture_id,
                            rgbaColor=[1, 1, 1, 1],
                            physicsClientId=self.physics_client
                        )
                        success_count += 1
                        print(f"  ✅ Scanned and applied to {link_name} (link {i})")
                        
                except Exception as e:
                    pass  # Silently skip failed attempts
            
            return success_count
            
        except Exception as e:
            print(f"❌ Link scanning failed: {e}")
            return 0
        """Capture a screenshot of the current view"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"building_view_{timestamp}.png"
        
        # Get camera parameters
        width, height = 1920, 1080
        
        # Get current camera view matrix
        camera_info = p.getDebugVisualizerCamera(physicsClientId=self.physics_client)
        view_matrix = camera_info[2]
        proj_matrix = camera_info[3]
        
        # Capture image
        img_arr = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self.physics_client
        )
        
        # Save image
        rgb_array = img_arr[2]
        from PIL import Image
        img = Image.fromarray(rgb_array, 'RGB')
        img.save(filename)
        
        print(f"📸 Screenshot saved: {filename}")
    
    def interactive_controls(self):
        """Display interactive control instructions"""
        if not self.gui:
            return
        
        print("\n🎮 Interactive Controls:")
        print("=" * 30)
        print("Mouse Controls:")
        print("  • Left click + drag: Rotate camera")
        print("  • Right click + drag: Zoom in/out")
        print("  • Middle click + drag: Pan camera")
        print("\nKeyboard Controls:")
        print("  • 'q' or ESC: Quit visualization")
        print("  • 's': Save screenshot")
        print("  • 'r': Reset camera view")
        print("  • 'h': Show/hide help")
    
    def run_visualization(self):
        """Main visualization loop"""
        print("🚀 Starting URDF Visualization...")
        
        # Initialize physics
        self.initialize_physics()
        
        # Load URDF
        if not self.load_urdf():
            return False
        
        # Analyze structure
        components = self.analyze_structure()
        
        # Apply textures if provided
        self.apply_textures()
        
        # Setup visualization
        self.setup_camera()
        self.add_measurement_tools()
        self.add_labels(components)
        
        if self.gui:
            # Interactive mode
            self.interactive_controls()
            
            print("\n🎯 Visualization ready! Press any key to continue...")
            input()
            
            # Main loop for GUI mode
            try:
                while True:
                    # Check for keyboard input
                    keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
                    
                    for key, state in keys.items():
                        if state == 3:  # Key pressed
                            if key == ord('q') or key == 27:  # 'q' or ESC
                                print("👋 Exiting visualization...")
                                return True
                            elif key == ord('s'):
                                self.capture_screenshot()
                            elif key == ord('r'):
                                self.setup_camera()
                                print("📷 Camera view reset")
                            elif key == ord('h'):
                                self.interactive_controls()
                    
                    # Step simulation (for physics updates)
                    p.stepSimulation(physicsClientId=self.physics_client)
                    time.sleep(1/60)  # 60 FPS
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                return True
        else:
            # Headless mode - just capture screenshots
            print("📸 Generating visualization images...")
            
            # Capture different views
            views = [
                {"yaw": 45, "pitch": -30, "distance": 40, "name": "overview"},
                {"yaw": 0, "pitch": -90, "distance": 35, "name": "top_view"},
                {"yaw": 0, "pitch": -15, "distance": 25, "name": "front_view"},
                {"yaw": 90, "pitch": -15, "distance": 25, "name": "side_view"}
            ]
            
            for view in views:
                p.resetDebugVisualizerCamera(
                    cameraDistance=view["distance"],
                    cameraYaw=view["yaw"],
                    cameraPitch=view["pitch"],
                    cameraTargetPosition=self.camera_target,
                    physicsClientId=self.physics_client
                )
                time.sleep(0.1)  # Allow camera to update
                self.capture_screenshot(f"building_{view['name']}.png")
            
            return True
    
    def cleanup(self):
        """Clean up PyBullet resources"""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            print("🧹 Cleaned up PyBullet resources")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize URDF building environment')
    parser.add_argument('urdf_path', nargs='?', default='D:\Documents\ITB\S2\Semester 2\Thesis 2\gym-pybullet-drones\gym_pybullet_drones\envs\RoomNoInteriorOctagonV2.urdf',
                      help='Path to URDF file (default: rectangular_building.urdf)')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (generate images only)')
    parser.add_argument('--output-dir', default='.',
                      help='Output directory for screenshots (default: current directory)')
    parser.add_argument('--floor-texture', type=str,
                      help='Path to floor texture image (JPG/PNG)')
    parser.add_argument('--wall-texture', type=str,
                      help='Path to wall texture image (JPG/PNG)')
    
    args = parser.parse_args()
    
    # Check if URDF file exists
    if not os.path.exists(args.urdf_path):
        print(f"❌ URDF file not found: {args.urdf_path}")
        print("💡 Make sure the URDF file is in the current directory or provide the full path")
        return 1
    
    # Change to output directory if specified
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Default texture paths (modify these to your paths)
    default_floor_texture = r"D:\Documents\ITB\S2\Semester 2\Thesis 2\gym-pybullet-drones\gym_pybullet_drones\envs\concrete-tile-background.jpg"
    default_wall_texture = r"D:\Documents\ITB\S2\Semester 2\Thesis 2\gym-pybullet-drones\gym_pybullet_drones\envs\Walltexture.jpeg"
    
    # Use command line args if provided, otherwise use defaults
    floor_texture = args.floor_texture if args.floor_texture else default_floor_texture
    wall_texture = args.wall_texture if args.wall_texture else default_wall_texture
    
    # Create and run visualizer
    visualizer = URDFVisualizer(
        args.urdf_path, 
        gui=not args.headless,
        floor_texture=floor_texture,
        wall_texture=wall_texture
    )
    
    try:
        success = visualizer.run_visualization()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return 1
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    sys.exit(main())