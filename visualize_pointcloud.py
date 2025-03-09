#!/usr/bin/env python3
"""
This script visualizes a single point cloud file to help debug pointcloud generation.

Usage:
    python visualize_pointcloud.py --input path_to_pointcloud.ply

Arguments:
    --input: Path to the input PLY file
    --show-coordinate-frame: Show the coordinate frame
    --background-color: Background color (r,g,b) values between 0-1
"""

import argparse
import numpy as np
import open3d as o3d
import sys


def main():
    parser = argparse.ArgumentParser(description='Visualize a single point cloud file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input PLY file')
    parser.add_argument('--show-coordinate-frame', action='store_true',
                        help='Show the coordinate frame')
    parser.add_argument('--background-color', type=str, default='0.1,0.1,0.1',
                        help='Background color (r,g,b) values between 0-1')
    
    args = parser.parse_args()
    
    # Parse background color
    try:
        bg_color = np.array([float(x) for x in args.background_color.split(',')])
        if len(bg_color) != 3 or any(x < 0 or x > 1 for x in bg_color):
            print("Background color must be three values between 0 and 1")
            sys.exit(1)
    except:
        print("Invalid background color format. Use r,g,b format with values between 0-1")
        sys.exit(1)
    
    # Load the point cloud
    try:
        print(f"Loading point cloud from {args.input}")
        pcd = o3d.io.read_point_cloud(args.input)
        
        # Print point cloud information
        print(f"Point cloud contains {len(pcd.points)} points")
        print(f"Point cloud bounds:")
        print(f"  Min: {pcd.get_min_bound()}")
        print(f"  Max: {pcd.get_max_bound()}")
        print(f"  Center: {pcd.get_center()}")
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add the point cloud
        vis.add_geometry(pcd)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = bg_color
        
        # Add coordinate frame if requested
        if args.show_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
        
        # Set initial viewpoint
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Run the visualizer
        print("Visualizing point cloud. Press 'q' to exit.")
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 