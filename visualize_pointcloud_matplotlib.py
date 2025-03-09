#!/usr/bin/env python3
"""
This script visualizes a single point cloud file using matplotlib instead of Open3D's visualizer.
This is useful for environments where OpenGL support is limited or not available.

Usage:
    python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply

Arguments:
    --input: Path to the input PLY file
    --point-size: Size of points in the visualization
    --max-points: Maximum number of points to render (for performance)
    --no-subsample: Disable point subsampling (show all points, may be slow)
    --save-image: Save the visualization as an image instead of displaying it
    --output: Path to save the output image (if --save-image is used)
    --view: View angle to use (natural, top, front, side, or custom)
    --rotate-top-view: Rotate the top view by 90 degrees clockwise
    --zoom: Zoom factor for the camera (higher values = closer view)
    --undistort: Apply lens undistortion to reduce fisheye effect
    --undistort-strength: Strength of the undistortion (0.0-1.0)
    --clean-view: Show only the pointcloud without grid, axes, title, or colorbar
    --bg-color: Background color (comma-separated RGB values from 0-1, e.g., "0,0,0" for black)
"""

import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import random


def undistort_points(points, strength=0.5):
    """
    Apply a simple lens undistortion to reduce fisheye effect.
    
    Args:
        points: Nx3 array of point coordinates
        strength: Strength of the undistortion (0.0-1.0)
    
    Returns:
        Undistorted points
    """
    # Make a copy to avoid modifying the original
    undistorted = points.copy()
    
    # Get the center of the points (in XY plane)
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    # Calculate the distance from each point to the center (in XY plane)
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # Calculate the maximum distance for normalization
    max_r = np.max(r)
    if max_r == 0:
        return undistorted
    
    # Normalize distances
    r_norm = r / max_r
    
    # Apply undistortion (Brown-Conrady model simplified)
    # The strength parameter controls how much correction to apply
    # Higher values of strength result in more correction
    k = strength * 0.5  # Scale the strength to a reasonable range
    
    # Calculate the undistortion factor
    factor = 1 + k * r_norm**2
    
    # Apply the undistortion to X and Y coordinates
    undistorted[:, 0] = center_x + dx * factor
    undistorted[:, 1] = center_y + dy * factor
    
    return undistorted


def main():
    parser = argparse.ArgumentParser(description='Visualize a single point cloud file using matplotlib')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input PLY file')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Size of points in the visualization')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Maximum number of points to render (for performance)')
    parser.add_argument('--no-subsample', action='store_true',
                        help='Disable point subsampling (show all points, may be slow)')
    parser.add_argument('--save-image', action='store_true',
                        help='Save the visualization as an image instead of displaying it')
    parser.add_argument('--output', type=str, default='pointcloud_visualization.png',
                        help='Path to save the output image (if --save-image is used)')
    parser.add_argument('--view', type=str, default='natural', 
                        choices=['natural', 'top', 'front', 'side', 'custom'],
                        help='View angle to use')
    parser.add_argument('--rotate-top-view', action='store_true',
                        help='Rotate the top view by 90 degrees clockwise')
    parser.add_argument('--zoom', type=float, default=1.0,
                        help='Zoom factor for the camera (higher values = closer view)')
    parser.add_argument('--undistort', action='store_true',
                        help='Apply lens undistortion to reduce fisheye effect')
    parser.add_argument('--undistort-strength', type=float, default=0.5,
                        help='Strength of the undistortion (0.0-1.0)')
    parser.add_argument('--clean-view', action='store_true',
                        help='Show only the pointcloud without grid, axes, title, or colorbar')
    parser.add_argument('--bg-color', type=str, default='0,0,0',
                        help='Background color (comma-separated RGB values from 0-1, e.g., "0,0,0" for black)')
    
    args = parser.parse_args()
    
    # Parse background color
    try:
        bg_color = [float(x) for x in args.bg_color.split(',')]
        if len(bg_color) != 3:
            print("Background color must have 3 values. Using black.")
            bg_color = [0, 0, 0]
    except:
        print("Invalid background color format. Using black.")
        bg_color = [0, 0, 0]
    
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
        
        # Convert to numpy arrays
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Apply lens undistortion if requested
        if args.undistort:
            print(f"Applying lens undistortion with strength {args.undistort_strength}")
            points = undistort_points(points, args.undistort_strength)
        
        # Rotate points for top view if requested
        if args.view == 'top' and args.rotate_top_view:
            print("Rotating top view by 90 degrees clockwise")
            # Create rotation matrix for 90 degrees clockwise around Z axis
            theta = np.radians(90)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            # Apply rotation to points
            points = np.dot(points, rotation_matrix.T)
        
        # Subsample points if there are too many and subsampling is not disabled
        if len(points) > args.max_points and not args.no_subsample:
            print(f"Subsampling point cloud from {len(points)} to {args.max_points} points for visualization")
            indices = random.sample(range(len(points)), args.max_points)
            points = points[indices]
            colors = colors[indices]
        elif args.no_subsample:
            print(f"Showing all {len(points)} points (no subsampling)")
            if len(points) > 500000:
                print("Warning: Rendering a large number of points may be slow")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
        
        # For natural view, we want to match the original image orientation
        if args.view == 'natural':
            # In the natural view, we want:
            # - X axis pointing right (as in the image)
            # - Y axis pointing down (as in the image)
            # - Z axis pointing away from the camera (depth)
            # This means we need to plot:
            # - X values on the horizontal axis
            # - Y values on the vertical axis (but inverted)
            # - Z values on the depth axis
            
            # Plot the points with natural orientation
            scatter = ax.scatter(
                points[:, 0],  # X (right in image)
                -points[:, 1], # -Y (down in image, but matplotlib Y is up)
                points[:, 2],  # Z (depth)
                c=colors,
                s=args.point_size,
                marker='.',
                alpha=0.8
            )
            
            # Set axis labels for natural view (if not clean view)
            if not args.clean_view:
                ax.set_xlabel('X (right →)')
                ax.set_ylabel('Y (down ↓)')
                ax.set_zlabel('Z (depth →)')
            
            # Set view angle for natural view (looking at XY plane)
            ax.view_init(elev=90, azim=-90)
            
            # Invert y-axis to match image coordinates (top to bottom)
            ax.invert_yaxis()
            
        else:
            # Standard 3D plotting for other views
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors,
                s=args.point_size,
                marker='.',
                alpha=0.8
            )
            
            # Set axis labels (if not clean view)
            if not args.clean_view:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            
            # Set view angle based on the selected view
            if args.view == 'top':
                ax.view_init(elev=90, azim=0)  # Top view (looking down)
            elif args.view == 'front':
                ax.view_init(elev=0, azim=0)   # Front view
            elif args.view == 'side':
                ax.view_init(elev=0, azim=90)  # Side view
            else:  # Custom view
                ax.view_init(elev=30, azim=45)  # Default angle
        
        # Set title (if not clean view)
        if not args.clean_view:
            title = f'Point Cloud Visualization: {args.input} (View: {args.view})'
            if args.undistort:
                title += f', Undistorted (strength={args.undistort_strength})'
            if args.zoom != 1.0:
                title += f', Zoom={args.zoom}x'
            if args.no_subsample:
                title += f', All {len(points)} points'
            ax.set_title(title)
        
        # Create a coordinate frame at the origin (if not clean view)
        if not args.clean_view:
            origin = [0, 0, 0]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
            z_axis = [0, 0, 1]
            
            # Plot coordinate axes
            scale = max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.2
            ax.quiver(origin[0], origin[1], origin[2], x_axis[0]*scale, x_axis[1]*scale, x_axis[2]*scale, color='r')
            ax.quiver(origin[0], origin[1], origin[2], y_axis[0]*scale, y_axis[1]*scale, y_axis[2]*scale, color='g')
            ax.quiver(origin[0], origin[1], origin[2], z_axis[0]*scale, z_axis[1]*scale, z_axis[2]*scale, color='b')
            
            # Add text labels for axes
            ax.text(scale*1.1, 0, 0, "X", color='r')
            ax.text(0, scale*1.1, 0, "Y", color='g')
            ax.text(0, 0, scale*1.1, "Z", color='b')
        
        # Set equal aspect ratio
        ax.set_aspect('auto')
        
        # Clean view settings
        if args.clean_view:
            # Remove axes, grid, and ticks
            ax.set_axis_off()
            ax.grid(False)
            plt.tight_layout(pad=0)
            # Remove margins and padding
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        # Adjust limits to include all points
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        center = pcd.get_center()
        
        # Apply zoom factor (smaller range = closer view)
        max_range = max(max_bound - min_bound) * 0.6 / args.zoom
        
        if args.view == 'natural':
            # For natural view, we need to adjust the limits differently
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([-center[1] - max_range, -center[1] + max_range])  # Inverted Y
            
            # For Z axis, start at 0 (camera position) and extend to max depth
            # Apply zoom to Z axis as well
            z_max = max_bound[2] * 1.1 / args.zoom
            ax.set_zlim([0, z_max])
        else:
            # Standard limits for other views
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
        
        # Add a colorbar (if not clean view)
        if args.view == 'natural' and not args.clean_view:
            # Add a colorbar to show depth
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Depth')
        
        # Save or display the visualization
        if args.save_image:
            print(f"Saving visualization to {args.output}")
            if args.clean_view:
                plt.savefig(args.output, dpi=300, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(args.output, dpi=300, bbox_inches='tight')
        else:
            print("Displaying visualization. Close the window to exit.")
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 