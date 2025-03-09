#!/usr/bin/env python3
"""
This script generates multiple views of a point cloud and saves them as images.
This is useful for debugging pointclouds in environments where interactive visualization is not available.

Usage:
    python generate_pointcloud_views.py --input path_to_pointcloud.ply --output-dir output_directory

Arguments:
    --input: Path to the input PLY file
    --output-dir: Directory to save the output images
    --point-size: Size of points in the visualization
    --max-points: Maximum number of points to render (for performance)
    --no-subsample: Disable point subsampling (show all points, may be slow)
    --num-views: Number of views to generate
    --rotate-top-view: Rotate the top view by 90 degrees clockwise
    --natural-view-only: Generate only the natural view (matching original image orientation)
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
import os
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
    parser = argparse.ArgumentParser(description='Generate multiple views of a point cloud and save them as images')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input PLY file')
    parser.add_argument('--output-dir', type=str, default='pointcloud_views',
                        help='Directory to save the output images')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Size of points in the visualization')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Maximum number of points to render (for performance)')
    parser.add_argument('--no-subsample', action='store_true',
                        help='Disable point subsampling (show all points, may be slow)')
    parser.add_argument('--num-views', type=int, default=6,
                        help='Number of views to generate')
    parser.add_argument('--rotate-top-view', action='store_true',
                        help='Rotate the top view by 90 degrees clockwise')
    parser.add_argument('--natural-view-only', action='store_true',
                        help='Generate only the natural view (matching original image orientation)')
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        
        # Calculate bounds for consistent view across all images
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        center = pcd.get_center()
        
        # Apply zoom factor (smaller range = closer view)
        max_range = max(max_bound - min_bound) * 0.6 / args.zoom
        
        # Generate natural view first (matching original image orientation)
        print("Generating natural view (matching original image orientation)")
        
        # Create figure and 3D axis for natural view
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
        
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
            
            # Set title
            title = 'Point Cloud View: Natural (matches original image)'
            if args.undistort:
                title += f', Undistorted (strength={args.undistort_strength})'
            if args.zoom != 1.0:
                title += f', Zoom={args.zoom}x'
            if args.no_subsample:
                title += f', All {len(points)} points'
            ax.set_title(title)
            
            # Create a coordinate frame at the origin
            origin = [0, 0, 0]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
            z_axis = [0, 0, 1]
            
            # Plot coordinate axes
            scale = max(max_bound - min_bound) * 0.2
            ax.quiver(origin[0], origin[1], origin[2], x_axis[0]*scale, x_axis[1]*scale, x_axis[2]*scale, color='r')
            ax.quiver(origin[0], origin[1], origin[2], y_axis[0]*scale, y_axis[1]*scale, y_axis[2]*scale, color='g')
            ax.quiver(origin[0], origin[1], origin[2], z_axis[0]*scale, z_axis[1]*scale, z_axis[2]*scale, color='b')
            
            # Add text labels for axes
            ax.text(scale*1.1, 0, 0, "X", color='r')
            ax.text(0, scale*1.1, 0, "Y", color='g')
            ax.text(0, 0, scale*1.1, "Z", color='b')
        
        # Set equal aspect ratio
        ax.set_aspect('auto')
        
        # Set view angle for natural view (looking at XY plane)
        ax.view_init(elev=90, azim=-90)
        
        # Invert y-axis to match image coordinates (top to bottom)
        ax.invert_yaxis()
        
        # Clean view settings
        if args.clean_view:
            # Remove axes, grid, and ticks
            ax.set_axis_off()
            ax.grid(False)
            plt.tight_layout(pad=0)
            # Remove margins and padding
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        # Set limits for natural view
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([-center[1] - max_range, -center[1] + max_range])  # Inverted Y
        
        # For Z axis, start at 0 (camera position) and extend to max depth
        # Apply zoom to Z axis as well
        z_max = max_bound[2] * 1.1 / args.zoom
        ax.set_zlim([0, z_max])
        
        # Add a colorbar to show depth (if not clean view)
        if not args.clean_view:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Depth')
        
        # Save the natural view
        natural_output_path = os.path.join(args.output_dir, 'pointcloud_natural.png')
        if args.clean_view:
            plt.savefig(natural_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(natural_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved natural view to {natural_output_path}")
        
        # If natural view only, we're done
        if args.natural_view_only:
            print("Natural view only requested. Skipping other views.")
            return
        
        # Generate standard views
        print(f"Generating {args.num_views} standard views...")
        
        # Define view angles
        if args.num_views == 6:
            # Standard 6 views: front, back, left, right, top, bottom
            elevations = [0, 0, 0, 0, 90, -90]
            azimuths = [0, 180, 90, 270, 0, 0]
            view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        else:
            # Generate evenly distributed views around the object
            elevations = []
            azimuths = []
            view_names = []
            
            for i in range(args.num_views):
                # Distribute views evenly around the object
                azimuth = i * (360 / args.num_views)
                elevation = 30  # Default elevation
                
                # For more than 6 views, add some variation in elevation
                if args.num_views > 6 and i % 2 == 0:
                    elevation = 60
                
                elevations.append(elevation)
                azimuths.append(azimuth)
                view_names.append(f'view_{i:02d}')
        
        # Generate each view
        for i, (elev, azim, name) in enumerate(zip(elevations, azimuths, view_names)):
            print(f"Generating view {i+1}/{len(elevations)}: {name} (elevation={elev}, azimuth={azim})")
            
            # Create a copy of the points for this view
            view_points = np.copy(points)
            
            # Rotate points for top view if requested
            if name == 'top' and args.rotate_top_view:
                print("  Rotating top view by 90 degrees clockwise")
                # Create rotation matrix for 90 degrees clockwise around Z axis
                theta = np.radians(90)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                # Apply rotation to points
                view_points = np.dot(view_points, rotation_matrix.T)
            
            # Create figure and 3D axis
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set background color
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            
            # Plot the points
            ax.scatter(
                view_points[:, 0], view_points[:, 1], view_points[:, 2],
                c=colors,
                s=args.point_size,
                marker='.',
                alpha=0.8
            )
            
            # Set axis labels and title (if not clean view)
            if not args.clean_view:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set title
                title = f'Point Cloud View: {name} (elev={elev}, azim={azim})'
                if args.undistort:
                    title += f', Undistorted (strength={args.undistort_strength})'
                if args.zoom != 1.0:
                    title += f', Zoom={args.zoom}x'
                if args.no_subsample:
                    title += f', All {len(points)} points'
                ax.set_title(title)
                
                # Create a coordinate frame at the origin
                origin = [0, 0, 0]
                x_axis = [1, 0, 0]
                y_axis = [0, 1, 0]
                z_axis = [0, 0, 1]
                
                # Plot coordinate axes
                scale = max(max_bound - min_bound) * 0.2
                ax.quiver(origin[0], origin[1], origin[2], x_axis[0]*scale, x_axis[1]*scale, x_axis[2]*scale, color='r')
                ax.quiver(origin[0], origin[1], origin[2], y_axis[0]*scale, y_axis[1]*scale, y_axis[2]*scale, color='g')
                ax.quiver(origin[0], origin[1], origin[2], z_axis[0]*scale, z_axis[1]*scale, z_axis[2]*scale, color='b')
                
                # Add text labels for axes
                ax.text(scale*1.1, 0, 0, "X", color='r')
                ax.text(0, scale*1.1, 0, "Y", color='g')
                ax.text(0, 0, scale*1.1, "Z", color='b')
            
            # Set equal aspect ratio
            ax.set_aspect('auto')
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Clean view settings
            if args.clean_view:
                # Remove axes, grid, and ticks
                ax.set_axis_off()
                ax.grid(False)
                plt.tight_layout(pad=0)
                # Remove margins and padding
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            
            # Set consistent limits with zoom
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
            
            # Save the image
            output_path = os.path.join(args.output_dir, f'pointcloud_{name}.png')
            if args.clean_view:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved view to {output_path}")
            
            # For top view, also generate a rotated version if not already rotated
            if name == 'top' and not args.rotate_top_view:
                print("  Generating rotated top view (90 degrees clockwise)")
                
                # Create rotation matrix for 90 degrees clockwise around Z axis
                theta = np.radians(90)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                # Apply rotation to points
                rotated_points = np.dot(points, rotation_matrix.T)
                
                # Create figure and 3D axis for rotated view
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Set background color
                ax.set_facecolor(bg_color)
                fig.patch.set_facecolor(bg_color)
                
                # Plot the rotated points
                ax.scatter(
                    rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                    c=colors,
                    s=args.point_size,
                    marker='.',
                    alpha=0.8
                )
                
                # Set axis labels and title (if not clean view)
                if not args.clean_view:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Set title
                    title = f'Point Cloud View: {name}_rotated (elev={elev}, azim={azim})'
                    if args.undistort:
                        title += f', Undistorted (strength={args.undistort_strength})'
                    if args.zoom != 1.0:
                        title += f', Zoom={args.zoom}x'
                    if args.no_subsample:
                        title += f', All {len(points)} points'
                    ax.set_title(title)
                    
                    # Plot coordinate axes
                    ax.quiver(origin[0], origin[1], origin[2], x_axis[0]*scale, x_axis[1]*scale, x_axis[2]*scale, color='r')
                    ax.quiver(origin[0], origin[1], origin[2], y_axis[0]*scale, y_axis[1]*scale, y_axis[2]*scale, color='g')
                    ax.quiver(origin[0], origin[1], origin[2], z_axis[0]*scale, z_axis[1]*scale, z_axis[2]*scale, color='b')
                    
                    # Add text labels for axes
                    ax.text(scale*1.1, 0, 0, "X", color='r')
                    ax.text(0, scale*1.1, 0, "Y", color='g')
                    ax.text(0, 0, scale*1.1, "Z", color='b')
                
                # Set equal aspect ratio
                ax.set_aspect('auto')
                
                # Set view angle
                ax.view_init(elev=elev, azim=azim)
                
                # Clean view settings
                if args.clean_view:
                    # Remove axes, grid, and ticks
                    ax.set_axis_off()
                    ax.grid(False)
                    plt.tight_layout(pad=0)
                    # Remove margins and padding
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                
                # Set consistent limits with zoom
                ax.set_xlim([center[0] - max_range, center[0] + max_range])
                ax.set_ylim([center[1] - max_range, center[1] + max_range])
                ax.set_zlim([center[2] - max_range, center[2] + max_range])
                
                # Save the rotated image
                rotated_output_path = os.path.join(args.output_dir, f'pointcloud_{name}_rotated.png')
                if args.clean_view:
                    plt.savefig(rotated_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(rotated_output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  Saved rotated view to {rotated_output_path}")
        
        print(f"All views saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 