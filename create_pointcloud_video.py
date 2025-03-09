"""
This script creates a video flying through a sequence of point clouds using matplotlib
instead of Open3D's visualization. This approach works better in headless environments.

Usage:
    python create_pointcloud_video.py --input-dir ./vis_video_pointcloud/ply --output-video pointcloud_video.mp4

Arguments:
    --input-dir: Directory containing the PLY files
    --output-video: Path to save the output video
    --width: Width of the output video
    --height: Height of the output video
    --fps: Frames per second of the output video
    --trajectory: Type of camera trajectory (circle, line, or auto)
    --start-frame: First frame to include
    --end-frame: Last frame to include
    --point-size: Size of points in the visualization
    --frame-step: Process every Nth frame
    --max-points: Maximum number of points to render (for performance)
"""

import argparse
import cv2
import glob
import numpy as np
import os
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D


def get_camera_trajectory(trajectory_type, frame_count, center, radius=3.0):
    """
    Generate camera positions and viewing angles for the trajectory.
    
    Args:
        trajectory_type: Type of trajectory (circle, line, or auto)
        frame_count: Number of frames
        center: Center point of the scene
        radius: Radius of the circle trajectory
        
    Returns:
        Tuple of (camera positions, elevation angles, azimuth angles)
    """
    positions = []
    elevations = []
    azimuths = []
    
    if trajectory_type == 'circle':
        # Circular trajectory around the center
        for i in range(frame_count):
            t = i / frame_count
            azimuth = 360 * t  # Full circle
            elevation = 20 + 10 * math.sin(2 * math.pi * t)  # Slight up and down
            
            positions.append(center)
            elevations.append(elevation)
            azimuths.append(azimuth)
    
    elif trajectory_type == 'line':
        # Linear trajectory from left to right
        for i in range(frame_count):
            t = i / (frame_count - 1)
            azimuth = 180 - 180 * t  # 180 to 0 degrees
            elevation = 20  # Fixed elevation
            
            positions.append(center)
            elevations.append(elevation)
            azimuths.append(azimuth)
    
    elif trajectory_type == 'auto':
        # Automatic trajectory that combines movements
        for i in range(frame_count):
            t = i / frame_count
            azimuth = 360 * t  # Full circle
            elevation = 20 + 15 * math.sin(2 * math.pi * t)  # More up and down
            
            positions.append(center)
            elevations.append(elevation)
            azimuths.append(azimuth)
    
    return positions, elevations, azimuths


def render_point_cloud_matplotlib(points, colors, width, height, elevation, azimuth, center, radius, point_size=1.0):
    """
    Render a point cloud using matplotlib.
    
    Args:
        points: Nx3 array of point coordinates
        colors: Nx3 array of RGB colors (0-1)
        width: Image width
        height: Image height
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        center: Center point of the scene
        radius: Radius for the plot limits
        point_size: Size of points
        
    Returns:
        Rendered image as a numpy array
    """
    # Create figure and 3D axis
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,
        s=point_size,
        marker='.',
        alpha=0.8
    )
    
    # Set the view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set axis limits
    ax.set_xlim([center[0] - radius, center[0] + radius])
    ax.set_ylim([center[1] - radius, center[1] + radius])
    ax.set_zlim([center[2] - radius, center[2] + radius])
    
    # Remove axis, grid, and background
    ax.set_axis_off()
    ax.grid(False)
    ax.set_facecolor((0, 0, 0))
    fig.set_facecolor((0, 0, 0))
    
    # Render the figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.array(canvas.renderer.buffer_rgba())
    
    # Convert RGBA to RGB
    image = image[:, :, :3]
    
    # Close the figure to free memory
    plt.close(fig)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Create a fly-through video of point clouds using matplotlib')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing PLY files')
    parser.add_argument('--output-video', type=str, default='pointcloud_video.mp4',
                        help='Path to save the output video')
    parser.add_argument('--width', type=int, default=1280,
                        help='Width of the output video')
    parser.add_argument('--height', type=int, default=720,
                        help='Height of the output video')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second of the output video')
    parser.add_argument('--trajectory', type=str, default='auto',
                        choices=['circle', 'line', 'auto'],
                        help='Type of camera trajectory')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame to include')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Last frame to include')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Size of points in the visualization')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Maximum number of points to render (for performance)')
    
    args = parser.parse_args()
    
    # Find all PLY files in the input directory
    ply_files = sorted(glob.glob(os.path.join(args.input_dir, '*.ply')))
    
    if not ply_files:
        print(f"No PLY files found in {args.input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    # Apply frame step
    if args.frame_step > 1:
        ply_files = ply_files[::args.frame_step]
        print(f"Using every {args.frame_step}th frame, resulting in {len(ply_files)} frames")
    
    # Apply frame range limits
    start_idx = args.start_frame
    end_idx = args.end_frame if args.end_frame is not None else len(ply_files)
    ply_files = ply_files[start_idx:end_idx]
    
    print(f"Processing frames {start_idx} to {end_idx-1}")
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (args.width, args.height))
    
    # Load the first point cloud to get scene bounds
    try:
        first_pcd = o3d.io.read_point_cloud(ply_files[0])
        center = first_pcd.get_center()
        
        # Estimate a good radius for the camera trajectory
        bbox = first_pcd.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        radius = max(bbox_extent) * 1.2  # 1.2 times the largest dimension
    except Exception as e:
        print(f"Error loading first point cloud: {e}")
        print("Using default center and radius")
        center = [0, 0, 0]
        radius = 3.0
    
    # Generate camera trajectory
    positions, elevations, azimuths = get_camera_trajectory(
        args.trajectory, len(ply_files), center, radius
    )
    
    # Process each frame
    for i, ply_file in enumerate(tqdm(ply_files)):
        try:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(ply_file)
            
            # Get points and colors
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Subsample points if there are too many
            if len(points) > args.max_points:
                indices = np.random.choice(len(points), args.max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Render the point cloud
            image = render_point_cloud_matplotlib(
                points,
                colors,
                args.width,
                args.height,
                elevations[i],
                azimuths[i],
                center,
                radius,
                args.point_size
            )
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video
            video_writer.write(image_bgr)
            
        except Exception as e:
            print(f"Error processing frame {i} ({ply_file}): {e}")
            # Create a blank frame with error message
            blank_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blank_frame, f"Error: {str(e)}", 
                       (50, args.height//2), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            video_writer.write(blank_frame)
    
    # Clean up
    video_writer.release()
    
    print(f"Video saved to {args.output_video}")


if __name__ == '__main__':
    main() 