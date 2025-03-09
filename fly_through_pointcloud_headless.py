"""
This script creates a video flying through a sequence of point clouds.
It is designed to work in headless environments by using Open3D's offscreen rendering.

Usage:
    python fly_through_pointcloud_headless.py --input-dir ./vis_video_pointcloud/ply --output-video fly_through.mp4

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
from PIL import Image

# Try to import Open3D with headless rendering support
import open3d as o3d


def get_camera_trajectory(trajectory_type, frame_count, center, radius=3.0):
    """
    Generate camera positions for the trajectory.
    
    Args:
        trajectory_type: Type of trajectory (circle, line, or auto)
        frame_count: Number of frames
        center: Center point of the scene
        radius: Radius of the circle trajectory
        
    Returns:
        List of camera positions
    """
    positions = []
    
    if trajectory_type == 'circle':
        # Circular trajectory around the center
        for i in range(frame_count):
            angle = 2 * math.pi * i / frame_count
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2] + radius * 0.3 * math.sin(angle * 2)  # Add some vertical movement
            positions.append([x, y, z])
    
    elif trajectory_type == 'line':
        # Linear trajectory from left to right
        start_pos = [center[0] - radius, center[1], center[2] + radius/2]
        end_pos = [center[0] + radius, center[1], center[2] + radius/2]
        
        for i in range(frame_count):
            t = i / (frame_count - 1)
            pos = [
                start_pos[0] + t * (end_pos[0] - start_pos[0]),
                start_pos[1] + t * (end_pos[1] - start_pos[1]),
                start_pos[2] + t * (end_pos[2] - start_pos[2])
            ]
            positions.append(pos)
    
    elif trajectory_type == 'auto':
        # Automatic trajectory that combines movements
        for i in range(frame_count):
            t = i / (frame_count - 1)
            angle = 2 * math.pi * t
            
            # Start with a slight zoom in, then circle around, then zoom out
            zoom_factor = 1.0 - 0.3 * math.sin(math.pi * t)
            
            x = center[0] + radius * zoom_factor * math.cos(angle)
            y = center[1] + radius * zoom_factor * math.sin(angle)
            z = center[2] + radius * 0.5 * math.sin(angle * 3)  # More vertical movement
            
            positions.append([x, y, z])
    
    return positions


def render_point_cloud(pcd, width, height, camera_position, look_at, up, point_size=5.0):
    """
    Render a point cloud using Open3D's offscreen rendering.
    
    Args:
        pcd: Open3D point cloud
        width: Image width
        height: Image height
        camera_position: Camera position
        look_at: Look-at point
        up: Up vector
        point_size: Size of points
        
    Returns:
        Rendered image as a numpy array
    """
    # Create a new visualizer for each frame to avoid state issues
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    # Add the point cloud
    vis.add_geometry(pcd)
    
    # Try to set rendering options
    try:
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0, 0, 0])  # Black background
    except Exception as e:
        print(f"Warning: Could not set render options: {e}")
    
    # Set the camera view
    view_control = vis.get_view_control()
    
    # Calculate the camera parameters
    front = np.array(camera_position) - np.array(look_at)
    front = front / np.linalg.norm(front)
    
    # Set the view
    view_control.set_lookat(look_at)
    view_control.set_front(front)
    view_control.set_up(up)
    
    # Update the visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the frame
    try:
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Could not capture screen buffer: {e}")
        # Fallback: create a blank image with text
        image_np = np.zeros((height, width, 3), dtype=np.uint8)
        # Add text explaining the issue
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_np, "Rendering failed - headless mode issue", 
                   (50, height//2), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Clean up
    vis.destroy_window()
    
    return image_np


def main():
    parser = argparse.ArgumentParser(description='Create a fly-through video of point clouds')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing PLY files')
    parser.add_argument('--output-video', type=str, default='fly_through.mp4',
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
    parser.add_argument('--point-size', type=float, default=2.0,
                        help='Size of points in the visualization')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='Process every Nth frame')
    
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
        radius = max(bbox_extent) * 1.5  # 1.5 times the largest dimension
    except Exception as e:
        print(f"Error loading first point cloud: {e}")
        print("Using default center and radius")
        center = [0, 0, 0]
        radius = 3.0
    
    # Generate camera trajectory
    camera_positions = get_camera_trajectory(args.trajectory, len(ply_files), center, radius)
    
    # Process each frame
    for i, ply_file in enumerate(tqdm(ply_files)):
        try:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(ply_file)
            
            # Set camera position
            cam_pos = camera_positions[i]
            look_at = center
            up = [0, 0, 1]  # Z-up orientation
            
            # Render the point cloud
            image = render_point_cloud(
                pcd, 
                args.width, 
                args.height, 
                cam_pos, 
                look_at, 
                up, 
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