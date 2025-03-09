"""
This script processes a video to generate depth maps and corresponding point clouds for each frame.
The resulting point clouds are saved in the specified output directory.

Usage:
    python video_to_pointcloud.py --encoder vitl --load-from path_to_model --max-depth 20 --video-path path_to_video --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --video-path: Path to the input video file.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
    --frame-step: Process every Nth frame (default: 1, process every frame).
    --save-depth-video: Save a video of the depth maps.
    --save-ply: Save point clouds as PLY files.
    --save-numpy: Save raw depth data as numpy arrays.
"""

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import open3d as o3d
import os
import torch
from PIL import Image
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2


def process_frame(frame, depth_model, input_size, focal_length_x, focal_length_y, max_depth=None, interpolate=False, interpolation_factor=2):
    """
    Process a single video frame to generate a depth map and point cloud.
    
    Args:
        frame: Input video frame (BGR format)
        depth_model: DepthAnythingV2 model instance
        input_size: Input size for the depth model
        focal_length_x: Focal length along the x-axis
        focal_length_y: Focal length along the y-axis
        max_depth: Maximum depth value (for metric depth models)
        interpolate: Whether to interpolate the depth map for denser pointclouds
        interpolation_factor: Factor by which to increase resolution through interpolation
        
    Returns:
        depth: Raw depth map
        points: 3D point cloud coordinates (N x 3)
        colors: RGB colors for each point (N x 3)
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Generate depth map
    depth = depth_model.infer_image(frame, input_size)
    
    # Convert BGR to RGB for point cloud colors
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Interpolate depth map and RGB image if requested
    if interpolate:
        # Calculate new dimensions
        new_height = int(height * interpolation_factor)
        new_width = int(width * interpolation_factor)
        
        # Resize depth map using bicubic interpolation
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Resize RGB image to match
        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Update dimensions
        height, width = new_height, new_width
    
    # Generate mesh grid for 3D projection
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Apply depth scaling if max_depth is provided (for metric models)
    if max_depth is not None:
        # Scale depth values to the specified max_depth
        depth = depth / depth.max() * max_depth
    
    # Filter out only invalid points (depth <= 0)
    # Less aggressive filtering to keep more points
    mask = depth > 0
    
    # Calculate 3D coordinates using pinhole camera model
    # Z = depth
    z = depth[mask]
    # X = (u - cx) * Z / fx
    x = (u[mask] - width / 2) * z / focal_length_x
    # Y = (v - cy) * Z / fy
    y = (v[mask] - height / 2) * z / focal_length_y
    
    # Stack coordinates and colors
    # Adjust coordinate system for better visualization:
    # X: right, Y: down, Z: forward (standard camera coordinate system)
    points = np.stack((x, y, z), axis=-1)
    colors = rgb_frame[mask] / 255.0
    
    return depth, points, colors


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from video frames.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map (for metric models).')
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to the input video file.')
    parser.add_argument('--outdir', type=str, default='./vis_video_pointcloud',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    parser.add_argument('--frame-step', default=1, type=int,
                        help='Process every Nth frame (default: 1, process every frame).')
    parser.add_argument('--save-depth-video', action='store_true',
                        help='Save a video of the depth maps.')
    parser.add_argument('--save-ply', action='store_true',
                        help='Save point clouds as PLY files.')
    parser.add_argument('--save-numpy', action='store_true',
                        help='Save raw depth data as numpy arrays.')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for the depth model.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Do not apply colorful palette to depth visualization.')
    parser.add_argument('--filter-outliers', action='store_true',
                        help='Filter outlier points from the point cloud.')
    parser.add_argument('--outlier-nb-neighbors', type=int, default=20,
                        help='Number of neighbors to consider for outlier filtering.')
    parser.add_argument('--outlier-std-ratio', type=float, default=2.0,
                        help='Standard deviation ratio for outlier filtering.')
    parser.add_argument('--voxel-size', type=float, default=0.0,
                        help='Voxel size for downsampling (0.0 means no downsampling).')
    parser.add_argument('--camera-params', type=str, default=None,
                        help='Path to camera parameters file (if available).')
    parser.add_argument('--auto-detect-encoder', action='store_true',
                        help='Automatically detect encoder type from model filename.')
    parser.add_argument('--keep-all-points', action='store_true',
                        help='Keep all points without filtering (overrides --filter-outliers).')
    parser.add_argument('--interpolate', action='store_true',
                        help='Interpolate depth map for denser pointclouds.')
    parser.add_argument('--interpolation-factor', type=float, default=2.0,
                        help='Factor by which to increase resolution through interpolation.')
    parser.add_argument('--point-density', type=str, choices=['low', 'medium', 'high', 'ultra'], default='medium',
                        help='Preset for point density (sets interpolation factor automatically).')

    args = parser.parse_args()
    
    # Set interpolation factor based on point density preset
    if args.point_density == 'low':
        args.interpolate = True
        args.interpolation_factor = 1.5
    elif args.point_density == 'medium':
        args.interpolate = True
        args.interpolation_factor = 2.0
    elif args.point_density == 'high':
        args.interpolate = True
        args.interpolation_factor = 3.0
    elif args.point_density == 'ultra':
        args.interpolate = True
        args.interpolation_factor = 4.0

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Auto-detect encoder type from filename if requested
    if args.auto_detect_encoder:
        if 'vits' in args.load_from.lower():
            args.encoder = 'vits'
            print(f"Auto-detected encoder: vits")
        elif 'vitb' in args.load_from.lower():
            args.encoder = 'vitb'
            print(f"Auto-detected encoder: vitb")
        elif 'vitl' in args.load_from.lower():
            args.encoder = 'vitl'
            print(f"Auto-detected encoder: vitl")
        elif 'vitg' in args.load_from.lower():
            args.encoder = 'vitg'
            print(f"Auto-detected encoder: vitg")
        else:
            print(f"Could not auto-detect encoder type from filename. Using specified encoder: {args.encoder}")

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Check if the model is a metric depth model
    is_metric = 'metric' in args.load_from
    model_config = {**model_configs[args.encoder]}
    if is_metric:
        model_config['max_depth'] = args.max_depth
        print(f"Using metric depth model with max_depth={args.max_depth}")
    else:
        print("Using relative depth model. Depth values will be scaled.")

    # Initialize the DepthAnythingV2 model with the specified configuration
    print(f"Initializing model with encoder={args.encoder}, features={model_config['features']}, out_channels={model_config['out_channels']}")
    try:
        depth_anything = DepthAnythingV2(**model_config)
        depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("\nThis error typically occurs when there's a mismatch between the model architecture and the weights.")
        print("Please make sure you're using the correct encoder type for your model weights.")
        print("For example, if you're using 'depth_anything_v2_vits.pth', you should use '--encoder vits'.")
        print("You can also try using '--auto-detect-encoder' to automatically detect the encoder type from the filename.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create subdirectories for different output types
    if args.save_ply:
        ply_dir = os.path.join(args.outdir, 'ply')
        os.makedirs(ply_dir, exist_ok=True)
    
    if args.save_numpy:
        numpy_dir = os.path.join(args.outdir, 'numpy')
        os.makedirs(numpy_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(args.video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {frame_rate} FPS, {total_frames} frames")
    
    # Load camera parameters if provided
    if args.camera_params and os.path.exists(args.camera_params):
        try:
            import json
            with open(args.camera_params, 'r') as f:
                camera_params = json.load(f)
            
            # Override focal length with values from camera parameters
            if 'fx' in camera_params and 'fy' in camera_params:
                args.focal_length_x = camera_params['fx']
                args.focal_length_y = camera_params['fy']
                print(f"Using camera parameters from file: fx={args.focal_length_x}, fy={args.focal_length_y}")
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            print("Using default focal length values")
    
    print(f"Using focal length: fx={args.focal_length_x}, fy={args.focal_length_y}")
    
    # Print interpolation settings if enabled
    if args.interpolate:
        print(f"Using interpolation with factor {args.interpolation_factor} for denser pointclouds")
        print(f"Point density preset: {args.point_density}")
    
    # Set up depth video writer if requested
    if args.save_depth_video:
        depth_video_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.video_path))[0] + '_depth.mp4')
        depth_video = cv2.VideoWriter(depth_video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
    
    # Color map for depth visualization
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Process video frames
    frame_idx = 0
    processed_count = 0
    
    with tqdm(total=total_frames // args.frame_step) as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % args.frame_step == 0:
                # Process the frame
                depth, points, colors = process_frame(
                    frame, 
                    depth_anything, 
                    args.input_size, 
                    args.focal_length_x, 
                    args.focal_length_y,
                    args.max_depth if is_metric else None,
                    args.interpolate,
                    args.interpolation_factor
                )
                
                # Save point cloud as PLY if requested
                if args.save_ply and len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Filter outliers if requested and not overridden
                    if args.filter_outliers and not args.keep_all_points:
                        print(f"Filtering outliers with nb_neighbors={args.outlier_nb_neighbors}, std_ratio={args.outlier_std_ratio}")
                        pcd, _ = pcd.remove_statistical_outlier(
                            nb_neighbors=args.outlier_nb_neighbors, 
                            std_ratio=args.outlier_std_ratio
                        )
                    
                    # Downsample if voxel size is specified
                    if args.voxel_size > 0:
                        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                    
                    ply_path = os.path.join(ply_dir, f"frame_{frame_idx:06d}.ply")
                    o3d.io.write_point_cloud(ply_path, pcd)
                
                # Save raw depth data as numpy array if requested
                if args.save_numpy:
                    numpy_path = os.path.join(numpy_dir, f"frame_{frame_idx:06d}.npy")
                    np.save(numpy_path, depth)
                
                # Save depth visualization if requested
                if args.save_depth_video:
                    # Normalize depth for visualization
                    norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    norm_depth = norm_depth.astype(np.uint8)
                    
                    # Apply color map if not grayscale
                    if args.grayscale:
                        vis_depth = np.repeat(norm_depth[..., np.newaxis], 3, axis=-1)
                    else:
                        vis_depth = (cmap(norm_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    
                    # Resize to original dimensions if interpolated
                    if args.interpolate:
                        vis_depth = cv2.resize(vis_depth, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                    
                    depth_video.write(vis_depth)
                
                processed_count += 1
                pbar.update(1)
            
            frame_idx += 1
    
    # Clean up
    video.release()
    if args.save_depth_video:
        depth_video.release()
    
    print(f"Processed {processed_count} frames from {args.video_path}")
    print(f"Results saved to {args.outdir}")
    
    # Print instructions for visualizing the point clouds
    if args.save_ply:
        print("\nTo visualize the point clouds, you can use:")
        print(f"  python visualize_pointcloud_matplotlib.py --input {os.path.join(ply_dir, 'frame_000000.ply')} --view natural --zoom 2.0 --undistort --clean-view --no-subsample --save-image")
        print(f"  python generate_pointcloud_views.py --input {os.path.join(ply_dir, 'frame_000000.ply')} --output-dir pointcloud_views --natural-view-only --zoom 2.0 --undistort --clean-view --no-subsample")
        print("\nThe natural view preserves the original image orientation (top is top, bottom is bottom, etc.)")
        print("Use --zoom to move the camera closer (higher values = closer view)")
        print("Use --undistort to reduce fisheye distortion effect")
        print("Use --clean-view to show only the pointcloud without grid, axes, title, or colorbar")
        print("Use --no-subsample to show all points without downsampling (may be slower)")


if __name__ == '__main__':
    main() 