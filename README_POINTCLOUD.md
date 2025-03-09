# Depth-Anything-V2 Pointcloud Generation

This README provides instructions for generating and visualizing pointclouds from videos using Depth-Anything-V2.

## Fixed Issues

The pointcloud generation code has been updated to fix several issues:

1. Improved 3D projection calculation using the correct pinhole camera model
2. Added filtering for invalid or distant points
3. Added support for proper depth scaling for both metric and relative depth models
4. Added options for filtering outliers and downsampling pointclouds
5. Added support for loading camera parameters from a file
6. Fixed model architecture mismatch issues with automatic encoder detection
7. Added alternative visualization tools for environments with limited OpenGL support
8. Improved pointcloud completeness with less aggressive filtering
9. Added support for rotated views in visualization
10. Added natural view option that preserves original image orientation
11. Added zoom and lens undistortion options to improve visualization
12. Added clean view option to show only the pointcloud without grid, axes, title, or colorbar
13. Added interpolation options for generating denser pointclouds
14. Added option to disable point subsampling in visualization

## Usage

### Generating Pointclouds from Video

```bash
python video_to_pointcloud.py \
  --encoder vits \
  --auto-detect-encoder \
  --load-from path_to_model_weights \
  --max-depth 20 \
  --video-path path_to_video \
  --outdir output_directory \
  --focal-length-x 470.4 \
  --focal-length-y 470.4 \
  --save-ply \
  --keep-all-points \
  --point-density high
```

### Important Parameters

- `--encoder`: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg']. **Must match your model weights!**
- `--auto-detect-encoder`: Automatically detect encoder type from model filename (recommended).
- `--load-from`: Path to the pre-trained model weights.
- `--max-depth`: Maximum depth value for the depth map (for metric models).
- `--video-path`: Path to the input video file.
- `--outdir`: Directory to save the output point clouds.
- `--focal-length-x`: Focal length along the x-axis.
- `--focal-length-y`: Focal length along the y-axis.
- `--frame-step`: Process every Nth frame (default: 1, process every frame).
- `--save-ply`: Save point clouds as PLY files.
- `--filter-outliers`: Filter outlier points from the point cloud.
- `--keep-all-points`: Keep all points without filtering (overrides --filter-outliers).
- `--outlier-nb-neighbors`: Number of neighbors to consider for outlier filtering (default: 20).
- `--outlier-std-ratio`: Standard deviation ratio for outlier filtering (default: 2.0).
- `--voxel-size`: Voxel size for downsampling (0.0 means no downsampling).
- `--camera-params`: Path to camera parameters file (if available).
- `--input-size`: Input size for the depth model (default: 518, try 768 or 1024 for more detail).
- `--point-density`: Preset for point density. Choices are ['low', 'medium', 'high', 'ultra'] (default: 'medium').
- `--interpolate`: Interpolate depth map for denser pointclouds.
- `--interpolation-factor`: Factor by which to increase resolution through interpolation (default: 2.0).

### Point Density Presets

The `--point-density` parameter provides convenient presets for generating pointclouds with different densities:

| Preset | Interpolation Factor | Description |
|--------|---------------------|-------------|
| low    | 1.5                 | Slightly denser than original |
| medium | 2.0                 | Good balance of density and performance |
| high   | 3.0                 | Dense pointcloud, may be slower |
| ultra  | 4.0                 | Very dense pointcloud, significantly slower |

For example, to generate a high-density pointcloud:
```bash
python video_to_pointcloud.py --auto-detect-encoder --load-from ./checkpoints/depth_anything_v2_vits.pth --max-depth 20 --video-path path_to_video --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4 --save-ply --keep-all-points --point-density high
```

### Model Architecture

Depth-Anything-V2 supports different encoder architectures:

| Encoder | Features | Out Channels | Model File Example |
|---------|----------|--------------|-------------------|
| vits    | 64       | [48, 96, 192, 384] | depth_anything_v2_vits.pth |
| vitb    | 128      | [96, 192, 384, 768] | depth_anything_v2_vitb.pth |
| vitl    | 256      | [256, 512, 1024, 1024] | depth_anything_v2_vitl.pth |
| vitg    | 384      | [1536, 1536, 1536, 1536] | depth_anything_v2_vitg.pth |

**Important**: You must use the correct encoder type that matches your model weights. If you're unsure, use the `--auto-detect-encoder` option to automatically detect the encoder type from the model filename.

### Camera Parameters

The focal length parameters are crucial for correct pointcloud generation. If you know the camera parameters, you can provide them directly with `--focal-length-x` and `--focal-length-y`, or you can create a JSON file with the camera parameters and provide it with `--camera-params`.

Example camera parameters file:
```json
{
  "fx": 470.4,
  "fy": 470.4,
  "cx": 320.0,
  "cy": 240.0
}
```

## Visualizing Pointclouds

### Using Open3D (requires OpenGL support)

To visualize a single pointcloud using Open3D:

```bash
python visualize_pointcloud.py --input path_to_pointcloud.ply --show-coordinate-frame
```

### Using Matplotlib (works in environments with limited OpenGL support)

If you encounter OpenGL errors with the Open3D visualizer, use the matplotlib-based visualization tools:

```bash
# Visualize a single pointcloud using the natural view with clean display and no subsampling
python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --zoom 2.0 --undistort --clean-view --no-subsample
```

```bash
# Generate multiple views of a pointcloud and save them as images
python generate_pointcloud_views.py --input path_to_pointcloud.ply --output-dir pointcloud_views --zoom 2.0 --undistort --clean-view --no-subsample
```

```bash
# Generate only the natural view with custom undistortion strength and clean display
python generate_pointcloud_views.py --input path_to_pointcloud.ply --output-dir pointcloud_views --natural-view-only --zoom 3.0 --undistort --undistort-strength 0.7 --clean-view --no-subsample
```

#### Visualization Options

- `--view`: Choose from 'natural', 'top', 'front', 'side', or 'custom' (default: 'natural')
  - **natural**: Preserves the original image orientation (recommended)
  - **top**: View from above
  - **front**: View from front
  - **side**: View from side
  - **custom**: Custom angle
- `--zoom`: Zoom factor for the camera (higher values = closer view, default: 1.0)
- `--undistort`: Apply lens undistortion to reduce fisheye effect
- `--undistort-strength`: Strength of the undistortion (0.0-1.0, default: 0.5)
- `--clean-view`: Show only the pointcloud without grid, axes, title, or colorbar
- `--bg-color`: Background color (comma-separated RGB values from 0-1, e.g., "0,0,0" for black)
- `--no-subsample`: Disable point subsampling (show all points, may be slow)
- `--max-points`: Maximum number of points to render if subsampling is enabled (default: 100000)
- `--rotate-top-view`: Rotate the top view by 90 degrees clockwise
- `--point-size`: Size of points in the visualization (default: 0.5)
- `--save-image`: Save the visualization as an image instead of displaying it
- `--output`: Path to save the output image (default: 'pointcloud_visualization.png')
- `--natural-view-only`: Generate only the natural view (for generate_pointcloud_views.py)

The matplotlib-based tools have several advantages:
- Work in environments with limited or no OpenGL support (like remote servers or Docker containers)
- Can save visualizations as images for later inspection
- Allow for generating multiple views of the pointcloud automatically
- Support natural view that preserves the original image orientation
- Provide zoom and lens undistortion options to improve visualization
- Offer clean view option to show only the pointcloud without distractions
- Allow viewing all points without subsampling for maximum detail

### Creating a Fly-through Video

To create a fly-through video of the generated pointclouds:

```bash
python fly_through_pointcloud.py --input-dir output_directory/ply --output-video pointcloud_video.mp4
```

## Troubleshooting

### Sparse Pointclouds

If your pointclouds appear too sparse or "dotty", especially in areas closer to the camera:

1. Use the `--point-density` option to generate denser pointclouds:
   ```bash
   python video_to_pointcloud.py --auto-detect-encoder --load-from ./checkpoints/depth_anything_v2_vits.pth --max-depth 20 --video-path path_to_video --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4 --save-ply --keep-all-points --point-density high
   ```

2. For even denser pointclouds, use the `--interpolation-factor` option with a higher value:
   ```bash
   python video_to_pointcloud.py --auto-detect-encoder --load-from ./checkpoints/depth_anything_v2_vits.pth --max-depth 20 --video-path path_to_video --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4 --save-ply --keep-all-points --interpolate --interpolation-factor 4.0
   ```

3. Increase the input size for the depth model:
   ```bash
   python video_to_pointcloud.py --auto-detect-encoder --load-from ./checkpoints/depth_anything_v2_vits.pth --max-depth 20 --video-path path_to_video --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4 --save-ply --keep-all-points --input-size 768
   ```

4. When visualizing, use the `--no-subsample` option to see all points without downsampling:
   ```bash
   python visualize_pointcloud_matplotlib.py --input output_directory/ply/frame_000000.ply --view natural --no-subsample
   ```

### Incorrect Pointcloud Orientation

If the pointcloud orientation doesn't match the original image:

1. Use the `--view natural` option with `visualize_pointcloud_matplotlib.py` to get a view that preserves the original image orientation:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural
   ```

2. Or generate only the natural view with `generate_pointcloud_views.py`:
   ```bash
   python generate_pointcloud_views.py --input path_to_pointcloud.ply --output-dir pointcloud_views --natural-view-only
   ```

The natural view ensures that:
- X axis points right (as in the original image)
- Y axis points down (as in the original image)
- Z axis represents depth (away from the camera)

### Clean Visualization

If you want to see only the pointcloud without grid, axes, title, or colorbar:

1. Use the `--clean-view` option:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --clean-view
   ```

2. You can also customize the background color:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --clean-view --bg-color "0.1,0.1,0.1"
   ```

### Fisheye Distortion

If the pointcloud appears to have a fisheye distortion effect:

1. Use the `--undistort` option to apply lens undistortion:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --undistort
   ```

2. Adjust the undistortion strength with `--undistort-strength` (values between 0.0-1.0):
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --undistort --undistort-strength 0.7
   ```

### Camera Distance

To move the camera closer to the pointcloud:

1. Use the `--zoom` option with a value greater than 1.0:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --zoom 2.0
   ```

2. Higher zoom values bring the camera closer to the pointcloud:
   ```bash
   python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural --zoom 3.0
   ```

### Incomplete Pointclouds

If your pointclouds appear incomplete or have too many points filtered out:

1. Use the `--keep-all-points` option to disable outlier filtering completely.
2. Adjust the outlier filtering parameters with `--outlier-nb-neighbors` and `--outlier-std-ratio`.
   - Increase `--outlier-std-ratio` to keep more points (less aggressive filtering).
   - Example: `--outlier-std-ratio 3.0` (default is 2.0)

### OpenGL Visualization Issues

If you encounter errors like "Failed to initialize GLEW" or "NoneType object has no attribute 'point_size'", it means your environment has limited OpenGL support. Use the matplotlib-based visualization tools instead:

```bash
python visualize_pointcloud_matplotlib.py --input path_to_pointcloud.ply --view natural
```

Or generate multiple views as images:

```bash
python generate_pointcloud_views.py --input path_to_pointcloud.ply --output-dir pointcloud_views
```

### Model Architecture Mismatch

If you see errors like "size mismatch" or "missing key in state_dict", it means there's a mismatch between the model architecture you're trying to use and the weights you're loading. To fix this:

1. Use the `--auto-detect-encoder` option to automatically detect the encoder type from the model filename.
2. Manually specify the correct encoder type with `--encoder` (vits, vitb, vitl, or vitg) that matches your model weights.

For example, if you're using `depth_anything_v2_vits.pth`, you should use `--encoder vits`.

### Other Common Issues

If the pointclouds still don't look right:

1. **Check the focal length**: The focal length parameters are crucial for correct 3D projection. Try different values or use the actual camera parameters if available.

2. **Check the depth scaling**: If using a relative depth model, the depth values are scaled. Try different `--max-depth` values.

3. **Filter outliers**: Use the `--filter-outliers` option with adjusted parameters to remove noise from the pointclouds.

4. **Downsample**: If the pointclouds are too dense, use the `--voxel-size` option to downsample them.

5. **Visualize a single frame**: Use the visualization scripts to inspect a single pointcloud in detail.

## Advanced Usage

For more advanced visualization options, check the additional scripts:
- `fly_through_pointcloud.py`: Basic fly-through video
- `fly_through_pointcloud_headless.py`: Fly-through video in headless environments
- `fly_through_pointcloud_advanced.py`: Advanced fly-through with smooth transitions
- `create_pointcloud_video.py`: Alternative visualization using matplotlib 