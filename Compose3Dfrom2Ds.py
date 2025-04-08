import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import glob
from tqdm import tqdm

def load_dicom_folder(dicom_folder_path):
    """
    Load a series of DICOM files from a folder into a 3D volume
    
    Args:
        dicom_folder_path (str): Path to folder containing DICOM files
        
    Returns:
        numpy.ndarray: 3D CT volume
    """
    print(f"Loading DICOM files from {dicom_folder_path}...")
    
    # Get all DICOM files in the directory
    dicom_files = glob.glob(os.path.join(dicom_folder_path, '*.dcm'))
    
    if not dicom_files:
        # Try with uppercase extension
        dicom_files = glob.glob(os.path.join(dicom_folder_path, '*.DCM'))
        
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder_path}")
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    # Read slice position from DICOM files to sort them correctly
    slice_positions = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
            # Try different tags for slice position
            if hasattr(ds, 'SliceLocation'):
                slice_positions.append((float(ds.SliceLocation), dicom_file))
            elif hasattr(ds, 'ImagePositionPatient'):
                slice_positions.append((float(ds.ImagePositionPatient[2]), dicom_file))
            else:
                # If position information is not available, use instance number or filename
                if hasattr(ds, 'InstanceNumber'):
                    slice_positions.append((float(ds.InstanceNumber), dicom_file))
                else:
                    slice_positions.append((os.path.basename(dicom_file), dicom_file))
        except Exception as e:
            print(f"Warning: Could not read position from {dicom_file}: {e}")
            slice_positions.append((os.path.basename(dicom_file), dicom_file))
    
    # Sort files by slice position
    if all(isinstance(pos[0], (int, float)) for pos in slice_positions):
        slice_positions.sort(key=lambda x: float(x[0]))
    else:
        # If some positions are not numeric, sort by filename
        slice_positions.sort(key=lambda x: x[1])
        
    sorted_dicom_files = [file for _, file in slice_positions]
    
    # Read the first file to get dimensions
    first_slice = pydicom.dcmread(sorted_dicom_files[0])
    
    # Create empty volume
    volume = np.zeros((len(sorted_dicom_files), first_slice.Rows, first_slice.Columns))
    
    # Read all slices
    for i, dicom_file in enumerate(tqdm(sorted_dicom_files, desc="Loading DICOM slices")):
        try:
            ds = pydicom.dcmread(dicom_file)
            volume[i, :, :] = ds.pixel_array
            
            # Apply rescaling if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                volume[i, :, :] = volume[i, :, :] * slope + intercept
            else:
                # If no rescaling, just use the pixel data as is
                volume[i, :, :] = volume[i, :, :]
        except Exception as e:
            print(f"Error loading {dicom_file}: {e}")
            # Fill with zeros or previous slice if available
    
    # Convert HU units to density kg/m^3
    density = np.zeros_like(volume)
    
    mask_air = (volume < -120)  # Air density
    density[mask_air] = 1000  # Set negative values to 1000 [water density)
    
    mask_fat = (volume < 0) & (volume > -120)  # Fat density
    density[mask_fat] = 900 + 0.6 * (volume[mask_fat] + 500)
    
    mask_soft = (volume >= -120) & (volume < 100)  # Soft tissue density
    density[mask_soft] = 1000 + 1 * (volume[mask_soft] + 100)
    
    mask_bone = volume >= 100  # Bone density
    density[mask_bone] = 1100 + 0.7 * (volume[mask_bone] - 100)

    density[density > 3000] = 3000  # Cap density to 3000 kg/m^3
    density[density < 0] = 700  # Cap density to 700 kg/m^3
    
    # density = density / density.max()  # Normalize density to [0, 1]
    density = density.astype(np.float32)  # Convert to float32
    volume = density.astype(np.float32)  # Convert to float32
    
    print(f"DICOM volume loaded with shape: {volume.shape}")
    print(f"Volume min: {np.min(volume)}, max: {np.max(volume)}")
    print(f"Volume mean: {np.mean(volume)}, std: {np.std(volume)}")
    print(f"Volume dtype: {volume.dtype}")
    
    return density, sorted_dicom_files

def resize_with_tf(image, target_size):
    """
    Resize an image using TensorFlow instead of scikit-image
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size (height, width)
        
    Returns:
        numpy.ndarray: Resized image
    """
    # Add batch and channel dimensions if needed
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :, np.newaxis]
    elif len(image.shape) == 3 and image.shape[2] <= 4:  # Has channels already
        image = image[np.newaxis, :, :, :]
    else:  # 3D volume or similar
        image = image[:, :, :, np.newaxis]
    
    # Resize using TensorFlow
    resized = tf.image.resize(image, target_size, method='bilinear')
    
    # Convert back to numpy and remove extra dimensions
    resized = resized.numpy()
    
    # Remove batch and channel dimensions if they were added
    if len(resized.shape) == 4 and resized.shape[0] == 1 and resized.shape[3] == 1:
        resized = resized[0, :, :, 0]
    elif len(resized.shape) == 4 and resized.shape[0] == 1:
        resized = resized[0]
    elif len(resized.shape) > 2 and resized.shape[-1] == 1:
        resized = resized[..., 0]
    
    return resized

def crop_with_tf(slice_data, target_size, transducer_param=None):
    """
    Crop an image using TensorFlow around the zone where the transducer is placed
    and the focal point.
    This function crops the image to the target size by centering the crop around the
    transducer position.
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size (height, width)
    Returns:
        numpy.ndarray: Cropped image
    """
    
    height, width = slice_data.shape
    center__crop_x = (transducer_param.get("position")[0] - 40)
    center__crop_y = (transducer_param.get("position")[1] + transducer_param.get("focal_point")[1]) // 2
    
    # Ensure the center is within bounds
    center__crop_x = min(max(center__crop_x, 0), width - 1)
    center__crop_y = min(max(center__crop_y, 0), height - 1)
    
    # Calculate crop dimensions
    target_height, target_width = target_size
    start_x = max(center__crop_x - target_width // 2, 0)
    start_y = max(center__crop_y - target_height // 2, 0)
    end_x = min(start_x + target_width, width)
    end_y = min(start_y + target_height, height)
    cropped = slice_data[start_y:end_y, start_x:end_x]
    return cropped

def create_focused_transducer_mask(slice_data, focal_point=None, transducer_position=None, 
                                  radius=40, curvature=0.7, thickness=5, beam_width=10):
    """
    Create a bowl-shaped transducer mask with a focal beam
    
    Args:
        slice_data (numpy.ndarray): 2D CT slice
        focal_point (tuple): (x, y) position of the focal point, if None will be auto-detected
        transducer_position (tuple): (x, y) position of the transducer center, if None will be auto-detected
        radius (int): Radius of the transducer in pixels
        curvature (float): How curved the bowl is (0-1, where 1 is a half circle)
        thickness (int): Thickness of the transducer in pixels
        beam_width (int): Width of the focal beam
        
    Returns:
        numpy.ndarray: Binary mask with the bowl-shaped transducer and focal beam
    """
    height, width = slice_data.shape
    mask = np.zeros_like(slice_data, dtype=np.float32)
    
    # Create threshold for edge detection
    threshold = 1200
    binary = (slice_data > threshold).astype(np.uint8)
    
    # Auto-detect transducer position if not provided
    if transducer_position is None:
        # Find the right edge of the head
        mid_row = height // 2
        right_edge = width // 4
        for col in range(width-1, width//2, -1):
            if binary[mid_row, col] > 0:
                right_edge = col
                break
        
        # Position the transducer just outside the head on the right side
        transducer_position = (right_edge + thickness, mid_row)
    
    # Auto-detect focal point if not provided
    if focal_point is None:
        # Find a point likely to be inside the brain - set to 1/3 from right to left
        brain_x = transducer_position[0] - radius
        brain_y = transducer_position[1]  # Same height as transducer
        
        # Search for brain tissue in this approximate area
        focal_point = (brain_x, brain_y)
               
        # If not found, just use an estimated position
        if focal_point is None:
            focal_point = (transducer_position[0] - radius, transducer_position[1])
    
    center_x, center_y = transducer_position
    focal_x, focal_y = focal_point
    
    # Create the bowl by drawing a series of arcs at radius = radius +n1,n2,n3,...,n4
    for n in range(thickness):
        # Calculate the current radius
        current_radius = radius + n
        
        # Create an arc for the outer edge
        for angle in np.linspace(-np.pi * curvature / 2, np.pi * curvature / 2, num=1000):
            arc_x = int(focal_x + current_radius * np.cos(angle))
            arc_y = int(focal_y + current_radius * np.sin(angle))
            
            # Check bounds
            if 0 <= arc_x < width and 0 <= arc_y < height:
                mask[arc_y, arc_x] = 1.0
                # # Draw the inner edge of the bowl
                # inner_radius = current_radius - thickness
                # inner_arc_x = int(center_x + inner_radius * np.cos(angle))
                # inner_arc_y = int(center_y + inner_radius * np.sin(angle))
                # if 0 <= inner_arc_x < width and 0 <= inner_arc_y < height:
                #     mask[inner_arc_y, inner_arc_x] = 1.0
    
    return mask, focal_point, transducer_position

def create_mask_for_slice(slice_data, transducer_params=None):
    """
    Create a mask for a CT slice with a focused transducer
    
    Args:
        slice_data (numpy.ndarray): 2D CT slice
        transducer_params (dict): Parameters for the transducer, if None will use defaults
        
    Returns:
        numpy.ndarray: Binary mask for the slice with transducer
    """
    if transducer_params is None:
        transducer_params = {
            'position': None,  # Auto-detect
            'focal_point': None,  # Auto-detect
            'radius': 40,
            'curvature': 0.7,
            'thickness': 5,
            'beam_width': 10
        }
    
    mask, focal_point, transducer_position = create_focused_transducer_mask(
        slice_data,
        focal_point=transducer_params.get('focal_point'),
        transducer_position=transducer_params.get('position'),
        radius=transducer_params.get('radius'),
        curvature=transducer_params.get('curvature'),
        thickness=transducer_params.get('thickness'),
        beam_width=transducer_params.get('beam_width')
    )
    
    # Update the transducer parameters with detected positions
    transducer_params['position'] = transducer_position
    transducer_params['focal_point'] = focal_point
    
    return mask, transducer_params

def generate_3d_intensity(dicom_folder, model_path, output_path, num_slices=20, resize_dims=(128, 128), transducer_params=None):
    """
    Generate 3D intensity distribution from DICOM folder with a focused transducer mask
    
    Args:
        dicom_folder (str): Path to folder containing DICOM files
        model_path (str): Path to the trained generator model
        output_path (str): Path to save the results
        num_slices (int): Number of slices to process
        resize_dims (tuple): Dimensions to resize images to for the generator
        transducer_params (dict): Parameters for the transducer
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load generator model
    print(f"Loading generator model from {model_path}...")
    generator = tf.keras.models.load_model(model_path)
    
    # Load DICOM volume
    volume, dicom_files = load_dicom_folder(dicom_folder)
    
    # Extract slices
    depth = volume.shape[0]
    indices = np.linspace(100 - num_slices / 2, 100 + num_slices / 2, num_slices, dtype=int)
    
    # Initialize the 3D distribution
    distribution_3d = np.zeros((num_slices, *(resize_dims)))
    
    # Find best slice for transducer positioning (middle of volume)
    middle_slice = volume[100, :, :]
    
    # Auto-detect transducer position if not provided
    if transducer_params is None:
        transducer_params = {
            'position': None,  # Auto-detect
            'focal_point': None,  # Auto-detect
            'radius': min(volume.shape[1], volume.shape[2]) // 6,  # Scale with image size
            'curvature': 0.7,
            'thickness': 5,
            'beam_width': 10
        }
    
    # For demonstration, create a test mask and save it
    test_mask, updated_params = create_mask_for_slice(middle_slice, transducer_params)
    transducer_params = updated_params  # Update with detected positions
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(middle_slice, cmap='gray')
    plt.title('CT Slice')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(test_mask, cmap='gray')
    plt.title('Focused Transducer Mask')
    plt.axis('off')
    
    # Mark the focal point and transducer position
    if transducer_params['focal_point'] is not None:
        fx, fy = transducer_params['focal_point']
        plt.plot(fx, fy, 'ro', markersize=8)
        
    if transducer_params['position'] is not None:
        tx, ty = transducer_params['position']
        plt.plot(tx, ty, 'bo', markersize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'transducer_mask_example.png'), dpi=300)
    plt.close()
    
    # Create a 3D visualization of the planned transducer position
    print("Planning 3D transducer positioning...")
    
    # Process each slice
    intensity_slices = []
    
    for i, idx in enumerate(tqdm(indices, desc="Processing slices")):
        # Get the CT slice
        ct_slice = volume[idx, :, :]
        ct_slice = ct_slice / np.max(ct_slice)  # Normalize to [0, 1]
        
        # Create transducer mask
        mask, _ = create_mask_for_slice(ct_slice, transducer_params)
        
        # # Resize inputs to expected dimensions using TensorFlow
        # ct_resized = resize_with_tf(ct_slice, resize_dims)
        # mask_resized = resize_with_tf(mask, resize_dims)
        
        # Crop the slice around the transducer position
        ct_resized = resize_with_tf(crop_with_tf(ct_slice, resize_dims, transducer_param=transducer_params), resize_dims)
        mask_resized = resize_with_tf(mask, resize_dims)
        
        # Normalize to [-1, 1]
        ct_normalized = (ct_resized - 0.5) * 2
        
        # Stack inputs
        input_data = np.stack([mask_resized, ct_normalized], axis=-1)
        input_tensor = tf.convert_to_tensor(input_data[np.newaxis, ...], dtype=tf.float32)
        
        # Generate intensity
        generated = generator(input_tensor, training=False)
        intensity = generated.numpy()[0, :, :, 0]
        
        # Denormalize to [0, 1]
        intensity = intensity * 0.5 + 0.5
        
        intensity_slices.append(intensity)
        
        # Place in 3D volume
        distribution_3d[i, :, :] = intensity
        
        # Create visualization for this slice
        if i % 4 == 0 or i == len(indices) - 1:  # Visualize every 4th slice and the last one
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(ct_normalized, cmap='gray')
            plt.title(f'CT Slice {idx}')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask_resized, cmap='viridis')
            plt.title('Transducer Mask')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(intensity, cmap='hot')
            plt.title('Generated Intensity')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'slice_{idx}_process.png'))
            plt.close()
    
    # Interpolate between slices
    print("Interpolating between slices...")
    z_coords = indices
    full_z_coords = np.arange(num_slices)
    
    for x in tqdm(range(resize_dims[0]), desc="Interpolating x-dimension"):
        for y in range(resize_dims[1]):
            values = np.array([distribution_3d[i, x, y] for i in range(num_slices)])
            distribution_3d[:, x, y] = np.interp(full_z_coords, z_coords, values)
    
    # Save the full 3D intensity distribution
    np.save(os.path.join(output_path, 'intensity_distribution_3d.npy'), distribution_3d)
    
    # Create visualizations of the 3D distribution
    print("Creating visualizations...")
    
    # Central orthogonal slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view
    im0 = axes[0].imshow(distribution_3d[num_slices//2, :, :], cmap='hot')
    axes[0].set_title(f'Axial (z={num_slices//2})')
    axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Coronal view - mark focal point and transducer position
    im1 = axes[1].imshow(distribution_3d[:, resize_dims[0]//2, :], cmap='hot')
    axes[1].set_title(f'Coronal (y={resize_dims[0]//2})')
    axes[1].set_axis_off()
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Sagittal view
    im2 = axes[2].imshow(distribution_3d[:, :, resize_dims[1]//2], cmap='hot')
    axes[2].set_title(f'Sagittal (x={resize_dims[1]//2})')
    axes[2].set_axis_off()
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'orthogonal_views.png'), dpi=300)
    plt.close()
    
    # Create a montage of axial slices
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle('Axial Slices of Generated 3D Intensity Distribution', fontsize=16)
    
    for i in range(rows * cols):
        if i < num_slices:
            slice_idx = i * num_slices // (rows * cols)
            ax = axes[i // cols, i % cols]
            ax.imshow(distribution_3d[slice_idx, :, :], cmap='hot')
            ax.set_title(f'Slice {slice_idx}')
            ax.set_axis_off()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_path, 'axial_montage.png'), dpi=300)
    plt.close()
    
    # Create 3D visualization with overlay - using maximum intensity projection
    print("Creating maximum intensity projections...")
    
    # Overlay CT and intensity with different colormaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial MIP
    ct_axial = np.max(volume, axis=0)
    intensity_axial = np.max(distribution_3d, axis=0)
    
    axes[0].imshow(ct_axial, cmap='gray', alpha=0.7)
    axes[0].imshow(intensity_axial, cmap='hot', alpha=0.6)
    axes[0].set_title('Axial Maximum Intensity Projection')
    axes[0].set_axis_off()
    
    # Coronal MIP
    ct_coronal = np.max(volume, axis=1)
    intensity_coronal = np.max(distribution_3d, axis=1)
    
    axes[1].imshow(ct_coronal, cmap='gray', alpha=0.7)
    axes[1].imshow(intensity_coronal, cmap='hot', alpha=0.6)
    axes[1].set_title('Coronal Maximum Intensity Projection')
    axes[1].set_axis_off()
    
    # Sagittal MIP
    ct_sagittal = np.max(volume, axis=2)
    intensity_sagittal = np.max(distribution_3d, axis=2)
    
    axes[2].imshow(ct_sagittal, cmap='gray', alpha=0.7)
    axes[2].imshow(intensity_sagittal, cmap='hot', alpha=0.6)
    axes[2].set_title('Sagittal Maximum Intensity Projection')
    axes[2].set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mip_overlays.png'), dpi=300)
    plt.close()
    
    # Save a visualization of the intensity at the focal plane
    if transducer_params['focal_point'] is not None:
        focal_x, focal_y = transducer_params['focal_point']
        
        # Create focal plane visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(volume[depth//2, :, :], cmap='gray', alpha=0.6)
        plt.imshow(distribution_3d[num_slices//2, :, :], cmap='hot', alpha=0.7)
        plt.plot(focal_x, focal_y, 'ro', markersize=10)
        
        # Draw circle around focal point
        circle = plt.Circle((focal_x, focal_y), 10, color='r', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        
        plt.title('Focal Plane with Intensity Distribution')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'focal_plane.png'), dpi=300)
        plt.close()
    
    print(f"Processing complete. Results saved to {output_path}")
    
    return distribution_3d

if __name__ == "__main__":
    # Configuration
    DICOM_FOLDER = "CTHeadThin001"  # Path to folder containing DICOM files
    MODEL_PATH = "SavedModels/generator.keras"   # Path to trained generator model
    OUTPUT_PATH = "results/intensity_3d"    # Path to save results
    NUM_SLICES = 20                         # Number of slices to process
    
    # Custom transducer parameters
    # Leave position and focal point as None for auto-detection
    TRANSDUCER_PARAMS = {
        'position': None,           # Auto-detect the position (x, y) or set manually
        'focal_point': None,        # Auto-detect focal point (x, y) or set manually
        'radius': 550,               # Radius of the transducer in pixels
        'curvature': 0.2,           # How curved the transducer is (0-1)
        'thickness': 20,             # Thickness of the transducer in pixels
        'beam_width': 20            # Width of the ultrasound beam
    }
    
    # Generate 3D intensity distribution
    distribution_3d = generate_3d_intensity(
        DICOM_FOLDER, 
        MODEL_PATH, 
        OUTPUT_PATH, 
        NUM_SLICES,
        transducer_params=TRANSDUCER_PARAMS
    )