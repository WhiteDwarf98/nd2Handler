import numpy as np
from nd2reader import ND2Reader
from PIL import Image
import os

def adjust_brightness_contrast(image, min_value, max_value):
    """
    Adjusts the brightness and contrast of an image based on the given min and max values.
    
    Parameters:
    - image: NumPy array representing the image.
    - min_value: Minimum value for brightness adjustment.
    - max_value: Maximum value for brightness adjustment.
    
    Returns:
    - image: The adjusted image as a NumPy array.
    """
    image = np.clip(image, min_value, max_value)  # Clip values to the range [min_value, max_value]
    image = ((image - min_value) / (max_value - min_value) * 255).astype(np.uint8)  # Scale to 0-255
    return image

def save_image(image, output_path, filename):
    """
    Saves a NumPy array as a JPEG image.
    
    Parameters:
    - image: NumPy array representing the image.
    - output_path: Path where the image will be saved.
    - filename: The name of the file (without extension).
    
    Returns:
    - None
    """
    output_file = os.path.join(output_path, f"{filename}.jpg")
    img = Image.fromarray(image)
    img.save(output_file)
    print(f"Saved {output_file}")

def process_z_planes(nd2, z_range_local, v, c, f, contrast, use_mip, output_path, filename_base):
    """Handle processing of z-planes, either with MIP or exporting each plane separately."""
    if use_mip:
        # Generate MIP across the z range
        image_stack = [np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1)) for z in z_range_local]
        mip_image = maximum_intensity_projection(np.stack(image_stack))
        mip_image = adjust_brightness_contrast(mip_image, *contrast)
        
        filename = f"{filename_base}_MIP_z_{z_range_local[0]}-{z_range_local[-1]}"
        save_image(mip_image, output_path, filename)
        print(f"Saved {filename}")
    else:
        # Process each z-plane individually
        for z in z_range_local:
            nd2.default_coords['z'] = z - 1
            image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
            image = adjust_brightness_contrast(image, *contrast)
            
            filename = f"{filename_base}_z_{z}"
            save_image(image, output_path, filename)
            print(f"Saved {filename}")

def process_nd2_file(input_file, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels, use_mip):
    """
    Processes an ND2 file by iterating through views, channels, frames, and z-planes, 
    adjusting brightness/contrast, optionally applying Maximum Intensity Projection (MIP), 
    and saving as JPEG images.
    
    Parameters:
    - input_file: Path to the ND2 file.
    - output_path: Path where the output JPGs will be saved.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
    - z_range: Tuple or dict indicating the range of z-planes to process for each channel.
    - view_range: Tuple indicating the range of views to process (start_view, end_view).
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    - use_mip: Boolean indicating whether to apply Maximum Intensity Projection (MIP) over the z-range.
    
    Returns:
    - None
    """
    with ND2Reader(input_file) as nd2:
        print(f"Processing file: {input_file}")
        axes = nd2.axes
        z_exists = 'z' in axes
        v_exists = 'v' in axes
        print("Available axes:", axes)

        # Set bundle_axes based on available dimensions
        nd2.bundle_axes = 'vtzcyx' if v_exists and z_exists else 'vtyxc' if v_exists else 'tzcyx' if z_exists else 'tyxc'

        # Iterate over views, channels, and frames as needed
        for v in range(view_range[0], view_range[1] + 1):
            if v_exists:
                nd2.default_coords['v'] = v - 1  # Set view if available

            for c in range(channel_range[0], channel_range[1] + 1):
                contrast = contrast_of_channels.get(c, (0, 255))  # Retrieve contrast values for channel
                for f in range(frame_range[0], frame_range[1] + 1):
                    nd2.default_coords['t'] = f - 1
                    nd2.default_coords['c'] = c - 1
                    
                    # Define local z-range for each channel if z_range is a dictionary
                    z_range_local = range(z_range[0], z_range[1] + 1) if isinstance(z_range, tuple) else range(z_range.get(c)[0], z_range.get(c)[1] + 1)

                    # Base filename for each combination
                    filename_base = f"{os.path.basename(input_file).replace('.nd2', '')}"
                    if v_exists:
                        filename_base += f"_view_{v}"
                    filename_base += f"_channel_{c}_frame_{f}"

                    # Process z-planes with MIP or single slices
                    if z_exists:
                        process_z_planes(nd2, z_range_local, v, c, f, contrast, use_mip, output_path, filename_base)
                    else:
                        # Process 2D frames without z-dimension
                        image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1)) if v_exists else np.array(nd2.get_frame_2D(c=c-1, t=f-1))
                        image = adjust_brightness_contrast(image, *contrast)
                        save_image(image, output_path, filename_base)
                        print(f"Saved {filename_base}")

# def maximum_intensity_projection(image):
#     """
#     Perform Maximum Intensity Projection (MIP) over a specified range of z-planes.
    
#     Parameters:
#     - image: 3D NumPy array representing the image stack, with dimensions (z, y, x).
#     - z_range: Tuple specifying the range of z-planes to include in the MIP (start_z, end_z).
    
#     Returns:
#     - 2D NumPy array representing the MIP over the specified z-range.
#     """
#     start_z, end_z = z_range
    
#     # Slicing the image stack to the specified z-range
#     limited_stack = image[start_z:end_z + 1, :, :]
    
#     # Perform Maximum Intensity Projection on the limited z-range
#     mip_image = np.max(limited_stack, axis=0)
    
#     return mip_image

def maximum_intensity_projection(stack):
    """Compute the Maximum Intensity Projection (MIP) over the z-axis of a 3D stack."""
    return np.max(stack, axis=0)

def process_nd2_folder(input_folder, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels, use_mip):
    """
    Processes all ND2 files in a given folder.
    
    Parameters:
    - input_folder: Path to the folder containing ND2 files.
    - output_path: Path where the output JPGs will be saved.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
    - z_range: Tuple indicating the range of z-planes to process (start_z, end_z).
    - view_range: Tuple indicating the range of views (also called "Series") to process (start_view, end_view).
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    
    Returns:
    - None
    """
    # Make sure the output folder exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate through all ND2 files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.nd2'):
            input_file = os.path.join(input_folder, file)
            process_nd2_file(input_file, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels, use_mip)

# Main part of the program
if __name__ == "__main__":
    # Define paths and variables
    input_folder = "C:/Users/Kai_F/Documents/GitHub/carolyn-nd2/input"  # Folder with ND2 files
    output_path = "C:/Users/Kai_F/Documents/GitHub/carolyn-nd2/output"  # Path to save the output files
    
    # Channel, frame, and z-plane ranges
    channel_range = (1, 2)  # Start and end channels (inclusive)
    frame_range = (1,1)  # Start and end frames (inclusive)
    z_range = (1, 30)        # Start and end z-planes (inclusive)
    
    # Define min and max values for each channel
    contrast_of_channels = {
        2: (500, 800),  # Channel 2: Min=500, Max=800
        3: (300, 700),  # Channel 3: Min=300, Max=700
        # Add more channels if necessary
    }
    
    # Process all ND2 files in the input folder
    process_nd2_folder(input_folder, output_path, channel_range, frame_range, z_range, contrast_of_channels)
