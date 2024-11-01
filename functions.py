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

def process_nd2_file(input_file, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels):
    """
    Processes an ND2 file by iterating through views, channels, frames, z-planes, 
    adjusting brightness/contrast, and saving as JPEG.
    
    Parameters:
    - input_file: Path to the ND2 file.
    - output_path: Path where the output JPGs will be saved.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
    - z_range: Tuple indicating the range of z-planes to process (start_z, end_z) OR
               dictionary indicating the range of z-planes to process for each channel
               e.g.: z_range = {1:(3,4), 2:(1,5)}
    - view_range: Tuple indicating the range of views (also called "Series") to process (start_view, end_view).
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    
    Returns:
    - None
    """
    with ND2Reader(input_file) as nd2:
        print(f"Processing file: {input_file}")
        axes = nd2.axes
        z_exists = 'z' in axes
        v_exists = 'v' in axes
        print("Available axes:", axes)
        
        if v_exists:
            nd2.bundle_axes = 'vtzcyx' if z_exists else 'vtyxc'
        else:
            nd2.bundle_axes = 'tzcyx' if z_exists else 'tyxc'

        # Iterate through view (v), channel (c), frame (t), and z-plane (z) as needed
        for v in range(view_range[0], view_range[1] + 1):
            if v_exists:
                nd2.default_coords['v'] = v - 1  # Set view if available

            for c in range(channel_range[0], channel_range[1] + 1):
                for f in range(frame_range[0], frame_range[1] + 1):

                    # Different z range for each channel
                    if isinstance(z_range, tuple):
                        z_range_local = range(z_range[0], z_range[1] + 1)
                    if isinstance(z_range, dict):
                        z_range_local = range(z_range.get(c)[0], z_range.get(c)[1] + 1)

                    for z in z_range_local:
                        nd2.default_coords['t'] = f - 1  # Frame (time point)
                        nd2.default_coords['c'] = c - 1  # Channel
                        if z_exists:
                            nd2.default_coords['z'] = z - 1  # Z-plane

                        # Retrieve the 2D frame as a NumPy array
                        if z_exists and v_exists:
                            image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
                        elif v_exists:
                            image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1))
                        elif z_exists:
                            image = np.array(nd2.get_frame_2D(c=c-1, t=f-1, z=z-1))
                        else:
                            image = np.array(nd2.get_frame_2D(c=c-1, t=f-1))

                        # Adjust brightness and contrast based on the channel's min/max values
                        min_value, max_value = contrast_of_channels.get(c) #contrast_of_channels.get(c, (0, 255))
                        image = adjust_brightness_contrast(image, min_value, max_value)

                        # Save the image as JPEG
                        filename = f"{os.path.basename(input_file).replace('.nd2', '')}"
                        if v_exists:
                            filename += f"_view_{v}"
                        filename += f"_channel_{c}_frame_{f}"
                        if z_exists:
                            filename += f"_z_{z}"
                        save_image(image, output_path, filename)



def process_nd2_folder(input_folder, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels):
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
            process_nd2_file(input_file, output_path, channel_range, frame_range, z_range, view_range, contrast_of_channels)

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
