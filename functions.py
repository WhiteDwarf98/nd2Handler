import numpy as np
from nd2reader import ND2Reader
from PIL import Image
import tifffile as tiff
import os
import pprint

def parse_range(range_string):
    """
    Parses a string representing ranges and returns a list of integers.
    
    Examples:
    - "1,2,3" -> [1, 2, 3]
    - "2-5,7,8" -> [2, 3, 4, 5, 7, 8]
    
    Parameters:
    - range_string (str): The input string representing the ranges. 
                          Only integers and the characters ',', '-' are allowed.
    
    Returns:
    - list[int]: A list of integers specified in the range string.
    - False: False in case of an error
    """    
    ranges = []
    for part in range_string.split(','):
        if '-' in part:  # Handle ranges like "2-5"
            start, end = map(int, part.split('-'))
            if start > end:
                print(f"Invalid range: '{part}'. Start must be less than or equal to end.")
                return False
            ranges.extend(range(start, end + 1))
        else:  # Handle single values like "7"
            try:
                ranges.append(int(part))
            except ValueError:
                print(f"Invalid number during parsing a string: '{part}'. Expected an integer.")
                return False
    
    return ranges

def get_contrast_for_channel(contrast_dict, channel):
    """
    Finds the contrast settings for a given channel based on a dictionary of ranges.
    
    Parameters:
    - contrast_dict (dict): Dictionary where keys are tuples of channel ranges (as integers) and values are contrast settings.
    - channel (int): The channel number to find the contrast for.
    
    Returns:
    - tuple: The contrast settings (min, max) for the given channel.
    """
    for key_range, contrast in contrast_dict.items():
        if channel in key_range:
            return contrast
    return (0, 255)  # Default contrast if channel is not found

def get_z_range_for_channel(z_range_dict, channel):
    """
    Finds the z-range for a given channel based on a dictionary of ranges.
    
    Parameters:
    - z_range_dict (dict): Dictionary where keys are tuples of channel ranges (as integers) and values are z-plane ranges (as lists).
    - channel (int): The channel number to find the z-range for.
    
    Returns:
    - list[int]: The z-plane range for the given channel.
    """
    for key_range, z_range in z_range_dict.items():
        if channel in key_range:
            return z_range
    return []  # Default empty range if channel is not found

def get_axes_info(nd2, do_print=True):
    """
    Returns the available axes in an ND2 file and the selectable range for each axis,
    starting from 1.

    Parameters:
    - nd2: Path to the ND2 file or nd2 object.
    - do_print: Boolean

    Returns:
    - A Tuple containing an array of available axes and a dictionary with axis names as keys and value ranges as tuples (start at 1),
    - or None, if do_print = True
    """
    
    def body(nd2):
        axes_info = {}

        # Get all axes available in the file
        available_axes = nd2.axes

        # Retrieve possible ranges for each available axis
        if 'c' in available_axes:
            axes_info['c'] = (1, nd2.sizes['c'])
        if 't' in available_axes:
            axes_info['t'] = (1, nd2.sizes['t'])
        if 'z' in available_axes:
            axes_info['z'] = (1, nd2.sizes['z'])
        if 'v' in available_axes:
            axes_info['v'] = (1, nd2.sizes['v'])
        axes_info['x'] = (1, nd2.sizes['x'])
        axes_info['y'] = (1, nd2.sizes['y'])

        # Display the range for each axis for user clarity
        if do_print:
            print("Available axes in nd2-file:")
            print(available_axes)
            for axis, (min_val, max_val) in axes_info.items():
                print(f"Axis '{axis}': Range {min_val} to {max_val}")
            return

        return available_axes, axes_info
    
    if isinstance(nd2, str):
        with ND2Reader(nd2) as nd2_object:
            return body(nd2_object)
    else:
        return body(nd2)

def validate_user_inputs(nd2, channel_range, frame_range, z_range, view_range, contrast_of_channels, file_format):
    """
    Validates user inputs to ensure they are within the valid range and that specified axes exist.
    
    Parameters:
    - nd2: Opened ND2 file object.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
    - z_range: Tuple or dictionary indicating the range of z-planes to process.
    - view_range: Tuple indicating the range of views to process (start_view, end_view).
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    
    Returns:
    - bool: True if all inputs are valid, False otherwise.
    """
    valid = True
    available_axes, axis_ranges = get_axes_info(nd2, False)
    var_dict = {
        'c': channel_range,
        't': frame_range,
        'z': z_range,
        'v': view_range,
        #'contrast': contrast_of_channels,
    }

    # Helper function to check ranges
    def check_range(axis_name, chosen_range):
        nonlocal axis_ranges
        nonlocal valid
        axis_range = axis_ranges[axis_name]

        # Check tuples
        if isinstance(chosen_range, tuple):
            check_range_single(axis_name, axis_range, chosen_range)
        
        # Check dictionaries
        elif isinstance(chosen_range, dict):
            if axis_name != 'z':
                print(f"Axis {axis_name} should be a tuple, not a dictionary!")
                valid = False
            else:
                check_dict(axis_name, axis_range, chosen_range)
        else:
            print(f"{axis_name} should be a tuple of (min, max), or a dictionary of tuples.")
    
    def check_dict(axis_name, axis_range, range_dict, check_tuples=True):
        nonlocal valid
        for channel, s_chosen_range, in range_dict.items():
            if not (channel_range[0] <= channel <= channel_range[1]):
                print(f"Channel {channel} chosen in the dict for axis {axis_name} is out of bounds.")
                print(f"Valid c values are from {axis_range[0]} to {axis_range[1]}.")
                valid = False
            if check_tuples:
                check_range_single(axis_name, axis_range, s_chosen_range)
        
    def check_range_single(axis_name, axis_range, chosen_range):
        nonlocal valid
        if chosen_range[0] < axis_range[0] or chosen_range[1] > axis_range[1]:
            print(f"Error: {axis_name} range ({chosen_range[0]}, {chosen_range[1]}) is out of bounds. "
                  f"Valid {axis_name} values are from {axis_range[0]} to {axis_range[1]}.")
            valid = False
        if chosen_range[0] > chosen_range[1]:
            print(f"Error: For axis {axis_name}, the minimum value should be less than the maximum.")
            valid = False

    # Check if given ranges are correct
    for axis in available_axes:
        if axis in ['x', 'y']:
            continue
        check_range(axis, var_dict[axis])
    
    # Check channels given in the contrast dict:
    check_dict("contrast", axis_ranges['c'], contrast_of_channels, False)

    # Check if there are unnecessary user inputs
    for axis, chosen_range in var_dict.items():
        if axis not in available_axes and chosen_range:
            print(f"Warning: Channel axis ({axis}) not found in this ND2 file. Ignoring {axis}-range.")

    # Check for file_format
    if not (file_format in ['jpg', 'tif']):
        print(f"Error: Invalid file_format '{file_format}'! Must be 'jpg' or 'tif'.")
        valid = False

    if not valid:
        print("===================================")
        print("Please mind the available ranges:")
        pprint.pprint(axis_ranges)
    return valid

def adjust_brightness_contrast(image, min_value, max_value, offset):
    """
    Adjusts the brightness and contrast of an image based on the given min and max values.
    
    Parameters:
    - image: NumPy array representing the image.
    - min_value: Minimum value for brightness adjustment.
    - max_value: Maximum value for brightness adjustment.
    - offset: The offset value to add to each pixel.
    
    Returns:
    - image: The adjusted image as a NumPy array.
    """
    image = image + offset  # Offset adjustment
    image = np.clip(image, min_value, max_value)  # Clip values to the range [min_value, max_value]
    image = ((image - min_value) / (max_value - min_value) * 255).astype(np.uint8)  # Scale to 0-255
    return image

def save_image(image, output_path, filename, file_format):
    """
    Saves a NumPy array as an image file in the specified format.
    
    Parameters:
    - image: NumPy array representing the image or image stack.
    - output_path: Path where the image will be saved.
    - filename: The name of the file (without extension).
    - file_format: File format to save the image as ('jpg' or 'tif').
    
    Returns:
    - None
    """
    output_file = os.path.join(output_path, f"{filename}.{file_format}")
    if file_format == 'jpg':
        img = Image.fromarray(image)
        img.save(output_file)
    elif file_format == 'tif':
        tiff.imwrite(output_file, image, photometric='minisblack')
    print(f"Saved {output_file}")

def process_z_planes(nd2, z_list, v, c, f, contrast, use_mip, stack_z_to_tiff, output_path, input_file, file_format, offset):
    """
    Handles processing of z-planes, either with MIP or exporting each plane separately, 
    with optional stacking of z-planes in a single TIFF file.
    
    Parameters:
    - nd2: ND2Reader object.
    - z_list: List of z-planes to process.
    - v: Current view index.
    - c: Current channel index.
    - f: Current frame index.
    - contrast: Tuple (min, max) for brightness adjustment.
    - use_mip: Boolean, whether to use Maximum Intensity Projection (MIP).
    - stack_z_to_tiff: Boolean, whether to stack z-planes in a TIFF file.
    - output_path: Path to save the output images.
    - input_file: The name of the input file.
    - file_format: Output file format ('jpg' or 'tif').
    - offset: Integer offset for brightness adjustment.
    
    Returns:
    - None
    """
    if use_mip:
        # Generate MIP across the z range
        image_stack = [np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1)) for z in z_list]
        mip_image = maximum_intensity_projection(np.stack(image_stack))
        mip_image = adjust_brightness_contrast(mip_image, *contrast, offset)
        
        filename = f"{input_file}_v{v}_c{c}_f{f}_MIP_z_{'-'.join(map(str, z_list))}"
        save_image(mip_image, output_path, filename, file_format)
    elif stack_z_to_tiff and file_format == 'tif':
        # Stack z-planes into a single TIFF file
        image_stack = []
        for z in z_list:
            image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
            image = adjust_brightness_contrast(image, *contrast, offset)
            image_stack.append(image)
        image_stack = np.stack(image_stack)
        filename = f"{input_file}_v{v}_c{c}_f{f}_z_{'-'.join(map(str, z_list))}"
        save_image(image_stack, output_path, filename, file_format)
    else:
        # Process each z-plane individually
        for z in z_list:
            nd2.default_coords['z'] = z - 1
            image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
            image = adjust_brightness_contrast(image, *contrast, offset)
            
            filename = f"{input_file}_v{v}_c{c}_f{f}_z_{z}"
            save_image(image, output_path, filename, file_format)

def process_nd2_file(
    input_file,
    output_path,
    channel_list,
    frame_list,
    view_list,
    contrast_of_channels,
    z_range,
    use_mip,
    stack_z_to_tiff,
    file_format,
    offset
    ):
    """
    Processes an ND2 file by iterating through views, channels, frames, and z-planes,
    adjusting brightness/contrast, optionally applying Maximum Intensity Projection (MIP),
    and saving images in the specified format.
    
    Parameters:
    - input_file: Path to the ND2 file.
    - output_path: Path where the output images will be saved.
    - channel_list: List of channels to process.
    - frame_list: List of frames to process.
    - view_list: List of views to process.
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    - z_range: Dictionary where keys are channel ranges and values are z-plane ranges.
    - use_mip: Boolean indicating whether to apply Maximum Intensity Projection (MIP) over the z-range.
    - stack_z_to_tiff: Boolean indicating whether to stack the z-range in a TIFF file.
    - file_format: Desired file format ('jpg' or 'tif').
    - offset: An integer to adjust the brightness by adding this value to each pixel.
    
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
        axes_to_bundle = ''.join(axis for axis in ['v', 't', 'z', 'c', 'y', 'x'] if axis in nd2.axes)
        nd2.bundle_axes = axes_to_bundle

        # Iterate over views, channels, and frames as needed
        for v in view_list:
            if v_exists:
                nd2.default_coords['v'] = v - 1  # Set view if available

            for c in channel_list:
                contrast = get_contrast_for_channel(contrast_of_channels, c)
                z_list = get_z_range_for_channel(z_range, c)
                for f in frame_list:
                    nd2.default_coords['t'] = f - 1
                    nd2.default_coords['c'] = c - 1

                    # Process z-planes with MIP, TIFF stacking, or as individual planes
                    if z_exists:
                        process_z_planes(nd2, z_list, v, c, f, contrast, use_mip, stack_z_to_tiff, output_path, input_file, file_format, offset)
                    else:
                        # Process 2D frames without z-dimension
                        image = np.array(nd2.get_frame_2D(v=v-1, c=c-1, t=f-1)) if v_exists else np.array(nd2.get_frame_2D(c=c-1, t=f-1))
                        image = adjust_brightness_contrast(image, *contrast, offset)
                        save_image(image, output_path, f"{input_file}_v{v}_c{c}_f{f}", file_format)

def maximum_intensity_projection(stack):
    """Compute the Maximum Intensity Projection (MIP) over the z-axis of a 3D stack."""
    return np.max(stack, axis=0)

def process_nd2_folder(
    input_folder,
    output_path,
    channel_range,
    frame_range,
    z_range,
    view_range,
    contrast_of_channels,
    use_mip,
    stack_z_to_tiff,
    file_format,
    offset):
    """
    Processes all ND2 files in a given folder.
    
    Parameters:
    - input_folder: Path to the folder containing ND2 files.
    - output_path: Path where the output images will be saved.
    - channel_range: String representing the range of channels to process (e.g., "1-2").
    - frame_range: String representing the range of frames to process (e.g., "1").
    - z_range: Dictionary where keys are channel ranges (as strings) and values are z-plane ranges (as strings).
    - view_range: String representing the range of views to process (e.g., "1,3,5").
    - contrast_of_channels: Dictionary with min/max brightness values for each channel (keys as strings).
    - use_mip: Boolean indicating whether to apply Maximum Intensity Projection (MIP).
    - stack_z_to_tiff: Boolean indicating whether to stack z-planes in a TIFF file.
    - file_format: Desired file format ('jpg' or 'tif').
    - offset: An integer to adjust the brightness by adding this value to each pixel.
    
    Returns:
    - None
    """
    # Parse ranges
    channel_list = parse_range(channel_range)
    frame_list = parse_range(frame_range)
    view_list = parse_range(view_range)

    if not all([channel_list, frame_list, view_list]):
        print("Error: Invalid range strings provided.")
        return

    # Parse contrast_of_channels and z_range keys and values
    contrast_of_channels = {
        tuple(parse_range(k)): v for k, v in contrast_of_channels.items()
    }
    z_range = {
        tuple(parse_range(k)): parse_range(v) for k, v in z_range.items()
    }

    # Make sure the output folder exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate through all ND2 files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".nd2"):
            process_nd2_file(
                os.path.join(input_folder, file),
                output_path,
                channel_list,
                frame_list,
                view_list,
                contrast_of_channels,
                z_range,
                use_mip,
                stack_z_to_tiff,
                file_format,
                offset
            )

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
