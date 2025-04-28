import numpy as np
import os
from nd2reader import ND2Reader
from PIL import Image
import tifffile as tiff
import pprint

class nd2Handler: 
    def __init__(
            self,
            channel_range,
            frame_range,
            z_range,
            view_range,
            contrast_of_channels,
            file_format = 'jpg',
            offset = 0,
            stack_z_to_tiff = False,
            do_intensity_projection = None,
            ):
        
        self.input_folder = None
        self.output_path = None
        self.current_file = None # Path to the file
        self.current_nd2 = None # nd2 object
        self.current_image = None # image as numpy array
        self.offset = offset
        self.do_intensity_projection = do_intensity_projection # None max
        self.stack_z_to_tiff = stack_z_to_tiff
        self.file_format = file_format

        ## parsing
        self.channel_list = self.parse_range(channel_range)
        self.frame_list = self.parse_range(frame_range)
        self.view_list = self.parse_range(view_range)
        # e.g. {"1-2": (500, 800)} -> {1: (500, 800), 2: (500, 800)}
        self.contrast_of_channels = {
            ch: v
            for k, v in contrast_of_channels.items()
            for ch in self.parse_range(k)
        }
        # e.g. {"1-2": "2,4"} -> {1:[2, 4], 2:[2, 4]}
        self.z_list = {
            ch: self.parse_range(v)
            for k, v in z_range.items()
            for ch in self.parse_range(k)
        }

    def validate_user_inputs(self):
        """
        Validates user inputs to ensure they are within the valid range and that specified axes exist.
        
        Returns:
        - bool: True if all inputs are valid, False otherwise.
        """
        valid = True
        available_axes, axis_ranges = get_axes_info(self.current_nd2, False)
        chosen_lists = {
            'c': self.channel_list,
            't': self.frame_list,
            'z': self.z_list,
            'v': self.view_list,
            #'contrast': contrast_of_channels,
        }

        # Check every axis
        for axis in available_axes:
            if axis in ['x', 'y']:
                continue
            axis_list = list(range(axis_ranges[axis][0], axis_ranges[axis][1] + 1))

            if isinstance(chosen_lists[axis], list):
                for i in chosen_lists[axis]:
                    if i not in axis_list:
                        print(f"Error: {axis} is out of bounds. ")
                        print(f"Valid {axis} values are from {axis_ranges[axis][0]} to {axis_ranges[axis][1]}.")
                        valid = False
            elif isinstance(chosen_lists[axis], dict):
                for channel, liste in chosen_lists[axis].items():
                    if channel not in axis_ranges['c']:
                        print(f"Error: Channel {channel} in axis {axis} is out of bounds.")
                        print(f"Valid c values are from {axis_ranges['c'][0]} to {axis_ranges['c'][1]}.")
                    for i in liste:
                        if i not in axis_list:
                            print(f"Error: {axis} is out of bounds. ")
                            print(f"Valid {axis} values are from {axis_ranges[axis][0]} to {axis_ranges[axis][1]}.")
                            valid = False

        # Check channels given in the contrast dict
        for channel, contrast in self.contrast_of_channels.items():
            if channel not in axis_ranges['c']:
                print(f"Error: Channel {channel} in axis {axis} is out of bounds.")
                print(f"Valid c values are from {axis_ranges['c'][0]} to {axis_ranges['c'][1]}.")

        # Check for file_format
        if not (self.file_format in ['jpg', 'tif']):
            print(f"Error: Invalid file_format '{self.file_format}'! Must be 'jpg' or 'tif'.")
            valid = False

        if not valid:
            print("===================================")
            print("Please mind the available ranges:")
            pprint.pprint(axis_ranges)
        return valid

    def process_folder(self, input_folder, output_path):
        self.input_folder = input_folder
        self.output_path = output_path
        
        # Make sure the output folder exists
        os.makedirs(output_path, exist_ok=True)
        
        for file in os.listdir(self.input_folder):
            if file.endswith(".nd2"):
                self.current_file = file
                self.process_file()

    def process_file(self):

        def process_z_planes(v, c, f):
            if 'z' not in self.current_nd2.axes:
                # Process 2D frames without z-dimension
                self.current_image = np.array(self.current_nd2.get_frame_2D(v=v-1, c=c-1, t=f-1)) if 'v' in self.current_nd2.axes else np.array(self.current_nd2.get_frame_2D(c=c-1, t=f-1))
                self.adjust_brightness_contrast(self.contrast_of_channels[c])
                self.save_image(f"{self.current_file}_v{v}_c{c}_f{f}")
            elif self.do_intensity_projection is not None:
                # Generate MIP across the z range
                self.current_image = np.stack([np.array(self.current_nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1)) for z in self.z_list[c]])
                self.intensity_projection()
                self.adjust_brightness_contrast(self.contrast_of_channels[c])            
                self.save_image(f"{self.current_file}_v{v}_c{c}_f{f}_IP_z_{'-'.join(map(str, self.z_list[c]))}")
            elif self.stack_z_to_tiff and self.file_format == 'tif':
                # Stack z-planes into a single TIFF file
                image_stack = []
                for z in self.z_list[c]:
                    self.current_image = np.array(self.current_nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
                    self.adjust_brightness_contrast(self.contrast_of_channels[c])
                    image_stack.append(self.current_image)
                self.current_image = np.stack(image_stack)
                self.save_image(f"{self.current_file}_v{v}_c{c}_f{f}_z_{'-'.join(map(str, self.z_list[c]))}")
            else:
                # Process each z-plane individually
                for z in self.z_list[c]:
                    self.current_nd2.default_coords['z'] = z - 1
                    self.current_image = np.array(self.current_nd2.get_frame_2D(v=v-1, c=c-1, t=f-1, z=z-1))
                    self.adjust_brightness_contrast(self.contrast_of_channels[c])
                    self.save_image(f"{self.current_file}_v{v}_c{c}_f{f}_z_{z}")


        self.current_nd2 = ND2Reader(os.path.join(self.input_folder, self.current_file))
        
        if not self.validate_user_inputs():
            print(f"{self.current_file} skipped.")
            print("="*40)
            return False
        
        print(f"Processing file: {self.current_file}")
        print("Available axes:", self.current_nd2.axes)

        # Set bundle_axes based on available dimensions - "axes_to_bundle"
        self.current_nd2.bundle_axes = ''.join(axis for axis in ['v', 't', 'z', 'c', 'y', 'x'] if axis in self.current_nd2.axes)

        # Iterate over views, channels, and frames as needed
        for v in self.view_list:
            if 'v' in self.current_nd2.axes:
                self.current_nd2.default_coords['v'] = v - 1  # Set view if available

            for c in self.channel_list:
                for f in self.frame_list:
                    self.current_nd2.default_coords['t'] = f - 1
                    self.current_nd2.default_coords['c'] = c - 1
                    process_z_planes(v,c,f)


    def save_image(self, filename):
        """
        Saves a NumPy array as an image file in the specified format.
        """
        if self.output_path is None:
            print("Please setup 'self.output_path' first.")
            return
        if self.current_image is None:
            print("Please generate self.current_image' first.")
            return
        
        if '.nd2' in filename:
            filename = filename.replace('.nd2', '')
        
        output_file = os.path.join(self.output_path, f"{filename}.{self.file_format}")
        if self.file_format == 'jpg':
            Image.fromarray(self.current_image).save(output_file)
        elif self.file_format == 'tif':
            tiff.imwrite(output_file, self.current_image, photometric='minisblack')
        print(f"Saved {output_file}")

    def adjust_brightness_contrast(self, contrast):
        if self.current_image is None:
            print("No current image loaded. Please execute 'process_file()' first.")
            return
        self.current_image = self.current_image + self.offset  # Offset adjustment
        self.current_image = np.clip(self.current_image, *contrast)  # Clip values to the range [min_value, max_value]
        self.current_image = ((self.current_image - contrast[0]) / (contrast[1] - contrast[0]) * 255).astype(np.uint8)  # Scale to 0-255, contrast[0] means min_value, contrast[1] max_value
        return self.current_image

    def intensity_projection(self):
        match self.do_intensity_projection:
            case "max":
                self.current_image = np.max(self.current_image, axis=0)
            case "average":
                self.current_image = np.mean(self.current_image, axis=0).astype(self.current_image.dtype)
            case None:
                return None
        return self.current_image


    def parse_range(self, range_string):
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
    
def get_axes_info(nd2 = None, do_print=True):
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