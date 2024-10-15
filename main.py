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

def process_nd2_file(input_file, output_path, channel_range, frame_range, contrast_of_channels):
    """
    Processes an ND2 file by iterating through channels and frames, adjusting brightness/contrast, and saving as JPEG.
    
    Parameters:
    - input_file: Path to the ND2 file.
    - output_path: Path where the output JPGs will be saved.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
    - contrast_of_channels: Dictionary with min/max brightness values for each channel.
    
    Returns:
    - None
    """
    with ND2Reader(input_file) as nd2:
        print(f"Processing file: {input_file}")
        print("Available axes:", nd2.axes)  # Display available axes in the ND2 file
        nd2.bundle_axes = 'tyxc'  # Set axes: time (t), y (height), x (width), channel (c)

        for c in range(channel_range[0], channel_range[1] + 1):
            for f in range(frame_range[0], frame_range[1] + 1):
                nd2.default_coords['t'] = f - 1  # Select frame (time point)
                nd2.default_coords['c'] = c - 1  # Select channel

                # Get the 2D frame as a NumPy array
                image = np.array(nd2.get_frame_2D(c=c-1, t=f-1))

                # Adjust brightness and contrast based on the channel's min/max values
                min_value, max_value = contrast_of_channels.get(c, (0, 255))  # Fallback to (0, 255) if no values specified
                image = adjust_brightness_contrast(image, min_value, max_value)

                # Save the image as JPEG
                filename = f"{os.path.basename(input_file).replace('.nd2', '')}_channel_{c}_frame_{f}"
                save_image(image, output_path, filename)

def process_nd2_folder(input_folder, output_path, channel_range, frame_range, contrast_of_channels):
    """
    Processes all ND2 files in a given folder.
    
    Parameters:
    - input_folder: Path to the folder containing ND2 files.
    - output_path: Path where the output JPGs will be saved.
    - channel_range: Tuple indicating the range of channels to process (start_channel, end_channel).
    - frame_range: Tuple indicating the range of frames to process (start_frame, end_frame).
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
            process_nd2_file(input_file, output_path, channel_range, frame_range, contrast_of_channels)

# Hauptteil des Programms
if __name__ == "__main__":
    # Pfade und Variablen definieren
    input_folder = "C:/Users/Kai_F/Documents/GitHub/carolyn-nd2/input"  # Ordner mit ND2-Dateien
    output_path = "C:/Users/Kai_F/Documents/GitHub/carolyn-nd2/output"  # Pfad zum Speichern der Ausgabedateien
    
    # Parameter für Kanal- und Frame-Intervalle (channel_range, frame_range)
    channel_range = (2, 3)  # Start and end channels (inclusive)
    frame_range = (19, 25)  # Start and end frames (inclusive)
    
    # Definiere Min- und Max-Werte für jeden Kanal
    contrast_of_channels = {
        2: (500, 800),  # Channel 2: Min=500, Max=800
        3: (300, 700),  # Channel 3: Min=300, Max=700
        # Weitere Kanäle können hier hinzugefügt werden
    }
    
    # Verarbeite alle ND2-Dateien im Input-Ordner
    process_nd2_folder(input_folder, output_path, channel_range, frame_range, contrast_of_channels)
