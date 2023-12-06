from videoreader import VideoReader
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from skimage import exposure
import os
from tqdm import tqdm
    
def get_eye_value(string):
    """
    Get the eye value based on the input string.
    
    Args:
        string (str): Input string.
        
    Returns:
        int: Eye value (0 if "IZQ" is in the string, 1 otherwise).
    """
    return 0 if "IZQ" in string else 1


def get_name_number(string):
    """
    Get the modified name based on the input string.
    
    Args:
        string (str): Input string.
        
    Returns:
        str: Modified name new name C02 --> C01
    """
    if "C" in string:
        number = string[1:3]
        new_name = f"C{int(number)-1:02}"
    else:
        number = string[2:4]
        new_name = f"P{int(number)-1:02}"
    
    return new_name


def get_slices_coords_without_completion(
                                image: np.ndarray, 
                                x_center: int, 
                                y_center: int) -> dict:
    """
    Get the coordinates for different slices in an image without completion.

    Args:
        image (numpy.ndarray): Input image.
        x_center (int): X coordinate of the center.
        y_center (int): Y coordinate of the center.

    Returns:
        dict: Dictionary containing the coordinates for each slice.
    """
    max_y, max_x, channels = image.shape
    diff = int((max_x - max_y) / 2)

    max_y, max_x = max_y - 1, max_x - 1
    x_h_line = np.array([int(i) for i in np.round(np.linspace(0, max_x, max_x + 1))])
    y_v_line = np.array([int(i) for i in np.round(np.linspace(0, max_y, max_y + 1))])

    y_135 = x_h_line + (y_center - x_center)
    y_45 = -x_h_line + y_center + x_center

    x_coords_slice_0 = x_h_line
    y_coords_slice_0 = np.array([y_center] * (max_x + 1))

    mask_slice_45 = (y_45 >= 0) & (y_45 <= max_y)
    x_coords_slice_45 = x_h_line[mask_slice_45]
    y_coords_slice_45 = y_45[mask_slice_45]

    x_coords_slice_90 = np.array([x_center] * (max_y + 1))
    y_coords_slice_90 = y_v_line

    mask_slice_135 = (y_135 >= 0) & (y_135 <= max_y)
    x_coords_slice_135 = x_h_line[mask_slice_135]
    y_coords_slice_135 = y_135[mask_slice_135]

    coords = {
        "x_slice0": x_coords_slice_0,
        "y_slice0": y_coords_slice_0,
        "x_slice45": x_coords_slice_45,
        "y_slice45": y_coords_slice_45,
        "x_slice90": x_coords_slice_90,
        "y_slice90": y_coords_slice_90,
        "x_slice135": x_coords_slice_135,
        "y_slice135": y_coords_slice_135
    }

    return coords

def calculate_slice_from_video(
                        video_name: str, 
                        slice_name: str, 
                        sample_index: str, 
                        x_center: int,
                        y_center: int, 
                        save_image: bool = False) -> None:
    """
    Calculate a slice from a video.

    Args:
        video_name (str): Name of the video.
        slice_name (str): Name of the slice (e.g., "slice0", "slice45", "slice90", "slice135").
        sample_index (str): Sample range index ("0", "1", "2", "3", "4").
        x_center (int): X coordinate of the center.
        y_center (int): Y coordinate of the center.
        save_image (bool, optional): Whether to save the image. Defaults to False.
    """

    # Read the video
    if "P" in video_name:
        video_reader = VideoReader(f'data/raw_videos/PK-ojos/{video_name}.avi')
    else:
        video_reader = VideoReader(f'data/raw_videos/C-ojos/{video_name}.avi')

    volume = np.zeros((300, 140, 210, 3), dtype=np.uint8)

    # Define sample ranges (frames)
    sample_ranges = {
        "0": (0, 300),
        "1": (300, 600),
        "2": (600, 900),
        "3": (900, 1200),
        "4": (1200, 1500)
    }

    # Extract video frames within the specified sample range
    for index, frame_index in enumerate(range(sample_ranges[sample_index][0], sample_ranges[sample_index][1])):
        volume[index] = cv2.cvtColor(video_reader[frame_index], cv2.COLOR_BGR2RGB)

    video_reader.close()
    num_frames, height, width, num_channels = volume.shape

    # Get the coordinates for the slice
    coords = get_slices_coords_without_completion(image = volume[0], 
                                                x_center = x_center, 
                                                y_center = y_center)
    x_coordinates, y_coordinates = coords["x_" + slice_name], coords["y_" + slice_name]

    new_slice = np.zeros((num_frames, x_coordinates.shape[0], 3), dtype=np.uint8)

    # Extract the slice from each frame
    for time_step in range(num_frames):
        image = volume[time_step]
        for index, (x, y) in enumerate(zip(x_coordinates, y_coordinates)):
            new_slice[time_step, index, 0] = image[y, x, 0]
            new_slice[time_step, index, 1] = image[y, x, 1]
            new_slice[time_step, index, 2] = image[y, x, 2]

    if new_slice.shape[0] != 210:
        # Resize the slice if necessary
        resized_slice = Image.fromarray(new_slice)
        resized_slice = resized_slice.resize((width, num_frames), resample=Image.NEAREST)
        new_slice = np.array(resized_slice)

    # Adjust the slice orientation
    new_slice = cv2.rotate(new_slice, cv2.ROTATE_90_CLOCKWISE)
    new_slice = cv2.flip(new_slice, 1)  # 0 x-axis, 1 y-axis, -1 both axes.
    new_slice = exposure.rescale_intensity(new_slice)

    if save_image:
        # Save the image if save_image is True
        image_to_save = Image.fromarray(new_slice)
        image_to_save.save(save_image)



#if main
if __name__ == "__main__":
    path_coordinates = "data/coordinates_ocular_fixation.csv"
    path_save_slices = "data/ocular_fixation_slices"
    #create folder to save slices
    if not os.path.exists(path_save_slices):
        os.makedirs(path_save_slices)
    # Read the coordinates CSV file
    coordinates_df = pd.read_csv(path_coordinates)
    coordinates_df["eye"] = coordinates_df["name"].apply(lambda x: get_eye_value(x))
    coordinates_df["new_name"] = coordinates_df["name"].apply(lambda x: get_name_number(x))

    # Iterate through unique patient names
    for original_patient_name in tqdm(sorted(coordinates_df["name"].unique())):
        new_patient_name = coordinates_df[coordinates_df["name"] == original_patient_name]["new_name"].unique()[0]  # new name, e.g, C02 --> C01
        
        # Iterate through eyes (left and right)
        for eye in coordinates_df["eye"].unique():
            patient_eye_df = coordinates_df[(coordinates_df["name"] == original_patient_name) & (coordinates_df["eye"] == eye)]  # sub dataframe of the patient
            
            # Iterate through unique samples
            for sample in patient_eye_df["sample"].unique():
                x_coord, y_coord = patient_eye_df.loc[patient_eye_df["sample"] == int(sample), ["xcord", "ycord"]].values[0]  # get the coordinates
                
                # Iterate through slices (0, 45, 90, 135 degrees)
                for slice_index, slice_name in enumerate(["slice0", "slice45", "slice90", "slice135"]):
                    calculate_slice_from_video(
                        video_name=original_patient_name,
                        slice_name=slice_name,
                        sample_index=str(sample),
                        x_center=x_coord,
                        y_center=y_coord,
                        save_image=f"{path_save_slices}/{new_patient_name}_{eye}_{sample}_{slice_index}.png"
                    )
