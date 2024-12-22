import numpy as np

def generate_positions(start_position, end_position, num_positions):
    """
    Generate an array of positions between a start and an end position.

    Args:
        start_position (list or tuple): The starting position [x, y].
        end_position (list or tuple): The ending position [x, y].
        num_positions (int): The number of positions to generate.

    Returns:
        list: A list of positions [[x1, y1], [x2, y2], ..., [xn, yn]].
    """
    if not (len(start_position) == 2 and len(end_position) == 2):
        raise ValueError("Start and end positions must be lists or tuples with two elements [x, y].")
    if num_positions < 2:
        raise ValueError("The number of positions must be at least 2.")
    
    # Generate linearly spaced values for x and y coordinates
    x_positions = np.linspace(start_position[0], end_position[0], num_positions)
    y_positions = np.linspace(start_position[1], end_position[1], num_positions)
    
    # Combine x and y coordinates into a list of [x, y]
    positions = [[x, y] for x, y in zip(x_positions, y_positions)]
    return positions
def calculate_angle_between_pixel_and_film(pixel_coords, intrinsic_matrix):
    """
    Calculate the angle between a pixel's ray and the film plane (optical axis).

    Args:
        pixel_coords (tuple): The (x, y) coordinates of the pixel in image space.
        intrinsic_matrix (np.ndarray): The camera intrinsic matrix (3x3) 
            of the form [[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]].

    Returns:
        float: The angle in radians between the ray from the pixel and the optical axis.
    """
    # Extract intrinsic parameters
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Convert pixel coordinates to normalized camera coordinates
    x_pixel, y_pixel = pixel_coords
    x_normalized = (x_pixel - cx) / fx
    y_normalized = (y_pixel - cy) / fy
    
    # Form the ray direction in camera coordinates
    ray_dir = np.array([x_normalized, y_normalized, 1.0])
    
    # Normalize the ray direction vector
    ray_dir_normalized = ray_dir / np.linalg.norm(ray_dir)
    
    # Calculate the angle with respect to the optical axis (z-axis)
    optical_axis = np.array([0.0, 0.0, 1.0])  # Z-axis
    dot_product = np.dot(ray_dir_normalized, optical_axis)
    angle = np.arccos(dot_product)  # Angle in radians
    
    return angle