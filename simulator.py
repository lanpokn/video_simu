#add a class:simulator
from flare.synthesis import image_add_flare
import tensorflow as tf
import cv2
import os
import numpy as np
from math_utils import calculate_angle_between_pixel_and_film
class simulator:
    def __init__(self, intrinsic_matrix=None):
        if intrinsic_matrix is None:
            # Default intrinsic matrix, assuming a camera with focal length 1000 and 
            # principal point at (960, 540) for a 1920x1080 image
            self.intrinsic_matrix = np.array([[1000, 0, 960],
                                              [0, 1000, 540],
                                              [0, 0, 1]])
        else:
            self.intrinsic_matrix = intrinsic_matrix
    def load_images(self,scene_path, flare_path):
        """Loads and processes scene and flare images for further use.

        Args:
        scene_path: Path to the scene image file.
        flare_path: Path to the flare image file.

        Returns:
        - scene: Preprocessed scene image tensor.
        - flare: Preprocessed flare image tensor, resized to match scene's size.
        """
        
        # Load the scene and flare images as tensors
        scene = tf.io.read_file(scene_path)
        flare = tf.io.read_file(flare_path)

        # Decode the images (assuming they are in standard formats like PNG/JPEG)
        scene = tf.image.decode_image(scene, channels=3)  # Scene should be 3-channel (RGB)
        flare = tf.image.decode_image(flare, channels=3)  # Flare should also be 3-channel (RGB)

        # Normalize the images to the range [0, 1]
        scene = tf.cast(scene, tf.float32) / 255.0
        flare = tf.cast(flare, tf.float32) / 255.0

        # Get the dimensions of the scene image
        scene_shape = tf.shape(scene)

        # Resize the flare image to match the scene image's dimensions
        flare = tf.image.resize(flare, [scene_shape[0], scene_shape[1]])

        return scene, flare
    def save_image(self,image_tensor, output_path, file_format='jpeg'):
        """Saves a TensorFlow image tensor to disk in the specified format.
        
        Args:
            image_tensor: The image tensor to be saved.
            output_path: Path where the image will be saved.
            file_format: The format to save the image, either 'jpeg' or 'png' (default is 'jpeg').
        """
        # Ensure the image tensor is in the [0, 1] range
        image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)

        # Convert the image tensor to 8-bit by multiplying by 255 and converting to uint8
        image_tensor = tf.cast(image_tensor * 255.0, tf.uint8)

        # Encode the image tensor to the desired format
        if file_format.lower() == 'jpeg':
            encoded_image = tf.image.encode_jpeg(image_tensor)
        elif file_format.lower() == 'png':
            encoded_image = tf.image.encode_png(image_tensor)
        else:
            raise ValueError("Unsupported file format. Use 'jpeg' or 'png'.")

        # Write the encoded image to the specified file
        tf.io.write_file(output_path, encoded_image)
    def save_images_or_video(self, scene_with_flare_list, output_path, save_as_video=True, file_format='jpeg', fps=30):
        """
        Save a list of images (scene_with_flare_list) as individual image files or a single video file.

        Args:
            scene_with_flare_list (list): List of TensorFlow image tensors to save.
            output_path (str): Path to save the output. For images, this should be a directory. For video, this is a file path.
            save_as_video (bool): Whether to save as a video (True) or as individual images (False).
            file_format (str): File format for individual images ('jpeg' or 'png'). Default is 'jpeg'.
            fps (int): Frames per second for the video. Default is 30.
        """
        if save_as_video:
            # Ensure the output path ends with a video extension
            if not output_path.endswith('.mp4'):
                raise ValueError("Output path for video must end with .mp4")

            # Get the dimensions from the first image tensor
            height, width, channels = scene_with_flare_list[0].shape

            # Use OpenCV to create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for image_tensor in scene_with_flare_list:
                # Convert the image tensor to a NumPy array
                image = tf.clip_by_value(image_tensor, 0.0, 1.0)
                image = (image.numpy() * 255).astype('uint8')
                video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR format

            video_writer.release()
            print(f"Video saved to {output_path}")
        else:
            # Ensure the output path is a directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for i, image_tensor in enumerate(scene_with_flare_list):
                # Construct the file path for each image
                image_path = os.path.join(output_path, f"image_{i:04d}.{file_format}")
                # Save each image using the self.save_image method
                self.save_image(image_tensor, image_path, file_format=file_format)

            print(f"Images saved to directory {output_path}")
    def position_to_shift(self,position, image_size):
        """
        Convert position (with respect to top-left corner) to shift (with respect to image center).
        
        Args:
        position: A list [pos_x, pos_y] representing the center of the flare in the image.
        image_size: A tuple (W, H) representing the width and height of the image.
        
        Returns:
        shift: A list [shift_x, shift_y] representing the movement of the flare center from the image center.
        """
        W, H,_ = image_size
        
        # Calculate the center of the image
        center_x = W / 2
        center_y = H / 2
        
        # Convert position to shift
        shift_x = position[0] - center_x
        shift_y = position[1] - center_y
        
        return [shift_x, shift_y]
    # def single_imgae_flare_initial(self,scene_path, flare_path,position):
    #     self.scene, self.flare = self.load_images(scene_path, flare_path)
    #     self.position = position
    # def single_image_add_flare(self,mode='ISP'):
    #     scene_srgb, flare_srgb, scene_with_flare, gamma =  image_add_flare(self.scene, self.flare, self.position,mode=mode)
    #     return scene_srgb, flare_srgb, scene_with_flare
    def single_image_flare_initial(self, scene_path, flare_path, position):
        """
        Initialize the scene image, flare image, and position parameter.
        
        Args:
            scene_path (str): Path to the scene image.
            flare_path (str): Path to the flare image.
            position (list): Flare position, either a single point [a, b] or multiple points [[a1, b1], [a2, b2], ...].
        """
        self.scene, self.flare = self.load_images(scene_path, flare_path)
        self.position = position

    def single_image_add_flare(self, mode='ISP'):
        """
        Add flare effect, supporting a single position or multiple positions.
        
        Args:
            mode (str): Mode option, default is 'ISP'.
        
        Returns:
            - If position is a single point, returns (scene_srgb, flare_srgb, scene_with_flare).
            - If position is multiple points, returns a list of scene_with_flare for each position.
        """
        if isinstance(self.position[0], (int, float)):  # Single position [a, b]
            scene_srgb, flare_srgb, scene_with_flare, gamma = image_add_flare(
                self.scene, self.flare, self.position, mode=mode
            )
            return scene_srgb, flare_srgb, scene_with_flare
        elif isinstance(self.position[0], list):  # Multiple positions [[a1, b1], [a2, b2], ...]
                scene_with_flare_list = []
                scene_srgb_list = []
                flare_srgb_list = []
                
                for pos in self.position:                    
                    # Compute theta for the flare attenuation at each position
                    theta = calculate_angle_between_pixel_and_film(pos, self.intrinsic_matrix)
                    flare = self.flare * np.cos(theta)  # Apply cos(theta) attenuation to flare
                    #TODO, may be it's better to add in raw domain?
                    #theta can be a parameter of image_add_flare
                    #现有方法根据flare强度分区的办法，可能会导致flare强度衰减失效，需要推导理论改进
                    scene_srgb, flare_srgb, scene_with_flare, _ = image_add_flare(self.scene, flare, pos, mode=mode)

                    # Store results in the lists
                    scene_with_flare_list.append(scene_with_flare)
                    scene_srgb_list.append(scene_srgb)
                    flare_srgb_list.append(flare_srgb)
                
                return scene_srgb_list, flare_srgb_list, scene_with_flare_list
        else:
            raise ValueError("Invalid position format. Must be [a, b] or [[a1, b1], [a2, b2], ...].")

