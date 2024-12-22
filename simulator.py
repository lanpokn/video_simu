#add a class:simulator
from flare.synthesis import image_add_flare
import tensorflow as tf

class simulator:
    def __init__(self):
        #TOBEDONE
        #may be can add some mode setting, provide a easy used method
        return
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
    def single_imgae_flare_initial(self,scene_path, flare_path,positions):
        self.scene, self.flare = self.load_images(scene_path, flare_path)
        self.position = positions
    def single_image_add_flare(self,mode='ISP'):
        scene_srgb, flare_srgb, scene_with_flare, gamma =  image_add_flare(self.scene, self.flare, self.position,mode=mode)
        return scene_srgb, flare_srgb, scene_with_flare
