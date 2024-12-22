import math
import tensorflow as tf
import numpy as np
import flare.utils as utils


def mixup(scene, flare,mode='ISP'):
  I = flare[:,:,0]+flare[:,:,1]+flare[:,:,2]
  I = I / 3
  # a = np.random.random()*4+3
  if mode == 'ISP':
    a = 5
    weight = 1/(1+np.e**(-a*(I-0.5)))
    weight = weight - tf.reduce_min(tf.compat.v1.layers.flatten(weight))
    weight = weight/tf.reduce_max(tf.compat.v1.layers.flatten(weight))
  else:
    a = 0
    weight = 1/(1+np.e**(-a*(I-0.5)))
  a1 = (scene[:,:,0]*(1-weight)+flare[:,:,0]*weight)
  a2 = (scene[:,:,1]*(1-weight)+flare[:,:,1]*weight)
  a3 = (scene[:,:,2]*(1-weight)+flare[:,:,2]*weight)
  return tf.clip_by_value(tf.stack([a1,a2,a3], axis=-1), 0.0, 1.0)

def add_flare(scene,
              flare,
              noise,
              flare_max_gain = 10.0,
              apply_affine = True,
              rotation = 0.,
              shift=[0,100],
              shear=[0.0,0.0],
              scale=[1.0,1.0],
              mode='ISP'):
  """Adds flare to natural images.
  Returns:
    - Flare-free scene in sRGB.
    - Flare-only image in sRGB.
    - Scene with flare in sRGB.
    - Gamma value used during synthesis.
  """
  # batch_size, flare_input_height, flare_input_width, _ = flare.shape

  # Since the gamma encoding is unknown, we use a random value so that the model
  # will hopefully generalize to a reasonable range of gammas.
  gamma = tf.random.uniform([], 1.8, 2.2)
  flare_linear = tf.image.adjust_gamma(flare, gamma)

  # Remove DC background in flare.
  # flare_linear = utils.remove_background(flare_linear)

  if apply_affine:
    # rotation = tf.random.uniform([batch_size], minval=-math.pi, maxval=math.pi)
    # shift = tf.random.normal([batch_size, 2], mean=0.0, stddev=10.0)
    # shear = tf.random.uniform([batch_size, 2],
    #                           minval=-math.pi / 9,
    #                           maxval=math.pi / 9)
    # scale = tf.random.uniform([batch_size, 2], minval=0.7, maxval=1.1)

    flare_linear = utils.apply_affine_transform(
        flare_linear,
        rotation=rotation,
        shift_x=shift[0],
        shift_y=shift[1],
        shear_x=shear[0],
        shear_y=shear[1],
        scale_x=scale[0],
        scale_y=scale[1])

  flare_linear = tf.clip_by_value(flare_linear, 0.0, 1.0)

  # flare_linear = tf.image.random_flip_left_right(
  #     tf.image.random_flip_up_down(flare_linear))

  # First normalize the white balance. Then apply random white balance.

  # flare_linear = utils.normalize_white_balance(flare_linear)
  # rgb_gains = tf.random.uniform([3], 0, flare_max_gain, dtype=tf.float32)
  # flare_linear *= rgb_gains

  # Further augmentation on flare patterns: random blur and DC offset.
  # blur_size = tf.random.uniform([], 0.1, 3)
  # flare_linear = utils.apply_blur(flare_linear, blur_size)
  # offset = tf.random.uniform([], -0.02, 0.02)
  # flare_linear = tf.clip_by_value(flare_linear + offset, 0.0, 1.0)

  flare_srgb = tf.image.adjust_gamma(flare_linear, 1.0 / gamma)
  # flare_srgb = flare_linear

  # Scene augmentation: random crop and flips.
  scene_linear = tf.image.adjust_gamma(scene, gamma)
  # scene_linear = tf.image.random_crop(scene_linear, flare_linear.shape)
  # scene_linear = tf.image.random_flip_left_right(
  #     tf.image.random_flip_up_down(scene_linear))

  # Additive Gaussian noise. The Gaussian's variance is drawn from a Chi-squared
  # distribution. This is equivalent to drawing the Gaussian's standard
  # deviation from a truncated normal distribution, as shown below.
  sigma = tf.abs(tf.random.normal([], 0, noise))
  noise = tf.random.normal(scene_linear.shape, 0, sigma)
  scene_linear += noise

  # Random digital gain.
  # TODO, hyperparameter, bg become dark due to autoexposure
  # gain = tf.random.uniform([], 0.85, 0.85)  # varying the intensity scale
  # scene_linear = tf.clip_by_value(gain * scene_linear, 0.0, 1.0)

  scene_srgb = tf.image.adjust_gamma(scene_linear, 1.0 / gamma)
  # scene_srgb = scene_linear

  # Combine the flare-free scene with a flare pattern to produce a synthetic
  # training example.
  # combined_1 = mixup(scene_linear[0], flare_linear[0])
  # combined_2 = mixup(scene_linear[1], flare_linear[1])
  # combined_srgb = tf.stack([combined_1, combined_2])
  combined_srgb = mixup(scene_linear, flare_linear,mode=mode)
  combined_srgb = tf.image.adjust_gamma(combined_srgb, 1.0 / gamma)
  combined_srgb = tf.clip_by_value(combined_srgb, 0.0, 1.0)
  
  return (utils.quantize_8(scene_srgb), utils.quantize_8(flare_srgb),
          utils.quantize_8(combined_srgb), gamma)

def load_images(scene_path, flare_path):
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
def save_image(image_tensor, output_path, file_format='jpeg'):
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
def position_to_shift(position, image_size):
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
scene_path = "D:/2025/event_simu/flare/scene/12798.jpg"
flare_path = "D:/2025/event_simu/flare/Compound_Flare/000935.png"

scene, flare = load_images(scene_path, flare_path)
position = [100, 100]
def image_add_flare(scene, flare, position,mode='ISP'):
    image_size = tf.shape(scene)


    shift = position_to_shift(position, image_size)
    # 将场景和flare图像传递给 `add_flare_at_position` 函数
    scene_srgb, flare_srgb, scene_with_flare, gamma = add_flare(scene, flare, noise=0, shift=shift,mode=mode)
    return scene_srgb, flare_srgb, scene_with_flare, gamma
image_add_flare(scene, flare, position)
#TODO, single image change to video, add  point light source