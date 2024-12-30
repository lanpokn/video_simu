import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2  # 用于加载和调整图像尺寸
from math_utils import generate_positions  # Ensure math_utils is imported

class DayFlare:
    def __init__(self, width=800, height=450,color=None):
        self.width = width
        self.height = height
        self.iResolution = np.array([width, height])
        self.iMouse = np.array([0.1, 0.1, 1.0])  # Example mouse position and click state
        self.iTime = 1.0  # Example time value
        self.iChannel0 = None  # Placeholder for the texture
        self.color = np.array(color) if color is not None else np.array([1.4/2.1, 1.2/2.1, 1.0/2.1])


    def load_texture(self, image=None):
        """
        Load a texture into iChannel0. If no image is provided, generate a default noise texture.

        Args:
            image (np.ndarray, optional): Input image as a numpy array.

        Returns:
            None
        """
        if image is None:
            # Generate a default noise texture
            np.random.seed(42)
            self.iChannel0 = np.random.rand(self.height, self.width).astype(np.float32)
        else:
            # Normalize the image to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Resize the image if dimensions do not match
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            # Use only the first channel for grayscale (if RGB or multi-channel)
            self.iChannel0 = image if image.ndim == 2 else image[..., 0]

    def noise(self, t):
        # """
        # Sample noise from iChannel0 based on the input t.

        # Args:
        #     t (float or array-like): Input position(s) for sampling.

        # Returns:
        #     float or np.ndarray: Sampled noise values.
        # """
        # if self.iChannel0 is None:
        #     raise ValueError("iChannel0 texture is not initialized. Please load a texture using `load_texture`.")

        # t = np.asarray(t)
        # tex_coords = (t % self.width) / self.width
        # tex_coords = (tex_coords * sel
        # 
        # f.iChannel0.shape[1]).astype(int)  # Map to texture indices
        # return self.iChannel0[0, tex_coords]
        return np.sin(10*t)*0.01

    def lensflare(self, uv, pos):
        main = uv - pos
        uvd = uv * np.linalg.norm(uv, axis=-1, keepdims=True)

        ang = np.arctan2(main[..., 1], main[..., 0])
        ang = np.abs(ang)
        dist = np.linalg.norm(main, axis=-1) ** 0.1

        f0 = 1.0 / (np.linalg.norm(uv - pos, axis=-1) * 16.0 + 1.0)
        f0 += f0 * (np.sin(self.noise(np.sin(ang * 2.0 + pos[0]) * 4.0 - np.cos(ang * 3.0 + pos[1])) * 16.0) * 0.1 + dist * 0.1 + 0.8)

        # GLSL中f1计算
        f1 = np.maximum(0.01 - (np.linalg.norm(uv + 1.2 * pos, axis=-1) ** 1.9), 0.0) * 7.0

        # GLSL中f2, f22, f23计算
        f2 = np.maximum(1.0 / (1.0 + 32.0 * np.power(np.linalg.norm(uvd + 0.8 * pos, axis=-1), 2.0)), 0.0) * 0.25
        f22 = np.maximum(1.0 / (1.0 + 32.0 * np.power(np.linalg.norm(uvd + 0.85 * pos, axis=-1), 2.0)), 0.0) * 0.23
        f23 = np.maximum(1.0 / (1.0 + 32.0 * np.power(np.linalg.norm(uvd + 0.9 * pos, axis=-1), 2.0)), 0.0) * 0.21

        # GLSL中f4, f42, f43计算
        uvx = (uv + uvd) * -0.5
        f4 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.4 * pos, axis=-1) ** 2.4), 0.0) * 6.0
        f42 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.45 * pos, axis=-1) ** 2.4), 0.0) * 5.0
        f43 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.5 * pos, axis=-1) ** 2.4), 0.0) * 3.0

        # GLSL中f5, f52, f53计算
        uvx = (uv + uvd) * -0.4
        f5 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.2 * pos, axis=-1) ** 5.5), 0.0) * 2.0
        f52 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.4 * pos, axis=-1) ** 5.5), 0.0) * 2.0
        f53 = np.maximum(0.01 - (np.linalg.norm(uvx + 0.6 * pos, axis=-1) ** 5.5), 0.0) * 2.0

        # GLSL中f6, f62, f63计算
        uvx = (uv + uvd) * -0.5
        f6 = np.maximum(0.01 - (np.linalg.norm(uvx - 0.3 * pos, axis=-1) ** 1.6), 0.0) * 6.0
        f62 = np.maximum(0.01 - (np.linalg.norm(uvx - 0.325 * pos, axis=-1) ** 1.6), 0.0) * 3.0
        f63 = np.maximum(0.01 - (np.linalg.norm(uvx - 0.35 * pos, axis=-1) ** 1.6), 0.0) * 5.0

        # 颜色分量
        c = np.zeros((*uv.shape[:2], 3))
        
        # R, G, B 通道的颜色分量
        c[..., 0] += f2 + f4 + f5 + f6
        c[..., 1] += f22 + f42 + f52 + f62
        c[..., 2] += f23 + f43 + f53 + f63

        # 调整强度和背景亮度
        c = c * 1.3 - np.linalg.norm(uvd, axis=-1, keepdims=True) * 0.05
        c += np.expand_dims(f0, axis=-1)

        return c

    @staticmethod
    def color_correction(color, factor, factor2):
        w = color[..., 0] + color[..., 1] + color[..., 2]
        return color * (1 - factor2) + np.expand_dims(w * factor, axis=-1) * factor2

    def render_frame(self):
        fragCoord = np.stack(np.meshgrid(np.arange(self.width), np.arange(self.height)), axis=-1)
        uv = fragCoord / self.iResolution - 0.5
        uv[..., 0] *= self.iResolution[0] / self.iResolution[1]

        mouse = self.iMouse[:2] - 0.5
        mouse[0] *= self.iResolution[0] / self.iResolution[1]

        if self.iMouse[2] < 0.5:
            mouse[0] = np.sin(self.iTime) * 0.5
            mouse[1] = np.sin(self.iTime * 0.913) * 0.5

        color = 2.1*self.color * self.lensflare(uv, mouse)

        noise_value = self.noise(fragCoord[..., 0] + fragCoord[..., 1] * self.width) * 0.015
        noise_rgb = np.repeat(noise_value[..., np.newaxis], 3, axis=-1)
        color -= noise_rgb

        color = self.color_correction(color, 0.5, 0.1)
        return np.clip(color, 0, 1)

    def render_video(self, start_position, end_position, num_frames, output_file="output.mp4"):

        positions = generate_positions(start_position, end_position, num_frames)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")

        def update(frame_index):
            self.iMouse[:2] = positions[frame_index]
            image = self.render_frame()
            im.set_array(image)
            return [im]

        initial_image = self.render_frame()
        im = ax.imshow(initial_image, animated=True)

        ani = animation.FuncAnimation(
            fig, update, frames=num_frames, blit=True, interval=30
        )

        ani.save(output_file, fps=30, extra_args=["-vcodec", "libx264"])
        print(f"Video saved as {output_file}")

# 使用示例
day_flare = DayFlare()
image_path = "D:/2025/event_simu/noise.jpg"
# day_flare.load_texture(image=cv2.imread(image_path))  # 加载默认纹理
day_flare.load_texture()
start_position = [0.1, 0.1]
end_position = [0.9, 0.9]
num_frames = 100
day_flare.render_video(start_position, end_position, num_frames, output_file="lensflare_animation.mp4")
