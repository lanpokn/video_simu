import numpy as np
import matplotlib.pyplot as plt

# Resolution and constants
width, height = 800, 600
iResolution = np.array([width, height])
iMouse = np.array([0.1, 0.1, 1.0])  # Example mouse position and click state
iTime = 1.0  # Example time value

# Functions for noise, lensflare, and color correction
def noise(t):
    return np.sin(12.9898 * t + 78.233) * 43758.5453 % 1.0

def lensflare(uv, pos):
    main = uv - pos
    uvd = uv * np.linalg.norm(uv)

    ang = np.arctan2(main[1], main[0])
    dist = np.linalg.norm(main)
    dist = dist ** 0.1
    n = noise(ang * 16.0 + dist * 32.0)

    f0 = 1.0 / (np.linalg.norm(uv - pos) * 16.0 + 1.0)
    f0 += f0 * (np.sin(noise(np.sin(ang * 2.0 + pos[0]) * 4.0 - np.cos(ang * 3.0 + pos[1])) * 16.0) * 0.1 + dist * 0.1 + 0.8)

    f1 = max(0.01 - (np.linalg.norm(uv + 1.2 * pos) ** 1.9), 0.0) * 7.0

    uvx = (uv + uvd) * -0.5
    f4 = max(0.01 - (np.linalg.norm(uvx + 0.4 * pos) ** 2.4), 0.0) * 6.0

    uvx = (uv + uvd) * -0.4
    f5 = max(0.01 - (np.linalg.norm(uvx + 0.2 * pos) ** 5.5), 0.0) * 2.0

    uvx = (uv + uvd) * -0.5
    f6 = max(0.01 - (np.linalg.norm(uvx - 0.3 * pos) ** 1.6), 0.0) * 6.0

    c = np.array([0.0, 0.0, 0.0])
    c[0] += f4 + f5 + f6
    c[1] += f4 * 0.8
    c[2] += f4 * 0.6

    c = c * 1.3 - np.array([np.linalg.norm(uvd) * 0.05] * 3)
    c += np.array([f0, f0, f0])

    return c

def color_correction(color, factor, factor2):
    w = color[0] + color[1] + color[2]
    return color * (1 - factor2) + np.array([w * factor] * 3) * factor2

# Main rendering function
def render_frame():
    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            fragCoord = np.array([x, y])
            uv = fragCoord / iResolution - 0.5
            uv[0] *= iResolution[0] / iResolution[1]  # Fix aspect ratio

            mouse = np.array([iMouse[0], iMouse[1]]) - 0.5
            mouse[0] *= iResolution[0] / iResolution[1]  # Fix aspect ratio

            if iMouse[2] < 0.5:
                mouse[0] = np.sin(iTime) * 0.5
                mouse[1] = np.sin(iTime * 0.913) * 0.5

            color = np.array([1.4, 1.2, 1.0]) * lensflare(uv, mouse)
            color -= noise(x + y * width) * 0.015
            color = color_correction(color, 0.5, 0.1)
            image[y, x] = np.clip(color, 0, 1)

    return image

# Render and display the frame
image = render_frame()
plt.imshow(image)
plt.axis('off')
plt.show()
