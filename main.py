from simulator import simulator
import math_utils
import numpy as np
scene_path = "D:/2025/event_simu/flare/scene/12798.jpg"
flare_path = "D:/2025/event_simu/flare/Compound_Flare/000935.png"
intrinsic_matrix = np.array([[1000, 0, 320], [0, 1000, 320], [0, 0, 1]])
R = simulator(intrinsic_matrix)
# position = [100, 100]
# R.single_imgae_flare_initial(scene_path, flare_path,position)
# _,_,scene_with_flare_ISP = R.single_image_add_flare(mode='ISP')
# R.save_image(scene_with_flare_ISP, "D:/2025/event_simu/scene_with_flare_ISP.jpg")
# _,_,scene_with_flare_direct = R.single_image_add_flare(mode='direct')
# R.save_image(scene_with_flare_direct, "D:/2025/event_simu/scene_with_flare_direct.jpg")
start = [100, 200] # Starting position
end = [1,1]  # Ending position
num = 15  # Number of positions
positions = math_utils.generate_positions(start, end, num)
# print(positions)
R.single_image_flare_initial(scene_path, flare_path,positions)
_,_,scene_with_flare_ISP = R.single_image_add_flare(mode='ISP')
# R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/scene_with_flare_ISP.mp4", save_as_video=True)
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/scene_with_flare_ISP", save_as_video=False)
#真正的直接加是这样的scene_with_flare = scene_srgb * (1 - flare_srgb) + flare_srgb  # Mix flare with scene
#效果巨烂
