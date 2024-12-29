from simulator import simulator
import math_utils
import numpy as np
scene_path = "D:/2025/event_simu/flare/scene/12798.jpg"
flare_path = "D:/2025/event_simu/flare/Compound_Flare/000935.png"
# flare_path = "D:/2025/event_simu/flare/Compound_Flare/000000.png"

# intrinsic_matrix = np.array([[1000, 0, 320], [0, 1000, 320], [0, 0, 1]])
intrinsic_matrix = np.array([[100, 0, 320], [0, 100, 320], [0, 0, 1]])

R = simulator(intrinsic_matrix)
# position = [100, 100]
# R.single_imgae_flare_initial(scene_path, flare_path,position)
# _,_,scene_with_flare_ISP = R.single_image_add_flare(mode='ISP')
# R.save_image(scene_with_flare_ISP, "D:/2025/event_simu/scene_with_flare_ISP.jpg")
# _,_,scene_with_flare_direct = R.single_image_add_flare(mode='direct')
# R.save_image(scene_with_flare_direct, "D:/2025/event_simu/scene_with_flare_direct.jpg")
start = [320, 320] # Starting position
# start = [20, 20] # Starting position
end = [1,1]  # Ending position
num = 150  # Number of positions
positions = math_utils.generate_positions(start, end, num)
# print(positions)
R.single_image_flare_initial(scene_path, flare_path,positions)

#ISP是在凸组合基础上，修复原有bug得到的
_,flare_ISP,scene_with_flare_ISP = R.single_image_add_flare(mode='ISP')
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_ISP.mp4", save_as_video=True)
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_ISP", save_as_video=False)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_ISP.mp4", save_as_video=True)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_ISP", save_as_video=False)
#direct是直接把二者在raw上相加
_,flare_ISP,scene_with_flare_ISP = R.single_image_add_flare(mode='direct')
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_direct.mp4", save_as_video=True)
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_direct", save_as_video=False)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_direct.mp4", save_as_video=True)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_direct", save_as_video=False)
#analytic是假定tone mapping是它说的多项式，进行逆变换，再用我理解的0-1转0-正无穷转换，再加上小数置为0
#如果不进行0-1转0-正无穷，效果极烂，说明原文的ISP理论部分存在致命问题
#如果强制flare在正中间，则不处理小数的analytic方法效果应为最佳
#应该找ISP的理论知识，来为我这个做法找一些依据，以及进一步的优化
_,flare_ISP,scene_with_flare_ISP = R.single_image_add_flare(mode='analytic')
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_analytic.mp4", save_as_video=True)
R.save_images_or_video(scene_with_flare_ISP, "D:/2025/event_simu/result/scene_with_flare_analytic", save_as_video=False)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_analytic.mp4", save_as_video=True)
R.save_images_or_video(flare_ISP, "D:/2025/event_simu/result/flare_analytic", save_as_video=False)