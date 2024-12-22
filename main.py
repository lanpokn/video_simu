from simulator import simulator
scene_path = "D:/2025/event_simu/flare/scene/12798.jpg"
flare_path = "D:/2025/event_simu/flare/Compound_Flare/000935.png"
position = [100, 100]
R = simulator()
R.single_imgae_flare_initial(scene_path, flare_path,position)
_,_,scene_with_flare_ISP = R.single_image_add_flare(mode='ISP')
R.save_image(scene_with_flare_ISP, "D:/2025/event_simu/scene_with_flare_ISP.jpg")
_,_,scene_with_flare_direct = R.single_image_add_flare(mode='direct')
R.save_image(scene_with_flare_direct, "D:/2025/event_simu/scene_with_flare_direct.jpg")
