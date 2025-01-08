import numpy as np
import matplotlib.pyplot as plt

def ACES_profession(x):
    # 定义输入和输出的转换矩阵
    ACESInputMat = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])

    ACESOutputMat = np.array([
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602]
    ])

    def RRTAndODTFit(v):
        """
        模拟 HLSL 的 RRTAndODTFit 函数
        """
        a = v * (v + 0.0245786) - 0.000090537
        b = v * (0.983729 * v + 0.4329510) + 0.238081
        return a / b

    # 转换为线性空间
    color = np.dot(ACESInputMat, x)

    # 应用 RRT 和 ODT 映射
    color = RRTAndODTFit(color)

    # 转换为 sRGB 空间
    color = np.dot(ACESOutputMat, color)

    return color
def ACES_profession_reverse(x):
    # 定义输入和输出的转换矩阵
    ACESInputMat = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])

    ACESOutputMat = np.array([
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602]
    ])
    ACESInputMat_inv = np.linalg.inv(ACESInputMat)
    ACESOutputMat_inv = np.linalg.inv(ACESOutputMat)
    def RRTAndODTFitInverse(y):
        """
        计算 RRTAndODTFit 的逆函数
        """
        # 常量定义
        A = 0.983729 * y - 1
        B = 0.4329510 * y - 0.0245786
        C = 0.238081 * y + 0.000090537

        # 判别式
        discriminant = B**2 - 4 * A * C
        # if discriminant < 0:
        #     raise ValueError("No real solution exists for the given y")

        # 计算两个可能解
        sqrt_discriminant = np.sqrt(discriminant)
        # v1 = (-B + sqrt_discriminant) / (2 * A)
        v2 = (-B - sqrt_discriminant) / (2 * A)

        # 返回符合 v > 0 的解
        return v2



    # 转换为线性空间
    color = np.dot(ACESOutputMat_inv, x)

    # 应用 RRT 和 ODT 映射
    color = RRTAndODTFitInverse(color)

    # 转换为 sRGB 空间
    color = np.dot(ACESInputMat_inv, color)

    return color
# x = np.linspace(0, 1, 500)
# # 将 x 转换为 3 通道输入 (RGB)
# input_colors = np.stack([x, x, x], axis=1)
# # 计算 ACES_profession 输出
# output_colors = np.array([ACES_profession_reverse(color) for color in input_colors])

# # 绘制结果
# plt.figure(figsize=(10, 6))
# plt.plot(x, output_colors[:, 0], label='Red Channel', color='red')
# plt.plot(x, output_colors[:, 1], label='Green Channel', color='green')
# plt.plot(x, output_colors[:, 2], label='Blue Channel', color='blue')
# plt.xlabel('Input Value')
# plt.ylabel('Output Value')
# plt.title('ACES Profession Output Curve')
# plt.legend()
# plt.grid()
# plt.show()
