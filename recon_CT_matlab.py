import matlab.engine
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt


eng = matlab.engine.start_matlab()
numpy_array = sio.loadmat('D:\\file_jiang\\Proj\\Train\\proj\\proj_0003_mask.mat')
numpy_array = numpy_array['proj2']
numpy_array = numpy_array.astype(np.float32)
proj_list = numpy_array.tolist()

# 将 numpy 数组转换为 MATLAB 可接受的形式
input_proj = matlab.single(proj_list)

# 调用 MATLAB 函数 proj_to_image
output_image = eng.proj_to_image(input_proj)

# 将 MATLAB 数组转换回 numpy 数组，并转换为适当的 float 类型
output_image = np.array(output_image)
img = output_image.astype(np.float32)
plt.imshow(img,'gray')
plt.show()