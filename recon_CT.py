import numpy as np
import odl
import matplotlib.pyplot as plt

# 加载 MAT 文件（需要使用 SciPy）
from scipy.io import loadmat
from fliter_proj import *
data = loadmat('./Train/proj/proj_0016_mask.mat')
projections = data['proj2']  # 假设数据变量的名字叫 'name_of_the_variable_inside_mat'
# 设置扇形束 CT 几何参数
num_detectors = projections.shape[0]       # 探测器单元数量
num_angles = projections.shape[1]          # 角度数

# 角度范围为 0 到 2pi
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

# 假设扇形束射线半径和探测器宽度，这些参数应该与实际扫描参数匹配
detector_size = 0.23  # 探测器元素的大小（实际单元大小）
detector_span = num_detectors * detector_size  # 探测器行的总宽度

# 创建投影空间，可能需要类型根据你的数据进行调整
reco_space = odl.uniform_discr([-25.6, -25.6], [25.6, 25.6], [512, 512], dtype='float32')
angle_partition=odl.uniform_partition(0, 2 * np.pi, num_angles)
detector_partition=odl.uniform_partition(-detector_span / 2,detector_span / 2,num_detectors)
src_radius=600
det_radius=600

# 创建扇形束几何
geometry = odl.tomo.FanFlatGeometry(angle_partition,detector_partition,src_radius, det_radius)

# 创建前向算子
ray_transform = odl.tomo.RayTransform(reco_space, geometry)

geo = {
    "DSD": 1200,
    "DSO": 600,
    "nDetector": np.array([704, 1]),
    "dDetector": np.array([1, 1]) * 0.23,
    "sDetector": None,
    "nVoxel": np.array([512, 512]) / 1,
    "dVoxel": np.array([1, 1]) * 0.1,
    "sVoxel": None,
    "detoffset": np.array([0, 0]),
    "orgoffset": np.array([0, 0, 0])
}

# compute sDetector and sVoxel
geo["sDetector"] = geo["nDetector"] * geo["dDetector"]
geo["sVoxel"] = geo["nVoxel"] * geo["dVoxel"]

angles = np.linspace(0, 2*np.pi, 320, dtype=np.float32)

projections = np.flip(projections,axis=1)
# Assume `proj` is predefined or loaded data.
filtered_proj = filtered(projections, geo, angles)

fbp = ray_transform.adjoint(filtered_proj.T)
#fbp = np.flipud(fbp.data.T)
fbp = fbp.data.T
fbp = fbp.clip(min=0)










print(type(fbp),fbp.shape,fbp.dtype)









#plt.imshow(fbp.T,'gray')
plt.imshow(fbp,'gray')
plt.show()

'''
plt.figure()


# 创建反向投影算子 (FDK, filtered back-projection)
fbp_op = odl.tomo.fbp_op(ray_transform, filter_type='Ram-Lak')

# 转换投影数据以匹配 ODL 的数据结构

sinogram = np.transpose(projections)
print(sinogram.shape,sinogram.dtype)
# 使用 FBP 算法重建 CT 图像
reconstructed_image = fbp_op(sinogram)

plt.imshow(reconstructed_image,cmap='gray')
plt.show()'''