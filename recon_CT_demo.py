import matlab
import matlab.engine
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import torch

# Start the MATLAB engine
eng = matlab.engine.start_matlab()

# Load the .mat file
mat_data = sio.loadmat('D:\\file_jiang\\Proj\\demo1.mat')
mat_data = mat_data['proj']
print(type(mat_data))
proj = mat_data.astype(np.float32)
proj_list = proj.tolist()
input_proj = matlab.single(proj_list)

'''
data = eng.load('D:\\file_jiang\\Proj\\demo1.mat')  # Replace 'mydata.mat' with the path to your .mat file
# Extract the 'proj' variable from the data
input_proj = data['proj']
'''
# Calculate the output image with your function
output_image = eng.proj_to_image(input_proj)

# Convert the MATLAB array to a numpy array
output_image = np.array(output_image)

# Visualize the output image
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()