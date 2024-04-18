import os.path

import matplotlib.pyplot as plt
from torch import nn,optim
import torch
from data_CT_process import *
from net import *
from torch_CT_recon_odl import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from resnet_demo import *
from loss import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = CTProjectionDataset(projection_dir='./Test3/proj',
                                        image_dir='./Test3/img')

    # 用DataLoader进行批处理
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
resnet = MyResNet()


resnet = resnet.to(device)



resnet.load_state_dict(torch.load('./Pretrained/resnet_params.pth'))
resnet.to(device)
resnet.eval()

# create the TensorRecon instance
recon_instance = TensorRecon()


# 开始测试
for i, (raw_ct_data, ideal_ct_data) in enumerate(test_dataloader):
    raw_ct_data, ideal_ct_data = raw_ct_data.to(device), ideal_ct_data.to(device)

    # 前向传播
    with torch.no_grad():  # In test phase, we don't need to compute gradients
        recon_instance = TensorRecon()  # 创建TensorRecon实例
        reconstructed_ct = recon_instance.tensor_to_image(raw_ct_data)
        reconstructed_ct = reconstructed_ct.to(device)
        final_ct = resnet(reconstructed_ct)
        img = final_ct.squeeze().detach().cpu().numpy()
        filename = f"{i}.png"
        io.imsave('D:\\参考文献\\Joint-Network paper\\Test-resnet\\P-change\\'+filename, img)
        #plt.imshow(img,'gray')
        #plt.show()