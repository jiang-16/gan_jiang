import os.path

from torch import nn,optim
import torch
from data_CT_unet import *
from net import *
from torch_CT_recon_odl import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from resnet_demo import *
from loss import *

# 脚本的其余部分


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = CTProjectionDataset(projection_dir='./Train/proj',
                                        image_dir='./Train/proj_ideal')

    # 用DataLoader进行批处理
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
net = UNet().to(device)
#resnet = MyResNet()


#resnet = resnet.to(device)
#criterion = nn.MSELoss()
criterion = CombinedLoss(weights=[0.8, 0.1, 0.1])
#optimizer = optim.Adam(list(net.parameters()) + list(resnet.parameters()), lr=0.0001)
optimizer = optim.Adam(list(net.parameters()), lr=0.001)
num_epochs = 500
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for batch_idx, (raw_ct_data, ideal_ct_data) in enumerate(train_dataloader):
            raw_ct_data, ideal_ct_data =raw_ct_data.to(device),ideal_ct_data.to(device)
            # 初始化梯度
            optimizer.zero_grad()

            # 前向传播
            corrected_proj_data = net(raw_ct_data)

            #计算损失
            loss = criterion(corrected_proj_data, ideal_ct_data)
            epoch_loss += loss.item()
            num_batches += 1

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 计算平均损失
    average_loss = epoch_loss / num_batches

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')
torch.save(net.state_dict(), 'net_params.pth')
#torch.save(resnet.state_dict(), 'resnet_params.pth')

'''  
  recon_instance = TensorRecon()  # 创建TensorRecon实例
  reconstructed_ct = recon_instance.tensor_to_image(corrected_proj_data)
  reconstructed_ct = reconstructed_ct.to(device)
  final_ct = resnet(reconstructed_ct)
'''