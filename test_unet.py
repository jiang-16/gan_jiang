import torch
from torch.utils.data import DataLoader
from data_CT_unet import CTProjectionDataset
from net import UNet
from torch_CT_recon_odl import *
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设定测试数据集
test_dataset = CTProjectionDataset(projection_dir='./Test3/proj', image_dir='./Test3/proj_ideal')

# 使用 DataLoader 进行批处理
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化网络，并加载训练好的权重
net = UNet().to(device)
net.load_state_dict(torch.load('./checkpoints/net_params.pth'))

# 将网络设置为评估模式
net.eval()

with torch.no_grad():
    for batch_idx, (raw_ct_data, ideal_ct_data) in enumerate(test_dataloader):
        raw_ct_data = raw_ct_data.to(device)
        ideal_ct_data = ideal_ct_data.to(device)

        # 前向传播获得预测结果
        corrected_proj_data = net(raw_ct_data)

        #numpy_array = corrected_proj_data.squeeze().detach().cpu().numpy()
        #numpy_array = numpy_array.astype(np.float32)
        #io.imsave('D:\\参考文献\\Joint-Network paper\\Fig\\fig3.png', numpy_array)

        recon = TensorRecon()
        img = recon.tensor_to_image(corrected_proj_data)
        img = img.detach().numpy()
        img = img.squeeze()
        filename = f"{batch_idx}.png"
        io.imsave('D:\\参考文献\\Joint-Network paper\\Test-unet\\P_change\\'+filename, img)

        #plt.imshow(img, 'gray')
        #plt.show()


        # 这里你可以进行量化评估，比如计算MSE loss，SSIM等
        # loss = criterion(corrected_proj_data, ideal_ct_data)
        # print("Test Loss: ", loss.item())

        # 也可以保存预测结果，然后在外部工具中进行评估
        # 为了简单，我们只打印每个batch的第一张预测结果和标签
        #print("Predicted CT Data: ", corrected_proj_data[0])
        #print("Ideal CT Data: ", ideal_ct_data[0])