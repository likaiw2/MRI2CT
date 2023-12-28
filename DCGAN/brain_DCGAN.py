import os

from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
import numpy as np
from torch.utils.data import DataLoader
# import torch
# from torch import nn
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import math
import time
import datetime
import matplotlib.pyplot as plt
from model import *

torch.cuda.empty_cache()

torch.manual_seed(0)
# modified RandomCrop3D class (refer to: https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421), which:
# with one call, crop a pair of (volume_ct/volume_mr) at the same position;
# with different calls, randomly crop volumes at different positions.
# 随机剪裁配对图片的不同对应位置
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_depth=image.shape[2] //2
    middle_height = image.shape[3] // 2
    middle_width = image.shape[4] // 2
    starting_depth=middle_depth-new_shape[2]//2
    final_depth=starting_depth+new_shape[2]//2
    starting_height = middle_height - new_shape[3] // 2
    final_height = starting_height + new_shape[3]
    starting_width = middle_width - new_shape[4] // 2
    final_width = starting_width + new_shape[4]
    cropped_image = image[:, :,starting_depth:final_depth, starting_height:final_height, starting_width:final_width]
    return cropped_image

def saveRawFile10(cur_step, res, t, l, volume):
    if not os.path.exists(dataSavePath):
        os.makedirs(dataSavePath)
    fileName = '%s%s_%.4d_%s_%d.raw' % (dataSavePath, res, t, l, cur_step)
    # fileName = '%s%s_%s_%d.raw' % (dataSavePath, res, l, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)
# add on 2022.9.26.


def saveRawFile2(cur_step, t, volume):
    fileName = '%sH_%s_%d.raw' % (dataSavePath, t, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

def saveModel(cur_step):
    # print("savemodel start!")
    if save_model:
        fileName = "%sDCGAN_%d.pth" % (dataSavePath, cur_step)
        torch.save({'gen': gen.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    'disc': disc.state_dict(),
                    'disc_opt': disc_opt.state_dict(),
                    }, fileName)  # , _use_new_zipfile_serialization=False)

# def saveModel(cur_step):
#     if save_model:
#         fileName = "%spix2pix_%d.pth" % (dataSavePath, cur_step)
#         torch.save({'gen': gen.state_dict(),
#                     'gen_opt': gen_opt.state_dict(),
#                     'disc_S': disc_S.state_dict(),
#                     'disc_T': disc_T.state_dict(),
#                     'disc_S_opt': disc_S_opt.state_dict(),
#                     'disc_T_opt': disc_T_opt.state_dict(),
#                     }, fileName)  # , _use_new_zipfile_serialization=False)

# 一个随机生成3D切片的类，
class MyRandomCrop3D3(object):
    def __init__(self, volume_sz, cropVolume_sz):
        c, d, h, w = volume_sz                  # 输入体积大小
        assert (d, h, w) >= cropVolume_sz
        self.volume_sz = tuple((d, h, w))       #裁剪体积大小
        self.cropVolume_sz = tuple(cropVolume_sz)

    def __call__(self, volume_ct, volume_mr):
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]     #记录裁剪的大小和位置
        return self._crop(volume_ct, volume_mr, *slice_dhw)

    @staticmethod
    def _get_slice(volume_sz, cropVolume_sz):           # 随机生成所需裁剪的切片位置，并处理边界情况，确保切片范围在给定的 3D 体积数组内或返回 None来处理捕捉到的异常
        try:
            lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
            return lower_bound, lower_bound + cropVolume_sz
        except:
            return (None, None)

    # 将采集到的起始和结束点的切片应用于 CT 和 MR 数据，返回两个三维体积(volume_ct和volume_mr) 这些张量将作为裁剪器的输出和输入提供给下游应用程序
    @staticmethod
    def _crop(volume_ct, volume_mr, slice_d, slice_h, slice_w):     
        # print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
        return volume_ct[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_mr[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]

# 定义一个体数据集，实现把数据读入并且保存在类中，读70组配对数据
class VolumesDataset(Dataset):
    def __init__(self, dataSourcePath, nTimesteps_train, dim,
                 fileStartVal, fileIncrement, constVal, float32DataType=np.float32,
                 transform=None):
        self.dataSourcePath = dataSourcePath
        self.nTimesteps_train = nTimesteps_train  # number of timesteps used for training.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return self.nTimesteps_train  # =70.

    # given an index, return a pair of (ct, mr).
    def __getitem__(self, index):  # index: [0, 69].
        # if index is outside normal range.
        if index < 0 or index >= self.nTimesteps_train:
            print('index is outside the normal range.\n')
            return


        # 1. at index, read original a pair of (volume_ct, volume_mr) .raw files.
        # (1.1)read original volume_ct.
        fileName = '%snorm_ct_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_ct = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_ct = torch.from_numpy(volume_ct)
        # reshape.
        volume_ct = volume_ct.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (1.2)read original volume_mr.
        fileName = '%snorm_mr_enContrast.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_mr = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_mr = torch.from_numpy(volume_mr)
        # reshape.
        volume_mr = volume_mr.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].


        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        #make sure crop_ct, crop_mr are the same size.
        assert crop_ct.shape == crop_mr.shape


        return crop_ct, crop_mr, index
        #correct: 2023.5.22.

# 对3d卷积神经网络的权重初始化
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

#模型的代码实现见model.py
def do_nothing():
    pass

#Training Process
# New parameters
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200
device=torch.device("cuda")
n_epochs = 1000
input_dim = 1
real_dim = 1
batch_size = 1          #原模型参数 10
lr = 0.0002             #原模型参数 5e-3(0.005)
# dropout_rate = 0.2      #原模型参数 0.2
target_shape = 256

totalTimesteps = 100
trainsetRatio = 0.7  # according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTimesteps_train = round(totalTimesteps * trainsetRatio)  # nTimesteps_train=70.
# display_step = 70     #需要一个一个epoch输出的时候用
display_step = np.ceil(np.ceil(nTimesteps_train / batch_size) * n_epochs / 20)   #一共输出20个epoch，供判断用

dataSourcePath = "dataSet1/brain/"
dataSavePath = "output/trial08/brain/"
#trial01:batchsize=1,hiddenchannel=32
#trial02:batchsize=1,hiddenchannel=16
#trial03:new dataset
#trial04:display_num=20
#trial05:add layer
#trial06:n_epoch=1000
#trial07:全局残差块
#trial08:规范了brain 和 pelvic的代码区分

fileStartVal = 1
fileIncrement = 1
constVal = 1
cropScaleFactor = (0.5, 0.5, 0.5)  # [depth, height, width].
dim = (160, 224, 168)   # [depth, height, width]. brain
# dim = (96, 240, 384)    # [depth, height, width]. pelvic
dim_crop = (int(dim[0] * cropScaleFactor[0]),
            int(dim[1] * cropScaleFactor[1]),
            int(dim[2] * cropScaleFactor[2]))
float32DataType = np.float32
myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
                               cropVolume_sz=dim_crop)
import torchvision
# dataset = torchvision.datasets.ImageFolder("maps", transform=transform)
trainDataset=VolumesDataset(dataSourcePath=dataSourcePath, nTimesteps_train=nTimesteps_train,
                            dim=dim,
                            fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal,
                            float32DataType=float32DataType,
                            transform=myRandCrop3D)

# gen = UNet(input_dim, real_dim).to(device)
gen = ResUNet_LRes(in_channel=input_dim, n_classes=1, dp_prob=0.2).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
#disc = Discriminator(input_dim + real_dim).to(device)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
save_model = True

if pretrained:
    loaded_state = torch.load("output/trial8/DCGAN_27999.pth")
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: get_gen_loss
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the true labels and returns a adversarial
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator
                    outputs and the real images and returns a reconstructuion
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.
    #### START CODE HERE ####
    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(real, fake)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    #### END CODE HERE ####
    return gen_loss
# UNIT TEST
def gen_reasonable_test(num_images=10):
    gen = torch.zeros_like
    disc = lambda x, y: torch.ones(len(x), 1)
    real = None
    condition = torch.ones(num_images, 3, 10, 10)
    adv_criterion = torch.mul
    recon_criterion = lambda x, y: torch.tensor(0)
    lambda_recon = 0
    assert get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon).sum() == num_images

    disc = lambda x, y: torch.zeros(len(x), 1)
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)).sum() == 0

    adv_criterion = lambda x, y: torch.tensor(0)
    recon_criterion = lambda x, y: torch.abs(x - y).max()
    real = torch.randn(num_images, 3, 10, 10)
    lambda_recon = 2
    gen = lambda x: real + 1
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon) - 2) < 1e-4

    adv_criterion = lambda x, y: (x + y).max() + x.max()
    assert torch.abs(get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon) - 3) < 1e-4
gen_reasonable_test()
print("Success!")
from skimage import color
import numpy as np

def train(save_model=True):
    # read the start time
    ot = time.time()
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("##train start(brain)##  time:",t1)
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for ct,mri,index in dataloader:
            residual_source = ct
            # print("ct: " , ct.shape)
            # print("mri: " , mri.shape)
            cur_batch_size=len(ct)
            ct=ct.to(device)
            mri=mri.to(device)
            # image_width = image.shape[3]
            # condition = image[:, :, :, :image_width // 2]
            # condition = nn.functional.interpolate(condition, size=target_shape)
            # real = image[:, :, :, image_width // 2:]
            # real = nn.functional.interpolate(real, size=target_shape)
            # cur_batch_size = len(condition)
            # condition = condition.to(device)
            # real = real.to(device)

            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(ct)
            # 6.6 00：46 p2p代码有两个输入值，但是本代码只需要一个，所以产生了报错  7.2 已解决
            # 输入是[1, 1, 80, 112, 84]
            disc_fake_hat = disc(fake.detach(),ct) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(mri, ct)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, mri, ct, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if (cur_step+1) % display_step == 0 or cur_step == 1:
                
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                print(t, f"  Epoch {epoch}: Step {cur_step}: Generator (Res_UNet) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                #计算单位运行时间
                dt = time.time() - ot
                elapsedTime = str(datetime.timedelta(seconds=dt))
                per_epoch = str(datetime.timedelta(seconds=dt / (epoch+1)))
                print(f"    epoch = {epoch}     dt={elapsedTime}    per-epoch={per_epoch}")
                # save fake.
                saveRawFile10(cur_step,
                              'fake_mr',
                              (fileStartVal + index * fileIncrement) / constVal,
                              '',
                              fake[0, 0, :, :, :])

                saveRawFile10(cur_step, 'truth_mr', (fileStartVal + index * fileIncrement) / constVal, '',
                              mri[0, 0, :, :, :])
                
                # show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                # show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                # show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    saveModel(cur_step=cur_step)
                    
            cur_step += 1
    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("##train finished(brain)##  time:",t2)
    print("total train time:")
    print("start:",t1)
    print("end:",t2)
train()