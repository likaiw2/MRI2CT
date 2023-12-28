'''
    single weights
'''

import os
from torch.utils.data import Dataset, DataLoader
import torch
import time

from model import *

weightsPath = "output/for_test/DCGAN_14799.pth"
dataSourcePath = "dataSet/"
dataSavePath = "testResults/trial02_14799/"
device = torch.device("cuda:0")

print(f"cuda is available: {torch.cuda.is_available()}.")
lr = 0.0002
batch_size = 1

torch.cuda.empty_cache()
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"]


class VolumesDataset_test(Dataset):
    def __init__(self, dataSourcePath, totalTimesteps, nTimesteps_test, dim,
                 fileStartVal, fileIncrement, constVal, float32DataType=np.float32,
                 transform=None):
        self.dataSourcePath = dataSourcePath
        self.totalTimesteps = totalTimesteps  # totalTimesteps=100.
        self.nTimesteps_test = nTimesteps_test  # number of timesteps used for test.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim
        # correct: 2022.10.14.

    def __len__(self):
        return self.nTimesteps_test  # =30.
        # correct: 2022.10.14.

    # given an index, return 1 pair of (ct, mr).
    def __getitem__(self, index):  # index: [0, 29].
        # (i)if index is outside normal range.
        if index < 0 or index >= self.nTimesteps_test:
            print('index is outside the normal range.\n')
            return

        # (ii)convert index range from [0, 29] to [70, 99].
        index += (self.totalTimesteps - self.nTimesteps_test)  # index: [70, 99].


        # 1. at index, read original a pair of (volume_ct, volume_mr) .raw files.
        # (1.1)read original volume_ct.
        fileName = '%snorm_ct.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_ct = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_ct = torch.from_numpy(volume_ct)
        # reshape.
        volume_ct = volume_ct.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].


        # (1.2)read original volume_mr.
        fileName = '%snorm_mr.%.3d.raw' % (self.dataSourcePath, (self.fileStartVal + index * self.fileIncrement) / self.constVal)
        volume_mr = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarry to tensor.
        volume_mr = torch.from_numpy(volume_mr)
        # reshape.
        volume_mr = volume_mr.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].


        # 2. given volume_ct/_mr, crop them to get cropped volumes, for data augmentation.
        if self.transform:
            crop_ct, crop_mr = self.transform(volume_ct, volume_mr)

        #make sure crop_ct and crop_mr are the same size.
        assert crop_ct.shape == crop_mr.shape


        return crop_ct, crop_mr, index
        # correct: 2023.5.22.

def saveRawFile10(dataSavePath, cur_step, res, t, l, volume):
    # fileName = '%s%s_%.4d_%s_%d.raw' % (dataSavePath, res, t, l, cur_step)
    fileName = '%s%s_%s_%d.raw' % (dataSavePath, res, l, cur_step)
    volume = volume.view(dim_crop[0], dim_crop[1], dim_crop[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

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
    @staticmethod
    def _crop(volume_ct, volume_mr, slice_d, slice_h, slice_w):     
        # print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
        return volume_ct[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_mr[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]

# add on 2022.9.26.

# load the net model
print("loading net!")
dim = 1
gen = ResUNet_LRes(in_channel=dim, n_classes=1, dp_prob=0.2).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(dim + dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# load the weights
print("loading weights!")
if os.path.exists(weightsPath):
    # net.load_state_dict(torch.load(weights))
    loaded_state = torch.load(weightsPath)
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
    print("Weight load success!")
else:
    print("load weights failed!")


# load test data
print("loading datas!")
totalTimesteps = 100
trainsetRatio = 0.7  # according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTimesteps_train = round(totalTimesteps * trainsetRatio)  # nTimesteps_train=70
nTimesteps_test = totalTimesteps - nTimesteps_train  # nTimesteps_test=30.
dim = (96, 240, 384)  # [depth, height, width].
fileStartVal = 1
fileIncrement = 1
constVal = 1
float32DataType = np.float32
cropScaleFactor = (1, 1, 1)  # [depth, height, width].
dim_crop = (int(dim[0] * cropScaleFactor[0]),
            int(dim[1] * cropScaleFactor[1]),
            int(dim[2] * cropScaleFactor[2]))
myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
                               cropVolume_sz=dim_crop)
dataset_test = VolumesDataset_test(dataSourcePath=dataSourcePath, totalTimesteps=totalTimesteps,
                                   nTimesteps_test=nTimesteps_test,
                                   dim=dim,
                                   fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal,
                                   float32DataType=float32DataType,
                                   transform=myRandCrop3D
                                   )


def test():
    print("testdata begin!")
    dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)  # make sure change shuffle=True to be shuffle=False.
    gen.eval()

    with torch.no_grad():
        # Dataloader returns the batches
        for ct,mri,index in dataloader:
            residual_source = ct
            cur_batch_size = len(ct)
            ct = ct.to(device)
            fake = gen(ct)

            saveRawFile10(dataSavePath, index+1, 'DCGAN_pred_norm_mr', index, '', fake[0, 0, :, :, :])
            # saveRawFile10(dataSavePath, index+1, 'truth_mr', index, '', mri[0, 0, :, :, :])
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(t, "  ", index+1, " saved")

test()