
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


# def compute_psnr_ssim(recoverd, clean):
#     assert recoverd.shape == clean.shape
#     recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
#     clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

#     recoverd = recoverd.transpose(0, 2, 3, 1)
#     clean = clean.transpose(0, 2, 3, 1)
#     psnr = 0
#     ssim = 0

#     for i in range(recoverd.shape[0]):
#         # psnr_val += compare_psnr(clean[i], recoverd[i])
#         # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
#         psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
#         ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)

#     return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


import numpy as np
import torch
from skimage.metrics import structural_similarity

def compute_psnr_ssim(clean, recoverd):
    # 输入clean和recoverd是torch张量，形状为 [batch, C, H, W] 或 [C, H, W]
    # 注意：在计算前我们将它们转换为numpy数组，并调整到 [H, W, C] 的形状
    psnr = 0
    ssim = 0
    valid_count = 0
    
    # 确保张量在CPU上
    clean = clean.cpu()
    recoverd = recoverd.cpu()
    
    for i in range(len(clean)):
        # 转换张量到NumPy数组并调整通道顺序到 [H, W, C]
        clean_np = clean[i].permute(1, 2, 0).numpy()   # 如果原始是[C,H,W] -> [H,W,C]
        recoverd_np = recoverd[i].permute(1, 2, 0).numpy()
        
        # 检查图像尺寸
        h, w = clean_np.shape[:2]
        if h < 7 or w < 7:
            # 跳过太小的图像，或者动态调整窗口大小
            # 这里我们选择动态调整窗口大小
            win_size = min(h, w, 7)
            if win_size % 2 == 0:
                win_size = max(3, win_size - 1)
        else:
            win_size = 7
            
        # 计算PSNR
        mse = np.mean((clean_np - recoverd_np) ** 2)
        if mse == 0:
            temp_psnr = 100
        else:
            temp_psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr += temp_psnr
        
        # 计算SSIM
        try:
            # 尝试使用channel_axis参数（新版skimage）
            ssim_val = structural_similarity(
                clean_np, recoverd_np, 
                data_range=1.0, 
                channel_axis=2,  # 因为我们的数组形状是[H, W, C]，所以通道在轴2
                win_size=win_size
            )
        except TypeError:
            # 如果失败，使用multichannel参数（旧版）
            ssim_val = structural_similarity(
                clean_np, recoverd_np, 
                data_range=1.0, 
                multichannel=True,
                win_size=win_size
            )
        ssim += ssim_val
        valid_count += 1
    
    if valid_count == 0:
        return 0, 0, 0
    
    return psnr / valid_count, ssim / valid_count, valid_count


def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0