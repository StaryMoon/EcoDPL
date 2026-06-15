import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR, PromptIR_Origin, PromptIR_EWC, PromptIR_PIGWM, PromptIR_EcoDPL

import lightning.pytorch as pl
import torch.nn.functional as F

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR_PIGWM(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    
    skipped_images = 0
    processed_images = 0

    with torch.no_grad():
        pbar = tqdm(testloader, total=len(testloader))
        for ([clean_name], degrad_patch, clean_patch) in pbar:
            # 检查图片尺寸
            _, _, H, W = degrad_patch.shape
            
            # 如果图片太大则跳过
            if H > 800 and W > 600:
                print(f"跳过大尺寸图片: {clean_name[0]} ({H}x{W})")
                skipped_images += 1
                continue
                
            pbar.set_description(f"处理: {clean_name[0]} ({H}x{W})")
            
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            try:
                restored = net(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
                save_image_tensor(restored, output_path + clean_name[0] + '.png')
                processed_images += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"显存不足，跳过图片: {clean_name[0]} ({H}x{W})")
                    skipped_images += 1
                    # 清理显存
                    torch.cuda.empty_cache()
                else:
                    raise

        print(f"Denoise sigma={sigma}: psnr: {psnr.avg:.2f}, ssim: {ssim.avg:.4f}")
        print(f"处理图片: {processed_images}, 跳过图片: {skipped_images} (尺寸过大或显存不足)")



def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    
    skipped_images = 0
    processed_images = 0

    with torch.no_grad():
        pbar = tqdm(testloader, total=len(testloader))
        for ([degraded_name], degrad_patch, clean_patch) in pbar:
            # 检查图片尺寸
            _, _, H, W = degrad_patch.shape
            
            # 如果图片太大则跳过
            if H > 800 and W > 600:
                print(f"跳过大尺寸图片: {degraded_name[0]} ({H}x{W})")
                skipped_images += 1
                continue
                
            pbar.set_description(f"处理: {degraded_name[0]} ({H}x{W})")
            
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            try:
                restored = net(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

                save_image_tensor(restored, output_path + degraded_name[0] + '.png')
                processed_images += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"显存不足，跳过图片: {degraded_name[0]} ({H}x{W})")
                    skipped_images += 1
                    # 清理显存
                    torch.cuda.empty_cache()
                else:
                    raise

        print(f"{task}: PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")
        print(f"处理图片: {processed_images}, 跳过图片: {skipped_images} (尺寸过大或显存不足)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="train_dehaze_ckpt/epoch=239-step=60000.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 设置CUDA设备
    torch.cuda.set_device(testopt.cuda)
    
    # 启动时清理显存
    torch.cuda.empty_cache()

    ckpt_path = testopt.ckpt_name
    
    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)


    print("加载模型: {}".format(ckpt_path))

    net  = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    
    # 冻结所有参数
    for param in net.parameters():
        param.requires_grad = False
    
    # 估算模型大小
    param_size = sum(p.numel() for p in net.parameters())
    print(f"模型参数: {param_size/1e6:.2f}M")
    
    # 测试模式分发
    try:
        if testopt.mode == 0:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print('开始 {} 测试 Sigma=15...'.format(name))
                test_Denoise(net, testset, sigma=15)

                print('开始 {} 测试 Sigma=25...'.format(name))
                test_Denoise(net, testset, sigma=25)

                print('开始 {} 测试 Sigma=50...'.format(name))
                test_Denoise(net, testset, sigma=50)
                
        elif testopt.mode == 1:
            print('开始雨纹去除测试...')
            derain_base_path = testopt.derain_path
            for name in derain_splits:
                print('开始测试 {} 雨纹去除...'.format(name))
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")
                
        elif testopt.mode == 2:
            print('开始去雾测试...')
            derain_base_path = testopt.derain_path
            name = derain_splits[0]
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="dehaze")
            
        elif testopt.mode == 3:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print('开始 {} 测试 Sigma=15...'.format(name))
                test_Denoise(net, testset, sigma=15)

                print('开始 {} 测试 Sigma=25...'.format(name))
                test_Denoise(net, testset, sigma=25)

                print('开始 {} 测试 Sigma=50...'.format(name))
                test_Denoise(net, testset, sigma=50)



            derain_base_path = testopt.derain_path
            print(derain_splits)
            for name in derain_splits:

                print('开始测试 {} 雨纹去除...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")

            print('开始SOTS去雾测试...')
            test_Derain_Dehaze(net, derain_set, task="dehaze")
            
    finally:
        # 最终清理
        torch.cuda.empty_cache()
        print("测试完成，显存已清理")