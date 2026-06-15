import os
import glob
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR_PIGWM
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class PromptIRModel(pl.LightningModule):
    def __init__(self, task_id=0):
        super().__init__()
        self.task_id = task_id
        self.net = PromptIR_PIGWM(decoder=True, task_id=task_id)
        self.loss_fn = nn.L1Loss()
        self.save_hyperparameters('task_id')
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        
        # 记录训练损失（按任务分类）
        if self.task_id == 1:
            self.log("train_loss_task1", loss)
        else:
            self.log("train_loss_task0", loss)
            
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150
        )
        return [optimizer], [scheduler]

def find_latest_checkpoint(ckpt_dir):
    """查找目录中最新的检查点文件"""
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not checkpoint_files:
        return None
    
    # 按修改时间排序（最新文件在前）
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def main():
    print("Options")
    print(opt)
    
    # 1. 创建日志记录器
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name=f"PromptIR-Task{opt.task_id}")
    else:
        logger = TensorBoardLogger(save_dir="logs/")
    
    # 2. 准备检查点目录
    ckpt_dir = f"{opt.ckpt_dir}/task{opt.task_id}/"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 3. 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=5,
        save_top_k=-1,
        filename='promptir-{epoch:03d}'
    )
    
    # 4. 创建数据加载器
    trainset = PromptTrainDataset(opt)
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True,
        drop_last=True, 
        num_workers=opt.num_workers
    )
    
    # 5. 创建或加载模型
    model = None
    
    # 对于任务1，尝试加载任务0的模型
    if opt.task_id == 1:
        task0_ckpt_dir = f"{opt.ckpt_dir}/task0/"
        checkpoint_path = find_latest_checkpoint(task0_ckpt_dir)
        
        if checkpoint_path:
            print(f"🔥 从任务0加载模型: {checkpoint_path}")
            
            # 加载模型（但更新任务ID为1）
            model = PromptIRModel.load_from_checkpoint(
                checkpoint_path, 
                task_id=opt.task_id,  # 重要：更新为任务1
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("✅ 任务0模型成功加载，准备在任务1上继续训练")
        else:
            print("⚠️ 警告：未找到任务0的检查点，将从头开始训练任务1")
    
    # 如果模型尚未创建，创建新模型
    if model is None:
        model = PromptIRModel(task_id=opt.task_id)
        if opt.task_id == 0:
            print("🚀 任务0训练：从头开始")
        else:
            print("🚀 任务1训练：从头开始（因未找到任务0模型）")
    
    # 6. 创建训练器并开始训练
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model=model, train_dataloaders=trainloader)
    
    # 7. 训练完成后的清理工作
    if opt.task_id == 0:
        print("🎯 任务0训练完成！请运行任务1前检查模型保存情况")
    else:
        print("🎯 任务1训练完成！所有任务结束")

if __name__ == '__main__':
    main()