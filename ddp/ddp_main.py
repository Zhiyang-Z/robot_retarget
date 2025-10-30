from utils.viz_robot import viz_robot
import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Trainer
from torch.utils.data import DataLoader
from dataset.retarget_dataset import RetargetDataset
from model.retarget_transformer import Retarget_Transformer
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
import yaml
import math

def ddp_main(rank: int, world_size: int, resume: bool):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")
    # configure
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = RetargetDataset(f"/home/zzhang18/proj/robot/",
                              f"/home/zzhang18/proj/robot/robot_retarget/dataset/delete_files.pkl",
                              f"/home/zzhang18/proj/robot/robot_retarget/dataset/frame_list.npy")
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=0))
    
    # for data in dataloader:
    #     h_pose, r_pose, r_root, r_rot = data
    #     viz_robot("/home/zhiyang/projects/robot/G1/g1_29dof.urdf", r_root[0], r_rot[0], r_pose[0])
    #     exit
    # # test dataloader speed
    # for i, data in tqdm(enumerate(dataloader)):
    #     pass
    # model defination
    model = Retarget_Transformer()
    model = model.to(f'cuda:{rank}')
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class=torch.optim.AdamW,
                                        lr=config['pretraining']['learning_rate'],
                                        betas=(0.9, 0.95),
                                        weight_decay=config['pretraining']['weight_decay'])
    # # lr scheduler
    # # Warmup (LR: 0 → base LR)
    # scheduler_warmup = LinearLR(optimizer, start_factor=1.0e-6, end_factor=1, total_iters=config['pretraining']['warmup_iters'])
    # # Cosine decay after warmup
    # scheduler_decay = CosineAnnealingLR(optimizer, T_max=config['pretraining']['lr_decay_iters'], eta_min=config['pretraining']['min_learning_rate'])
    # # Constant LR after decay
    # scheduler_constant = ConstantLR(optimizer, factor=config['pretraining']['min_learning_rate'] / config['pretraining']['learning_rate'], total_iters=1e9) # keep constant forever
    # # Combine them sequentially
    # scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay, scheduler_constant],
    #                          milestones=[config['pretraining']['warmup_iters'], config['pretraining']['warmup_iters'] + config['pretraining']['lr_decay_iters']])
    
    # # load state for continue training
    # if resume:
    #     checkpoint = torch.load(config['path']['load'], "cpu")
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # fast forward the scheduler
    #     for _ in range(checkpoint['global_step']):
    #         scheduler.step()
    #     print(f"checkpoint loaded from {config['path']['load']}")
    # # train
    # grad_accum_steps = math.ceil(config['pretraining']['batch_size'] / (world_size*config['pretraining']['batch_size_per_gpu']*config['pretraining']['pretrain_length']))
    # print(f'grad accum steps: {grad_accum_steps}')
    pre_trainer = Trainer(dataloader, model, optimizer, "retarget")
    print('training start...')
    pre_trainer.train()
    # print('test start...')
    # pre_trainer.test()

    ddp_cleanup()