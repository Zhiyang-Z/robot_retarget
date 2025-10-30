import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from einops import rearrange
import numpy as np
from utils.utils import DDPM
from tqdm import tqdm
from utils.viz_robot import viz_robot
import wandb

class Trainer:
    def __init__(
        self,
        train_data_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mark: str
    ) -> None:
        self.mark = mark
        
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.device = f'cuda:{self.rank}'

        self.model = model
        self.model = self.model.to(self.device)
        self.model = DDP(self.model)
        # compile should be after DDP, refer to https://pytorch.org/docs/main/notes/ddp.html
        # self.model = torch.compile(self.model)
        
        self.model.train()

        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()

        self.ddpm = DDPM(1000, True, self.device)

        if self.rank == 0:
            wandb.init(project="robot_retarget", entity="zhiyang")

    def train(self):
        step = -1
        for epoch in range(2000000000):
            self.train_data_loader.sampler.set_epoch(epoch)
            for i, (h_pose, r_pose, r_root, r_rot) in tqdm(enumerate(self.train_data_loader)):
                self.model.train()
                h_pose, r_pose = h_pose.to(self.device), r_pose.to(self.device)

                r_pose_noise, noise, noise_levels = self.ddpm.forward(r_pose)

                self.optimizer.zero_grad()
                avg_loss = torch.zeros((1,), device=self.device)
                avg_grad_norm = torch.zeros((1,), device=self.device)
                noise_pred = self.model(r_pose_noise, noise_levels, h_pose, 0.1)
                loss = self.loss_fn(noise_pred, noise)
                avg_loss[0] = loss.item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                avg_grad_norm[0] = grad_norm.item()
                self.optimizer.step()
                step += 1
                # collect training info
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.AVG)

                # if step % 10000 == 0 and self.rank == 0:
                #     self.sample(h_pose, r_root, r_rot)

                if self.rank == 0:
                    wandb.log({"epoch": epoch}, step=step, commit = False)
                    wandb.log({"grad_norm": avg_grad_norm.item()}, step=step, commit = False)
                    wandb.log({"loss": avg_loss[0].item()}, step=step, commit=True)

                if step % 10000 == 0 and self.rank == 0:
                    torch.save(self.model.state_dict(), f"/home/zzhang18/proj/robot/robot_retarget/saved_models/{self.mark}_{step}.pt")

    @torch.no_grad
    def sample(self, h_pose, r_root, r_rot):
        self.model.eval()
        sample_num_per_rank = h_pose.shape[0]

        noise = torch.randn(sample_num_per_rank, 64, 29).to(self.device)
        ddim_step = np.array(range(0, 1000, 5))
        for noise_idx in reversed(range(len(ddim_step))):
            t = torch.full((sample_num_per_rank,), ddim_step[noise_idx], dtype=torch.long).to(self.device)
            t_next = torch.full((sample_num_per_rank,), ddim_step[noise_idx - 1] if noise_idx > 0 else -1, dtype=torch.long).to(self.device)
            cond_noise_pred = self.model(noise, t, h_pose, 0)
            noise_pred = cond_noise_pred
            noise = self.ddpm.denoise_ddim(noise, noise_pred, t, t_next, 0)

        video = viz_robot("/home/zhiyang/projects/robot/G1/g1_29dof.urdf", r_root[0], r_rot[0], noise[0].cpu().numpy())
        # import imageio.v2 as imageio
        # imageio.mimsave('output.mp4', video, fps=30)

