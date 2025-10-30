import torch
import numpy as np
import matplotlib.pyplot as plt

class DDPM:
    def __init__(self,
                 T: int,
                 noise_level_share: bool,
                 device: str,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
    ) -> None:
        ''' T is the max diffusion step. noise_level_share indicate whether add different
        noise levels among the sequence dimension(2nd dimension) '''
        self.T = T
        self.noise_level_share = noise_level_share
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32).to(device)
        self.one_minus_betas = 1 - self.betas
        self.alphas = torch.cumprod(self.one_minus_betas, dim=0) # someplaces use alpha_bar

    def forward(self, x: torch.tensor):
        x = x.to(self.device)
        B, L = x.shape[0], x.shape[1]

        # diffuse
        noise = torch.randn_like(x)
        noise_levels = None
        if self.noise_level_share:
            noise_levels = torch.randint(0, self.T, (B,), device=self.device)
        else:
            noise_levels = torch.randint(0, self.T, (B, L), device=self.device)
        alphas_sqrt = self.alphas[noise_levels].sqrt()
        one_minus_alphas_sqrt = (1 - self.alphas[noise_levels]).sqrt()
        while alphas_sqrt.ndim < x.ndim:
            alphas_sqrt = alphas_sqrt.unsqueeze(-1)
            one_minus_alphas_sqrt = one_minus_alphas_sqrt.unsqueeze(-1)
            assert alphas_sqrt.ndim == one_minus_alphas_sqrt.ndim
        
        x_noise = alphas_sqrt * x + one_minus_alphas_sqrt * noise
        return x_noise, noise, noise_levels
    
    def forward_with_noise_level(self, x: torch.tensor, noise_levels: torch.tensor):
        x = x.to(self.device)
        B, L = x.shape[0], x.shape[1]

        # diffuse
        noise = torch.randn_like(x)
        assert noise_levels.shape == (B,) # or noise_level.shape == (B, L)
        alphas_sqrt = self.alphas[noise_levels].sqrt()
        one_minus_alphas_sqrt = (1 - self.alphas[noise_levels]).sqrt()
        while alphas_sqrt.ndim < x.ndim:
            alphas_sqrt = alphas_sqrt.unsqueeze(-1)
            one_minus_alphas_sqrt = one_minus_alphas_sqrt.unsqueeze(-1)
            assert alphas_sqrt.ndim == one_minus_alphas_sqrt.ndim
        
        x_noise = alphas_sqrt * x + one_minus_alphas_sqrt * noise
        return x_noise, noise
    
    def denoise(self, xt: torch.tensor, noise_pred: torch.tensor, t: torch.tensor):
        '''t is the xt's noise levels in [0, T). t can be two dimension tensor.'''
        assert xt.shape == noise_pred.shape
        xt, noise_pred, t = xt.to(self.device), noise_pred.to(self.device), t.to(self.device)

        beta_t, one_minus_beta, alpha_t = self.betas[t], self.one_minus_betas[t], self.alphas[t]
        while beta_t.ndim < xt.ndim:
            beta_t = beta_t.unsqueeze(-1)
            one_minus_beta = one_minus_beta.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
            assert beta_t.ndim == one_minus_beta.ndim and beta_t.ndim == alpha_t.ndim
        
        noise = torch.randn_like(xt)
        mask = (t == 0)
        noise[mask] = 0
        
        x_t_1 = (1/(one_minus_beta.sqrt())) * (xt - (beta_t / ((1 - alpha_t).sqrt())) * noise_pred) + beta_t.sqrt() * noise
        return x_t_1
    
    def denoise_ddim(self, xt: torch.tensor, noise_pred: torch.tensor, t: torch.tensor, t_next: torch.tensor, eta=0.0):
        '''t is the xt's noise levels in [0, T).
           t_next is the xt's noise levels in [0, T) after denoise.
           t < 0 to indicate the demoise process has already completed, no need to denoise further
           when t<0 and t=0, for these two cases, alpha_t_next can be arbitary value(negative value recommend). This special case is processed
           in alpha_t[t<0] = 1 and alpha_t_next[t<=0] = 1'''
        assert xt.shape == noise_pred.shape and t.shape == t_next.shape
        assert (t > t_next).sum() == t.numel(), f'got {(t > t_next).sum()} and {t.numel()}' # t must greater than t_next
        xt, noise_pred, t, t_next = xt.to(self.device), noise_pred.to(self.device), t.to(self.device), t_next.to(self.device)

        alpha_t, alpha_t_next = self.alphas[t], self.alphas[t_next]
        alpha_t[t<0] = 1 # process denoise already completed case.
        alpha_t_next[t<=0] = 1 # process the last denoise step and denoise already completed cases.
        while alpha_t.ndim < xt.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_next = alpha_t_next.unsqueeze(-1)
            assert alpha_t.ndim == alpha_t_next.ndim
        one_minus_alpha_t, one_minus_alpha_t_next = 1 - alpha_t, 1 - alpha_t_next

        x0_pred = (xt - one_minus_alpha_t.sqrt()*noise_pred) / (alpha_t.sqrt())

        sigma = eta*(
            (((1-alpha_t_next)/(1-alpha_t))*(1-(alpha_t)/(alpha_t_next))).sqrt()
        )
        x_t_direction = (one_minus_alpha_t_next-sigma**2).sqrt()*noise_pred
        noise = torch.randn_like(xt)
        mask = (t == 0)
        noise[mask] = 0

        x_t_next = alpha_t_next.sqrt()*x0_pred + x_t_direction + sigma * noise
        return x_t_next