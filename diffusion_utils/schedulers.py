import torch
import numpy as np
from abc import ABCMeta, abstractmethod

def create_scheduler(config):
    if config.sde.scheduler == "cosine":
        return Cosine(config.sde.beta_min, config.sde.beta_max)
    elif config.sde.scheduler == "sd":
        return CosineSD(config.sde.coef_d)

class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def alpha_std(self, t):
        pass

    def reverse(self, alpha):
        pass


class Cosine(Scheduler):
    def __init__(self, beta_0, beta_1, beta_2=0., beta_3=0.):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3

    def beta_t(self, t):
        return self.beta_0 + self.beta_1 * t + self.beta_2 * t ** 2 + self.beta_3 * t ** 3

    def alpha_std(self, t):
        t = t[:, None, None]
        log_mean_coeff = -1 / 2 * (
                self.beta_0 * t +
                self.beta_1 * (t ** 2) / 2 +
                self.beta_2 * (t ** 3) / 3 +
                self.beta_3 * (t ** 4) / 4
        )
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
        return torch.clip(alpha, 0, 1), torch.clip(std, 0, 1)

    def reverse(self, alpha):
        t = (-1 / 2 * self.beta_0 + np.sqrt(
            (1 / 2 * self.beta_0) ** 2 - (self.beta_1 - self.beta_0) * np.log(alpha))) / (
                    1 / 2 * (self.beta_1 - self.beta_0))
        return t


class CosineSD(Scheduler):
    def __init__(self, d=1):
        self.d = d
        self.t_thr = 0.95

    def beta_t(self, t):
        t = torch.clip(t, 0, self.t_thr)
        tan = torch.tan(np.pi * t / 2)
        beta_t = np.pi * self.d ** 2 * tan * (1 + tan ** 2) / (1 + self.d ** 2 * tan ** 2)
        return beta_t

    def alpha_std(self, t):
        t = t[:, None, None]
        tan = torch.tan(np.pi * t / 2)
        alpha_t = 1 / torch.sqrt(1 + tan ** 2 * self.d ** 2)
        std_t = torch.sqrt(1 - alpha_t ** 2)
        return torch.clip(alpha_t, 0, 1), torch.clip(std_t, 0, 1)
