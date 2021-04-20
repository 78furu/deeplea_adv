import torch.nn as nn
import functools
import torch


class Score(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.data.channels, nef, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 28 x 28
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * config.data.image_size * config.data.image_size, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * config.data.image_size * config.data.image_size)
        )

        #self.layers = [l for l in self.u_net] + [l for l in self.fc]

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
        return score