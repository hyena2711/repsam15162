"""
    source1(VP): https://github.com/hjbahng/visual_prompting
    source2(AR): https://github.com/savan77/Adversarial-Reprogramming 
"""

import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    def __init__(self, pad_size, image_size):
        super(PadPrompter, self).__init__()

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, pad_size, image_size):
        super(FixedPatchPrompter, self).__init__()
        self.image_size = image_size
        self.pad_size = pad_size
        
        self.patch = nn.Parameter(torch.randn([1, 3, self.pad_size, self.pad_size]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.image_size, self.image_size]).cuda()
        prompt[:, :, :self.pad_size, :self.pad_size] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, pad_size, image_size):
        super(RandomPatchPrompter, self).__init__()
        self.image_size = image_size
        self.pad_size = pad_size
        
        self.patch = nn.Parameter(torch.randn([1, 3, self.pad_size, self.pad_size]))

    def forward(self, x):
        x_ = np.random.choice(self.image_size - self.pad_size)
        y_ = np.random.choice(self.image_size - self.pad_size)

        prompt = torch.zeros([1, 3, self.image_size, self.image_size]).cuda()
        prompt[:, :, x_:x_ + self.pad_size, y_:y_ + self.pad_size] = self.patch

        return x + prompt





def padding(pad_size, image_size):
    return PadPrompter(pad_size, image_size)

def fixed_patch(pad_size, image_size):
    return FixedPatchPrompter(pad_size, image_size)


def random_patch(pad_size, image_size):
    return RandomPatchPrompter(pad_size, image_size)
