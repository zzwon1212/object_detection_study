#!/usr/bin/env python3

import sys

import torch
import torch.nn as nn

class MNISTloss(nn.Module):
  def __init__(self, device=torch.device("cpu")):
    super(MNISTloss, self).__init__()
    self.loss = nn.CrossEntropyLoss().to(device)

  def forward(self, out, ground_truth):
    loss_val = self.loss(out, ground_truth)
    return loss_val

def get_criterion(crit="mnist", device=torch.device("cpu")):
  if crit == "mnist":
    return MNISTloss(device=device)
  else:
    print("unknown criterion")
    sys.exit(1)
