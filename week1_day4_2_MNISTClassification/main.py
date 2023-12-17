#!/usr/bin/env python3

import argparse
import sys
import os

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch import optim

from model.models import *
from loss.loss import *
from util.tools import *

def parse_args():
  parser = argparse.ArgumentParser(description="MNIST")

  parser.add_argument("--mode", dest="mode", help="train / eval / test",
                      default=None, type=str)
  parser.add_argument("--download", dest="download", help="download MNIST dataset",
                      default=False, type=bool)
  parser.add_argument("--output_dir", dest="output_dir", help="output directory",
                      default="./output", type=str)
  parser.add_argument("--checkpoint", dest="checkpoint", help="checkpoint trained model",
                      default=None, type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  args = parser.parse_args()

  return args

def get_data():
  download_root = "./mnist_dataset"

  my_transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (1.0, ))
  ])

  train_dataset = MNIST(root=download_root,
                        transform=my_transform,
                        train=True,
                        download=args.download)
  eval_dataset = MNIST(root=download_root,
                       transform=my_transform,
                       train=False,
                       download=args.download)
  test_dataset = MNIST(root=download_root,
                       transform=my_transform,
                       train=False,
                       download=args.download)

  return train_dataset, eval_dataset, test_dataset

def main():
  print(torch.__version__)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

  if torch.cuda.is_available():
    print("gpu")
    device = torch.device("cuda")
  else:
    print("cpu")
    device = torch.device("cpu")

  # Get MNIST Dataset
  train_dataset, eval_dataset, test_dataset = get_data()

  # Make DataLoader
  train_loader = DataLoader(train_dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True)
  eval_loader = DataLoader(eval_dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=0,
                           pin_memory=False,
                           drop_last=False)
  test_loader = DataLoader(test_dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=0,
                           pin_memory=False,
                           drop_last=False)

  # LeNet5
  _model = get_model("lenet5")

  if args.mode == "train":
    model = _model(batch=8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
    model.to(device)
    model.train()

    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    # loss
    criterion = get_criterion(crit="mnist", device=device)

    epoch = 2
    iter = 0 # TODO ??? iter가 굳이 필요한가? 아래 i로 대체 되는데?
    for e in range(epoch):
      total_loss = 0
      for i, batch in enumerate(train_loader):
        img = batch[0]
        img = img.to(device)
        ground_truth = batch[1]
        ground_truth = ground_truth.to(device)

        # forward
        out = model(img)

        # loss
        loss_val = criterion(out, ground_truth) # Mean(1) loss for this batch(8)

        # backpropagation
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss_val.item() # Total(1) loss for batches'(7500) loss

        if iter % 100 == 0:
          print(f"{e} epoch {i} iter loss: {loss_val.item()}")

        iter += 1

      scheduler.step()

      mean_loss = total_loss / i # 평균 of 한 이미지의 loss
      print(f"->{e} epoch mean loss: {mean_loss}")

      torch.save(model.state_dict(), args.output_dir+"/model_epoch"+str(e)+".pt")
    print("End training")
  elif args.mode == "eval":
    model = _model(batch=1, n_classes=10, in_channel=1, in_width=32, in_height=32)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    acc = 0
    num_eval = 0

    for i, batch in enumerate(eval_loader):
      img = batch[0]
      img = img.to(device)
      ground_truth = batch[1]

      # inference
      out = model(img)
      out = out.cpu()
      # print(out)

      if out == ground_truth:
        acc += 1
      num_eval += 1

    print(f"Evaluation score: {acc} / {num_eval}")
  elif args.mode == "test":
    model = _model(batch=1, n_classes=10, in_channel=1, in_width=32, in_height=32)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    acc = 0
    num_eval = 0

    for i, batch in enumerate(test_loader):
      img = batch[0]
      img = img.to(device)
      ground_truth = batch[1]

      # inference
      out = model(img)
      out = out.cpu()
      # print(out)

      show_img(img.cpu().numpy(), str(out.item()))

# /opt/conda/bin/python main.py --mode train --download True --output_dir "./output"
# /opt/conda/bin/python main.py --mode eval --download True --output_dir ./output --checkpoint ./output/model_epoch1.pt
# /opt/conda/bin/python main.py --mode test --download True --output_dir ./output --checkpoint ./output/model_epoch1.pt
if __name__ == "__main__":
  args = parse_args()
  main()

  # TODO ??? Why? eval이랑 test 지금 같은 거 아닌가?
  # print(get_data()[1] == get_data()[2])
