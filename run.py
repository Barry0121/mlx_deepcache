# Example training loop with frozen feature LeNet and MixNet
import os
from pathlib import Path
import logging
from argparse import ArgumentParser
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch import nn, optim
from mixnet import MixNet
from Lenet.extract_weight import get_feature_model

# Parse input arguments
argparser = ArgumentParser()
# Metadata
argparser.add_argument("--epochs", type=int, default=50)
argparser.add_argument("--lr", type=float, default=2.e-3)
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--from_checkpoint", type=bool, default=True)
# MixNet Parameters
argparser.add_argument("--n", type=int, default=3)
argparser.add_argument("--aux", type=int, default=0)
argparser.add_argument("--max_iter", type=int, default=40)
argparser.add_argument("--eps", type=int, default=0.0001)
argparser.add_argument("--prox_lam", type=int, default=0.01)
argparser.add_argument("--weight_normalize", type=bool, default=True)
# LeNet's Parameters / Other input CNNs
argparser.add_argument("--freeze_cnn", type=bool, default=True)
args = argparser.parse_args()

# Configuration of logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [
                        logging.FileHandler(filename=Path(os.path.join("exps", f"n={args.n}_aux={args.aux}_freeze={args.freeze_cnn}", "logfile.log")),
                                            mode='a'),
                        logging.StreamHandler()
                    ])

# Instantiate dataset, model, and optimizer
class LeNetMixNet(nn.Module):
    def __init__(self, n, aux, max_iter, eps, prox_lam, weight_normalize, freeze):
        self.mn = MixNet(n=n, aux=aux, max_iter=max_iter, eps=eps, prox_lam=prox_lam, weight_normalize=weight_normalize)
        self.lenet_features = get_feature_model(freeze=freeze)

    def forward(self, x):
        # TODO: test and change accordingly
        v_i = self.lenet_features(x)
        z_o = self.mn(v_i)
        return z_o

    def train(self, status):
        self.mn.train(status)


model = LeNetMixNet(args.n, args.aux, args.max_iter, args.eps, args.prox_lam, args.weight_normalize, args.freeze_cnn)
test_mn = MixNet(args.n, args.aux, args.max_iter, args.eps, args.prox_lam, args.weight_normalize)

logging.info(str(model))
logging.info(str(test_mn))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(test_mn.parameters(), args.lr)
# optimizer = optim.Adam(model.parameters(), args.lr) # Try this after mn runs
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)

checkpoint_path = Path(f"models/n={args.n}_aux={args.aux}_freeze={args.freeze_cnn}_checkpoint.pt")
if args.from_checkpoint and os.path.exists(checkpoint_path):
    logging.info(f"learning rate = {args.lr}, batch size = {args.batch_size}, checkpoint = {checkpoint_path}.")
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
else:
    logging.info(f"learning rate = {args.lr}, batch size = {args.batch_size}.")

# TODO: write dataset here
train_dataset = None
validation_dataset = None
test_dataset = None

# Train, validation and test
train_losses = []
val_losses = []
best_loss = float('inf')
logging.info("Starting model training...")
with logging_redirect_tqdm():
    # Support starting from non-zero epochs
    if args.from_checkpoint and os.path.exists(checkpoint_path):
        epoch_iterator = tqdm.trange(start_epoch, args.epochs)
    else:
        epoch_iterator = tqdm.trange(args.epochs)
    for e in epoch_iterator:
        # Training loop
        model.train(True)
        count_train_loss = 0
        for i, (input, target) in enumerate(train_dataset):
            optimizer.zero_grad()
            pred = model(input)
            train_loss = criterion(pred, target)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            count_train_loss += train_loss
        train_losses.append(count_train_loss/(i+1))

        # Validation loop
        model.train(False)
        val_loss = 0
        with torch.no_grad:
            for i, (input, target) in enumerate(validation_dataset):
                pred = model(input)
                val_loss += criterion(pred, target)
            val_losses.append(val_loss/(i+1))

        logging.info(f"Train loss = {train_losses[-1]}, Validation loss = {val_losses[-1]}")

        # Intermediate validation and save model weight
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
logging.info("Model training completed.")

# Test evaluation
model.train(False)
test_loss = 0
with torch.no_grad:
    for i, (input, target) in enumerate(test_dataset):
        pred = model(input)
        test_loss += criterion(pred, target)
logging.info(f"Final evaluation on test dataset: loss = {test_loss/(i+1)}")