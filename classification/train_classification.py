import argparse
import json
from pathlib import Path
import os
import time
from datetime import timedelta
import sys
sys.path.append("../..")

import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, ThreadDataLoader
from monai.data.utils import pad_list_data_collate
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random
import pandas as pd

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import utils.custom_transforms as custom_transforms
from utils.utils import define_instance


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion training on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device




def launch_train(args):

    ROOT_DIR = args.root_dir
    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"StrokeUADiag/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    ddp_bool = args.gpus > 1  # whether to use distributed data parallel

    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)


    train_df = pd.read_csv(ROOT_DIR+f"StrokeUADiag/data_splits_lists/{args.dataset['name']}/train.csv")
    train_files = [{"img": img, "label": label} for img, label in zip(train_df["participant_id"], train_df["high_nihss"])]
    # replace participant_id with full image path
    for item in train_files:
        item["img"] = ROOT_DIR+"datasets/StrokeUADiag_classification_inputs/stacked_"+item["img"]+".nii.gz"
        item["label"] = int(item["label"])

    val_df = pd.read_csv(ROOT_DIR+f"StrokeUADiag/data_splits_lists/{args.dataset['name']}/val.csv")
    val_files = [{"img": img, "label": label} for img, label in zip(val_df["participant_id"], val_df["high_nihss"])]

    for item in val_files:
        item["img"] = ROOT_DIR+"datasets/StrokeUADiag_classification_inputs/stacked_"+item["img"]+".nii.gz"
        item["label"] = int(item["label"])
    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]



    train_transforms = define_instance(args, "train_transforms")
    train_ds = CacheDataset(data=train_files, transform=train_transforms)


    val_transforms = define_instance(args, "val_transforms")
    val_ds = CacheDataset(data=val_files, transform=val_transforms)


    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler
    )

    model = define_instance(args, "network_def").to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5 * world_size)
    
    loss_function = torch.nn.CrossEntropyLoss()

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=False)
    
    if rank==0:
        os.makedirs(ROOT_DIR+f"StrokeUADiag/tensorboard/{SUB_EXPERIMENT_NAME}", exist_ok=True)
        writer = SummaryWriter(ROOT_DIR+f"StrokeUADiag/tensorboard/{SUB_EXPERIMENT_NAME}")

    max_epochs = args.max_epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = 0
    epoch_loss_values = list()
    metric_values = list()

    scaler = GradScaler("cuda")


    for epoch in range(max_epochs):
        model.train()
        #if rank==0 and args.diffusion_train["lr_scheduler"] != "none":
        #    lr_scheduler.step()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        epoch_loss = 0
        #progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        #progress_bar.set_description(f"Epoch {epoch}")

        #for step, batch in progress_bar:
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data['img'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            

            #progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        if rank==0:
            writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                val_epoch_loss = 0
                for step, val_data in enumerate(val_loader):
                    val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
                    val_outputs = model(val_images)

                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)

                    metric_count += len(value)
                    num_correct += value.sum().item()


                metric = num_correct / metric_count
                metric_values.append(metric)

                #progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            
            if rank==0:

                writer.add_scalar("val_accuracy", metric, epoch)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1

                    if ddp_bool:
                        torch.save(model.module.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))

                    print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current accuracy: {metric:.4f}"
                        f"\nbest accuracy: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    writer.add_scalar("best_accuracy", best_metric, best_metric_epoch)

    print(f"Training complete, best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

