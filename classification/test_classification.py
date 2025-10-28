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
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, ThreadDataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


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




def launch_test(args):

    ROOT_DIR = args.root_dir
    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"StrokeUADiag/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    ddp_bool = False

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

    test_df = pd.read_csv(ROOT_DIR+f"StrokeUADiag/data_splits_lists/{args.dataset['name']}/test.csv")
    test_files = [{"img": img, "label": label} for img, label in zip(test_df["participant_id"], test_df["high_nihss"])]
    for item in test_files:
        item["img"] = ROOT_DIR+"datasets/StrokeUADiag_classification_inputs/stacked_"+item["img"]+".nii.gz"

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]



    test_transforms = define_instance(args, "test_transforms")
    test_ds = CacheDataset(data=test_files, transform=test_transforms)




    if ddp_bool:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, num_replicas=world_size, rank=rank)
    else:
        test_sampler = None

    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers, pin_memory=True, sampler=test_sampler
    )

    model = define_instance(args, "network_def").to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5 * world_size)
    
    loss_function = torch.nn.CrossEntropyLoss()

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=False)

    


    if ddp_bool:
        # if ddp, distribute data across n gpus
        test_loader.sampler.set_epoch(0)

    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        val_epoch_loss = 0
        for step, test_data in enumerate(test_loader):
            test_images, test_labels = test_data['img'].to(device), test_data['label'].to(device)
            test_outputs = model(test_images)
            print(f"test_outputs: {test_outputs}")
            print(f"test_outputs.argsort(dim=1): {test_outputs.argmax(dim=1)}")
            y_pred_list.extend(test_outputs.argmax(dim=1).cpu().numpy()) #TODO check
            y_true_list.extend(test_labels.cpu().numpy())


    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    # compute accuracy
    accuracy = (y_pred == y_true).sum() / len(y_true)

    # compute precision, recall, f1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    f2 = f1_score(y_true, y_pred, average='weighted', beta=2.0)

    if rank==0:
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-score: {f1:.4f}")
        print(f"Test F2-score: {f2:.4f}")

# TODO 95 confidence intervals