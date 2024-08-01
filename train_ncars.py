import os
import math
import random
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

from utils.spikefpn_config import spikefpn_cfg
from utils.datasets import Resize_frame, NCARS_SBT
from models.architecture import SpikeFPN_NCARS


def parse_args():
    parser = argparse.ArgumentParser(description="N-CARS Classification")

    parser.add_argument("--device", default=0, help="cuda device, i.e. 0 or cpu") 
    parser.add_argument("--data_path", type=str, default="/dvs_dataset/N-CARS")
    parser.add_argument("--log_path", type=str, default="./log")

    # Basic setting
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--max_epoch", type=int, default=60, help="The upper bound of warm-up")
    parser.add_argument("--lr_epoch", nargs="+", default=[100, 200], type=int, help="lr epoch to decay")
    parser.add_argument("--wp_epoch", type=int, default=1, help="The upper bound of warm-up")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch to train")
    parser.add_argument("-r", "--resume", default=None, type=str, help="keep training")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for SGD")
    parser.add_argument("--input_size", default=256, type=int, help="input size")

    # Optimizer & schedule setting
    parser.add_argument("--optimizer", default="adamw", type=str, help="sgd, adamw")
    parser.add_argument("--lr_schedule", default="step", type=str, help="step, cos")

    # Model setting
    parser.add_argument("-v", "--version", default="SpikeFPN_NCARS")
    parser.add_argument("-t", "--time_steps", default=10, type=int, help="SpikeFPN time steps")
    parser.add_argument("-tf", "--time_per_frame", default=10, type=int, help="SpikeFPN time per frame")
    parser.add_argument("-fs", "--frame_per_stack", default=1, type=int, help="SpikeFPN frame per stack")
    parser.add_argument("--no_warmup", action="store_true", default=False, help="do not or do use warmup")

    return parser.parse_args()


def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split(".")[:-1] + ["conv1", "weight"]
    elif is_b:
        this_str = this_str.split(".")[:-1] + ["snn_optimal", "b"]
    elif is_cell:
        this_str = this_str.split(".")
        index = this_str.index("_ops")
        this_str = this_str[:index]
    else:
        this_str = this_str.split(".")
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append(f"[{value:s}]")
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append("."+value)
    return "".join(new_index)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    args = parse_args()

    print("Setting Arguments: ", args)
    print("----------------------------------------------------------")
    os.makedirs(args.log_path, exist_ok=True)

    if args.device != "cpu":
        print("use cuda:{}".format(args.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")
    else:
        print("use CPU.")
        device = torch.device("cpu")

    print("Model: ", args.version)

    # Load model
    train_size = val_size = args.input_size
    
    train_dataset = NCARS_SBT(
        root_dir = args.data_path,
        mode = "train", 
        time_inteval_ms = args.time_per_frame,
        stacks = args.time_steps, 
        channels = args.frame_per_stack, 
        transform = Resize_frame(train_size)
    )
    val_dataset = NCARS_SBT(
        root_dir = args.data_path,
        mode = "validate", 
        time_inteval_ms = args.time_per_frame, 
        stacks = args.time_steps, 
        channels = args.frame_per_stack, 
        transform = Resize_frame(val_size)
    )
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    val_dataloader = torch.utils.data.DataLoader(
        dataset = val_dataset, 
        shuffle = False,
        batch_size = args.batch_size, 
        pin_memory = True,
    )
    
    # Build model
    anchor_size = spikefpn_cfg["anchor_size_gen1"]
    model = SpikeFPN_NCARS(
        device = device, 
        input_size = train_size, 
        num_classes = 2, 
        cfg = spikefpn_cfg, 
        time_steps = args.time_steps,
        init_channels = args.frame_per_stack,
        args = args
    )
    anchor_size = model.anchor_list

    all_keys = [convert_str2index(name,is_cell=True) for name, _ in model.named_parameters() if "_ops" in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval(f"model.{key:s}.mem")
            mem_keys.append(key)
        except:
            print(key)
            pass
    print("mem", mem_keys)

    model = model.to(device)
    params = sum([param.nelement() for param in model.parameters()])
    print(f"Params: {params / 1e6} M.")

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        shuffle = True,
        batch_size = args.batch_size, 
        pin_memory = True,
    )
    model.set_mem_keys(mem_keys)
    model.train()
    # Keep training
    if args.resume is not None:
        print(f"Keep training model: {args.resume:s}")
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    # Optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    if args.optimizer == "sgd":
        print("Using SGD with momentum.")
        optimizer = optim.SGD(
            model.parameters(), 
            lr = tmp_lr, 
            momentum = args.momentum,
            weight_decay = args.weight_decay
        )
    elif args.optimizer == "adamw":
        print("Using AdamW.")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = tmp_lr, 
            weight_decay = args.weight_decay
        )

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    epoch_size = len(train_dataloader)

    criterion = nn.BCELoss()

    best_acc = -100.
    warmup = not args.no_warmup

    # Start training loop
    for epoch in range(args.start_epoch, max_epoch):
        # Using step LR
        if args.lr_schedule == "step":
            if epoch in args.lr_epoch:
                tmp_lr = tmp_lr * 0.5
                set_lr(optimizer, tmp_lr)
        # Using cosine LR decay
        elif args.lr_schedule == "cos" and not warmup:
            T_max = args.max_epoch - 15
            lr_min = base_lr * 0.1 * 0.1
            if epoch > T_max:
                print("Cosine annealing has done.")
                args.lr_schedule == None
                tmp_lr = lr_min
                set_lr(optimizer, tmp_lr)
            else:
                tmp_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + math.cos(math.pi*epoch / T_max))
                set_lr(optimizer, tmp_lr)
        if epoch in spikefpn_cfg["lr_epoch"]:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (frames, labels) in enumerate(train_dataloader):
            for key in mem_keys:
                exec(f"model.{key:s}.mem=None")

            # Warm-up strategy for learning rate
            ni = iter_i + epoch * epoch_size
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                print("Warm-up has done.")
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            frames = frames.float().to(device)
            labels = labels.float().to(device)
            logits = model(frames).squeeze()
            loss = criterion(logits, labels)

            # NAN checking for loss
            if torch.isnan(loss):
                print("NAN")
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter_i % 100 == 0:
                outstream = (f"[Epoch {epoch + 1}/{max_epoch}][Iter {iter_i:03d}/{epoch_size:03d}][lr {optimizer.param_groups[0]['lr']:.9f}][Loss: {loss:.4f} || pid: {os.getpid()}]")
                with open(f"{args.log_path}/train_log.txt", "a", encoding="utf-8") as file:
                    print(outstream, flush=True, file=file)

        # Validation
        if (epoch + 1) % 1 == 0:
            print("Start Validation.")

            model.set_grid(val_size)
            model.eval()
            with torch.no_grad():
                # Accuracy Data
                total = 0
                correct = 0
                # AUC Score Data
                labels_list = []
                predictions_list = []

                for id_, (frames, labels) in enumerate(val_dataloader):
                    frames = frames.to(device)
                    labels = labels.to(device)
                    logits = model(frames).squeeze()

                    predictions = (logits >= 0.5).float()
                    # Accuracy Collection 
                    total += labels.size(0)
                    correct += (predictions == labels).sum()
                    # AUC Score Collection
                    labels_list.extend(labels.cpu().numpy())
                    predictions_list.extend(predictions.cpu().numpy())

                # Accuracy Calculation
                acc = correct / total
                acc = acc.item()
                # AUC Score Calculation
                auc_score = roc_auc_score(np.array(labels_list), np.array(predictions_list))

            with open(f"{args.log_path}/train_log.txt", "a", encoding="utf-8") as file:
                print(f"Epoch {epoch + 1}, Accuracy: {acc:.4f}, AUC Score: {auc_score:.4f}", flush=True, file=file)

            # Update best accuracy and save model weight
            if acc > best_acc: 
                best_acc = acc
                torch.save(model.state_dict(), f"{args.log_path}/{args.version}_{repr(epoch + 1)}_{str(round(best_acc, 4))}.pth")

            # Set training mode
            model.set_grid(train_size)
            model.train()
