import os
import tqdm
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from models.architecture import SpikeFPN_NCARS
from utils.spikefpn_config import spikefpn_cfg
from utils.datasets import Resize_frame, NCARS_SBT


def parse_args():
    parser = argparse.ArgumentParser(description="N-CARS Classification")
    
    parser.add_argument("--data_path", type=str, default="/dvs_dataset/N-CARS")
    parser.add_argument("--device", default="0", help="CUDA device, i.e. 0 or cpu")
    
    # Basic setting
    parser.add_argument("-size", "--img_size", default=256, type=int, help="img_size")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    
    # Model setting
    parser.add_argument("--weight", default="./weights/ncars_weight.pth", type=str)
    parser.add_argument("--conf_thresh", default=0.3, type=float, help="NMS threshold")
    parser.add_argument("--nms_thresh", default=0.5, type=float, help="NMS threshold")
    parser.add_argument("-t", "--time_steps", default=10, type=int, help="SpikeFPN time steps")
    parser.add_argument("-tf", "--time_per_frame", default=10, type=int, help="SpikeFPN time per frame")
    parser.add_argument("-fs", "--frame_per_stack", default=1, type=int, help="SpikeFPN frame per stack")

    return parser.parse_args()


def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split(".")[:-1] + ["conv1", "weight"]
    elif is_b:
        this_str = this_str.split(".")[:-1] + ["snn_optimal", "b"]
    elif is_cell:
        this_str = this_str.split(".")[:3]
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
                new_index.append("." + value)
    return "".join(new_index)


if __name__ == "__main__":
    args = parse_args()

    # Set device
    if args.device != "cpu":
        print("Use cuda:{}".format(args.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")
    else:
        print("Use CPU.")
        device = torch.device("cpu")

    print(f"Using weight: {args.weight}.")

    # Define dataset and data loader
    test_dataset = NCARS_SBT(
        root_dir = args.data_path,
        mode = "test", 
        time_inteval_ms = args.time_per_frame, 
        stacks = args.time_steps, 
        channels = args.frame_per_stack, 
        transform = Resize_frame(args.img_size)
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        shuffle = False,
        batch_size = args.batch_size, 
        pin_memory = True,
    )

    # Build model
    model = SpikeFPN_NCARS(
        device = device, 
        input_size = args.img_size, 
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
    print("mem", mem_keys)
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    with torch.no_grad():
        # Accuracy Data
        total = 0
        correct = 0
        # AUC Score Data
        labels_list = []
        predictions_list = []

        for id_, (frames, labels) in enumerate(tqdm.tqdm(test_dataloader)):
            for key in mem_keys:
                exec(f"model.{key:s}.mem=None")

            frames = frames.float().to(device)
            labels = labels.float().to(device)
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

    print(f"Using {args.weight}, Testing Accuracy: {acc:.4f}, AUC Score: {auc_score:.4f}")
