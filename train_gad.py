import os
import math
import random
import argparse
import numpy as np
import torch
import torch.optim as optim

from models.architecture import SpikeFPN_GAD
from utils.spikefpn_config import spikefpn_cfg
from utils import create_labels
from utils.tools import ori_target_frame_collate
from utils.tools import calculate_loss_new, get_box_score, resized_box_to_original
from utils.datasets import Resize_frame, GAD_SBT
from utils.criterion import build_criterion
from utils.gad_evaluate import coco_eval


def parse_args():
    parser = argparse.ArgumentParser(description="GAD Object Detection")

    parser.add_argument("--device", default=1, help="cuda device, i.e. 0 or cpu") 
    parser.add_argument("--data_path", type=str, default="/dvs_dataset/GAD")
    parser.add_argument("--log_path", type=str, default="./log")

    # Basic setting
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--max_epoch", type=int, default=30, help="The upper bound of warm-up")
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
    parser.add_argument("-v", "--version", default="SpikeFPN_GAD")
    parser.add_argument("-t", "--time_steps", default=3, type=int, help="SpikeFPN time steps")
    parser.add_argument("-tf", "--time_per_frame", default=20, type=int, help="SpikeFPN time per frame")
    parser.add_argument("-fs", "--frame_per_stack", default=3, type=int, help="SpikeFPN frame per stack")
    parser.add_argument("--conf_thresh", default=0.3, type=float, help="NMS threshold")
    parser.add_argument("--nms_thresh", default=0.5, type=float, help="NMS threshold")
    parser.add_argument("--no_warmup", action="store_true", default=False, help="do not or do use warmup")
    parser.add_argument("--scale_loss", default="batch", type=str, help="scale loss: batch or positive samples")

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
                new_index.append("." + value)
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
        print(f"Using CUDA:{args.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Model: ", args.version)

    train_size = val_size = args.input_size
    
    num_classes = 2
    train_dataset = GAD_SBT(
        root_dir = args.data_path, 
        object_classes = "all", 
        height = 240, 
        width = 304, 
        mode = "train", 
        ms_per_frame = args.time_per_frame, 
        frame_per_sequence = args.frame_per_stack, 
        T = args.time_steps, 
        transform = Resize_frame(train_size), 
        sbt_method = "before"
    )
    val_dataset = GAD_SBT(
        root_dir = args.data_path, 
        object_classes = "all", 
        height = 240, 
        width = 304, 
        mode = "val", 
        ms_per_frame = args.time_per_frame, 
        frame_per_sequence = args.frame_per_stack, 
        T = args.time_steps, 
        transform = Resize_frame(val_size), 
        sbt_method = "before"
    )
    num_train = len(train_dataset)
    train_indices = list(range(num_train))
    random.shuffle(train_indices)
    val_dataloader = torch.utils.data.DataLoader(
        dataset = val_dataset, 
        shuffle = False,
        batch_size = args.batch_size, 
        collate_fn = ori_target_frame_collate,
        num_workers = 0,
        pin_memory = True,
    )
    
    # Build model
    model = SpikeFPN_GAD(
        device = device, 
        input_size = train_size, 
        num_classes = num_classes, 
        cfg = spikefpn_cfg, 
        center_sample = False,
        time_steps = args.time_steps,
        init_channels = args.frame_per_stack,
        args = args
    )
    anchor_size = model.anchor_list

    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if "_ops" in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval(f"model.{key:s}.mem")
            mem_keys.append(key)
        except:
            print(key)
    print("mem", mem_keys)


    model = model.to(device)
    params = sum([param.nelement() for param in model.parameters()])
    print(f"Params: {params / 1e6} M.")

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        shuffle = True,
        batch_size = args.batch_size, 
        collate_fn = ori_target_frame_collate,
        num_workers = 0,
        pin_memory = True
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

    max_epoch = args.max_epoch
    epoch_size = len(train_dataloader)

    criterion = build_criterion(args, spikefpn_cfg, num_classes=num_classes)

    best_map = -100.
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
                tmp_lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * epoch / T_max))
                set_lr(optimizer, tmp_lr)
        if epoch in spikefpn_cfg["lr_epoch"]:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets, original_label, original_frame, file) in enumerate(train_dataloader):

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

            targets = [label.tolist() for label in targets]
            targets = create_labels.gt_creator(
                img_size = train_size, 
                strides = model.stride, 
                label_lists = targets, 
                anchor_size = anchor_size, 
                multi_anchor = True,
                center_sample = False
            )
            
            images = images.float().to(device)
            targets = targets.float().to(device)

            conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred = model(images)

            conf_loss, cls_loss, box_loss, total_loss = calculate_loss_new(conf_pred, cls_pred, x1y1x2y2_pred, targets, criterion)

            # NAN checking for loss
            if torch.isnan(total_loss):
                print("NAN")
                continue

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter_i % 500 == 0:
                outstream = (f"[Epoch {epoch + 1:d}/{max_epoch:d}][Iter {iter_i:04d}/{epoch_size:04d}][lr {optimizer.param_groups[0]['lr']:.6f}][Loss: conf {conf_loss.item():.2f} || cls {cls_loss.item():.2f} || box {box_loss.item():.2f} || total {total_loss.item():.2f} || pid: {os.getpid()}]")
                with open(f"{args.log_path}/train_log.txt", "a", encoding="utf-8") as file:
                    print(outstream, flush=True, file=file)

        # Validation
        if (epoch + 1) % 1 == 0:
            print("Start validation.")
            model_eval = model

            # Set validation mode
            model_eval.set_grid(val_size)
            model_eval.eval()
            batch_num = len(val_dataloader)
            conf_loss_list = []
            cls_loss_list = []
            box_loss_list = []
            total_loss_list = []
            gt_label_list = []
            pred_label_list = []
            classes_name = train_dataset.object_classes
            with torch.no_grad():
                for id_, data in enumerate(val_dataloader):

                    image, targets, original_label, original_frame, file = data
                    for label in original_label:
                        gt_label_list.append(label)
                    targets = [label.tolist() for label in targets]
                    size = np.array([[
                        image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]
                    ]])
                    targets = create_labels.gt_creator(
                        img_size = val_size, 
                        strides = model.stride, 
                        label_lists = targets, 
                        anchor_size = anchor_size, 
                        multi_anchor = True,
                        center_sample = False
                    )
                    image = image.float().to(device)
                    targets = targets.float().to(device)

                    conf_pred, cls_pred, reg_pred, box_pred = model_eval(image)

                    conf_loss, cls_loss, box_loss, total_loss = calculate_loss_new(conf_pred, cls_pred, box_pred, targets, criterion)

                    conf_loss_list.append(conf_loss)
                    cls_loss_list.append(cls_loss)
                    box_loss_list.append(box_loss)
                    total_loss_list.append(total_loss)

                    bboxes, scores, cls_inds = get_box_score(
                        conf_pred, cls_pred, box_pred, 
                        num_classes = num_classes, 
                        conf_thresh = args.conf_thresh, 
                        nms_thresh = args.nms_thresh
                    )
                    bboxes = [box * size for box in bboxes]
                    bboxes = [resized_box_to_original(box, val_size, 240, 304) for box in bboxes]
                    
                    for i in range(len(bboxes)):
                        pred_label = []
                        for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                            x1 = float(box[0])
                            y1 = float(box[1])
                            x2 = float(box[2])
                            y2 = float(box[3])
                            
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            score = float(score) # object score * class score
                            A = {"image_id": id_ * 64 + i, "category_id": cls_ind, "bbox": bbox,
                                "score": score} # COCO JSON format
                            pred_label.append(A)
                        pred_label_list.append(pred_label)
                map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=240, width=304, labelmap=classes_name)
            cur_map = map50
            
            conf_loss_item = sum(conf_loss_list).item() / batch_num
            cls_loss_item = sum(cls_loss_list).item() / batch_num
            box_loss_item = sum(box_loss_list).item() / batch_num
            total_loss_item = sum(total_loss_list).item() / batch_num

            print("val/conf loss", conf_loss_item)
            print("val/cls loss",  cls_loss_item)
            print("val/box loss",  box_loss_item)
            print("val/total loss",  total_loss_item)
            print(f"val mAP(0.5:0.95):{map50_95}, mAP(0.5):{map50}")

            with open(f"{args.log_path}/train_log.txt", "a", encoding="utf-8") as file:
                map_string = f"Epoch {epoch}, Validated mAP(0.5:0.95): {map50_95}, mAP(0.5): {map50}\n"
                file.write(map_string)
            # Update the best mAP
            if cur_map > best_map: 
                best_map = cur_map
                with open(f"{args.log_path}/train_log.txt", "a", encoding="utf-8") as file:
                    best_map_string = f"Validated best mAP(0.5:0.95): {map50_95}, mAP(0.5): {best_map}\n"
                    file.write(best_map_string)
                # Save model
                print("Saving state at epoch ", epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(args.log_path, f"{args.version}_{repr(epoch + 1)}_{str(round(best_map, 4))}.pth"))

            # Set training mode
            model_eval.set_grid(train_size)
            model_eval.train()
