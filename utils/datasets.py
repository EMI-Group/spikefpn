import os
import cv2
import torch
import numpy as np


class Resize_frame(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, frame, label):
        frame = frame.transpose(1, 2, 0)
        h_original, w_original = frame.shape[0], frame.shape[1]
        r = min(self.input_size / h_original, self.input_size / w_original)
        h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
        resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation=cv2.INTER_NEAREST)
        h_pad, w_pad = self.input_size - h_resize, self.input_size - w_resize
        h_pad /= 2
        w_pad /= 2
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
        final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        final_label = np.zeros_like(label)
        if label is not None:
            final_label[:, 0] = label[:, 0]
            final_label[:, 1:] = np.round(label[:, 1:] * r)
            final_label[:, 1] = np.round(final_label[:, 1] + w_pad)
            final_label[:, 2] = np.round(final_label[:, 2] + h_pad)
        else:
            final_label = None
        if len(final_frame.shape) == 2:
            final_frame = np.expand_dims(final_frame, axis=-1)
        return final_frame.transpose(2, 0, 1), final_label


class GAD_SBT(object):
    def __init__(
        self, 
        root_dir, 
        object_classes, 
        height, 
        width, 
        mode = "train", 
        ms_per_frame = 10, 
        frame_per_sequence = 5, 
        T = 5, 
        transform = None, 
        sbt_method = "mid"
    ):
        self.file_dir = os.path.join(root_dir, f"sbt_{ms_per_frame}ms_{frame_per_sequence}frame_{T}stack_{sbt_method}", mode)
        self.files = os.listdir(self.file_dir)
        self.box_file_id = np.load(os.path.join(root_dir, f"{mode}_box_file_id.npy"))
        self.root_dir = root_dir
        self.mode = mode
        self.width = width
        self.height = height
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.transform = transform
        self.sbt_method = sbt_method
        if object_classes == "all":
            self.nr_classes = 2
            self.object_classes = ["car", "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes
    
    def __len__(self):
        return len(self.files) // 2

    def __getitem__(self, idx):
        """
        returns frame and label, loading them from files
        :param idx:
        :return: x,y, label
        """
        frame = np.load(os.path.join(self.file_dir, "sample{}_frame.npy".format(idx))).astype(np.float32)
        label = np.load(os.path.join(self.file_dir, "sample{}_label.npy".format(idx))).astype(np.float32)
        file, id = self.box_file_id[idx]
        if self.transform is not None:
            resized_frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = resized_frame.shape[1], resized_frame.shape[2]
            resized_frame = resized_frame.reshape(self.T, self.frame_per_sequence, h, w)
            return resized_frame, resized_label, label, frame, file
        return frame, label, label, file


class NCARS_SBT(object):
    def __init__(
        self, 
        root_dir, 
        mode = "train", 
        time_inteval_ms = 10, 
        stacks = 10,
        channels = 1, 
        transform = None, 
    ):
        self.data_dir: list[str] = []
        self.label: list[float] = []
        for category in ("cars", "background"):
            file_dir = f"{root_dir}/SBT{time_inteval_ms}ms_S{stacks}C{channels}/{mode}_{category}"
            data_files = os.listdir(file_dir)
            self.data_dir.extend(
                [f"{file_dir}/{data_file}" for data_file in data_files]
            )
            self.label.extend(
                [1.0 if category == "cars" else 0.0] * len(data_files)
            )
        self.stacks = stacks
        self.channels = channels
        self.transform = transform
    
    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        frame = np.load(self.data_dir[index]).astype(np.float32)
        label = self.label[index]
        if self.transform is not None:
            frame, _ = self.transform(frame.reshape(-1, *frame.shape[2:]), None)
            frame = frame.reshape(-1, self.channels, *frame.shape[1:])
        frame = torch.from_numpy(frame)
        if frame.shape[0] < self.stacks:
            padding = frame[-1].repeat(self.stacks-frame.shape[0], 1, 1, 1)
            frame = torch.cat((frame, padding), dim=0)
        label = torch.tensor(label, dtype=torch.float32)
        return frame, label
