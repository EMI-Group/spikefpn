import os
import tqdm
import random
import argparse
import numpy as np
from src.io import dat_events_tools, npy_events_tools
from numpy.lib import recfunctions as rfn


class GAD(object):
    def __init__(
        self, 
        root_dir, 
        object_classes, 
        height, 
        width, 
        augmentation = False, 
        mode = "train", 
        ms_per_frame = 10, 
        frame_per_sequence = 5, 
        T = 5, 
        shuffle = True, 
        transform = None
    ):
        """
        Creates an iterator over the GAD dataset.

        :param root_dir: path to dataset root
        :param object_classes: list of string containing objects or "all" for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param ms_per_frame: sbt frame interval
        :param ms_per_frame: number of frame per sequence
        :param T: prev T sequence
        :param augmentation: flip, shift and random window start for training
        :param mode: `train`, `test` or `val`
        """

        file_dir = os.path.join("detection_dataset_duration_60s_ratio_1.0", mode)
        self.files = os.listdir(os.path.join(root_dir, file_dir))
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files if time_seq_name[-3:] == "npy"]

        self.root_dir = root_dir
        self.mode = mode
        self.width = width
        self.height = height
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.augmentation = augmentation
        self.transform = transform
        self.window_time = ms_per_frame * 1000 * frame_per_sequence * T

        if object_classes == "all":
            self.nr_classes = 2
            self.object_classes = ["car", "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes

        self.sequence_start = []
        self.sequence_time_start = []
        self.labels = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files,  self.sequence_start))
            random.shuffle(zipped_lists)
            self.files, self.sequence_start = zip(*zipped_lists)
    
    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the unique indices file.
        """
        file_name_bbox_id = []
        print(f"Building GAD `{self.mode}` Dataset.")
        pbar = tqdm.tqdm(total=len(self.files), unit="File", unit_scale=True)

        for i_file, file_name in enumerate(self.files):
            bbox_file = os.path.join(self.root_dir, file_name + "_bbox.npy")
            event_file = os.path.join(self.root_dir, file_name + "_td.dat")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            labels = np.stack([dat_bbox["t"], dat_bbox["x"], dat_bbox["y"], dat_bbox["w"], dat_bbox["h"], dat_bbox["class_id"]], axis=1)
            if len(self.labels) == 0:
                self.labels = labels
            else:
                self.labels = np.concatenate((self.labels, labels), axis=0)

            unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
            for unique_time in unique_ts:
                sequence_start_end = self.searchEventSequence(event_file, unique_time, time_before=self.window_time)
                self.sequence_start.append(sequence_start_end)
                self.sequence_time_start.append(unique_time - self.window_time + 1)

            file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]

            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id
    
    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        bbox_file = os.path.join(self.root_dir, self.files[idx][0] + "_bbox.npy")
        event_file = os.path.join(self.root_dir, self.files[idx][0] + "_td.dat")

        # Bounding Box
        f_bbox = open(bbox_file, "rb")
        """
        dat_bbox types (v_type):
        [("ts", "uint64"), ("x", "float32"), ("y", "float32"), ("w", "float32"), ("h", "float32"), ("class_id", "uint8"), ("confidence", "float32"), ("track_id", "uint32")]
        """
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        bbox_time_idx = self.files[idx][1]

        # Get bounding boxes at current time step
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = dat_bbox[v_type[0][0]].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx+1]

        bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]

        # Required Information ["class_id", "x", "y", "w", "h"]
        np_bbox = rfn.structured_to_unstructured(bboxes)[:, [5, 1, 2, 3, 4]]
        np_bbox = self.cropToFrame(np_bbox)

        label = np.zeros([np_bbox.shape[0], 5])
        label[:np_bbox.shape[0], :] = np_bbox

        # Events
        events = self.readEventFile(event_file, self.sequence_start[idx])
        frame = self.sbt_frame(events, self.sequence_time_start[idx], ms_per_frame=self.ms_per_frame, frame_per_sequence=self.frame_per_sequence, T=self.T)
        if self.transform is not None:
            frame, label = self.transform(frame.reshape(-1, self.height, self.width), label)
        else:
            frame = frame.reshape(-1, self.height, self.width)
        h, w = frame.shape[1], frame.shape[2]
        frame = frame.reshape(self.T, self.frame_per_sequence, h, w)

        return frame.astype(np.int8), label.astype(np.int64) 
    
    def sbt_frame(self, events, start_time, ms_per_frame=10, frame_per_sequence=5, T=5):
        final_frame = np.zeros((T, frame_per_sequence, self.height, self.width))
        num_events = events.shape[0]
        for i in range(num_events):
            total_index = (events[i, 2] - start_time) // (ms_per_frame * 1000)
            frame_index = int(total_index % frame_per_sequence)
            sequence_index = int(total_index // frame_per_sequence)
            final_frame[sequence_index, frame_index, events[i, 1], events[i, 0]] += events[i, 3]
        return np.sign(final_frame)

    def searchEventSequence(self, event_file, bbox_time, time_before=250000):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        start_time = max(0, bbox_time - time_before + 1)

        nr_events = dat_events_tools.count_events(event_file)
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        low = 0
        high = nr_events 
        start_position = 0
        end_position = 0

        while high > low:
            middle = (low + high) // 2

            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[("ts", "u4"), ("_", "i4")], count=1)["ts"][0]

            if mid > start_time:
                high = middle
            elif mid < start_time:
                low = middle + 1
            else:
                break
        file_handle.seek(ev_start + low * ev_size)
        buffer = np.fromfile(file_handle, dtype=[("ts", "u4"), ("_", "i4")], count=high - low)["ts"]
        final_index = np.searchsorted(buffer, start_time, side="left")
        start_position = low + final_index

        low = 0
        high = nr_events 
        while high > low:
            middle = (low + high) // 2

            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[("ts", "u4"), ("_", "i4")], count=1)["ts"][0]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                break
        file_handle.seek(ev_start + low * ev_size)
        buffer = np.fromfile(file_handle, dtype=[("ts", "u4"), ("_", "i4")], count=high-low)["ts"]
        final_index = np.searchsorted(buffer, bbox_time, side="right")
        end_position = low + final_index

        file_handle.close()
        # Now we know it is between low and high
        return start_position, end_position

    def readEventFile(self, event_file, file_position):
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        file_handle.seek(ev_start + file_position[0] * ev_size)
        dat_event = np.fromfile(file_handle, dtype=[("ts", "u4"), ("_", "i4")], count=file_position[1]-file_position[0])
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        y = np.right_shift(np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        p[p == 0] = -1
        events_np = np.stack([x, y, dat_event["ts"], p], axis=-1)

        return events_np

    def cropToFrame(self, np_bbox):
        # Checks if bounding boxes are inside frame. If not crop to border
        pt1 = [np_bbox[:, 1], np_bbox[:, 2]]
        pt2 = [np_bbox[:, 1] + np_bbox[:, 3], np_bbox[:, 2] + np_bbox[:, 4]]
        pt1[0] = np.clip(pt1[0], 0, self.width - 1)
        pt1[1] = np.clip(pt1[1], 0, self.height - 1)
        pt2[0] = np.clip(pt2[0], 0, self.width - 1)
        pt2[1] = np.clip(pt2[1], 0, self.height - 1)
        np_bbox[:, 1] = pt1[0]
        np_bbox[:, 2] = pt1[1]
        np_bbox[:, 3] = pt2[0] - pt1[0]
        np_bbox[:, 4] = pt2[1] - pt1[1]

        return np_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="GAD framing")

    parser.add_argument("-dp", "--data_path", default="/dvs_dataset/GAD/raw")
    parser.add_argument("-sp", "--save_path", default="/dvs_dataset/GAD/framed")
    parser.add_argument("-d", "--device", default="cpu", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("-t", "--time_steps", default=3, type=int, help="SpikeFPN time steps")
    parser.add_argument("-tf", "--time_per_frame", default=20, type=int, help="SpikeFPN time per frame")
    parser.add_argument("-fs", "--frame_per_stack", default=3, type=int, help="SpikeFPN frame per stack")
    parser.add_argument("-dm", "--data_mode", type=str, default="val", help="train, val or test")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = GAD(
        root_dir = args.data_path, 
        object_classes = "all", 
        height = 240, 
        width = 304, 
        augmentation = False, 
        mode = args.data_mode, 
        ms_per_frame = args.time_per_frame,
        frame_per_sequence = args.frame_per_stack, 
        T = args.time_steps, 
        shuffle = False, 
        transform = None
    )
    save_dir = os.path.join(args.save_path, f"sbt_{args.time_per_frame}ms_{args.frame_per_stack}frame_{args.time_steps}stack_before", args.data_mode)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving SBT framed GAD `{args.data_mode}` data frames and labels.")
    pbar = tqdm.tqdm(total=len(dataset), unit="File", unit_scale=True)
    for i, (frame, label) in enumerate(dataset):
        frame_file_path = os.path.join(save_dir, f"sample{i}_frame.npy")
        label_file_path = os.path.join(save_dir, f"sample{i}_label.npy")
        np.save(frame_file_path, frame)
        np.save(label_file_path, label)
        pbar.update(1)
    pbar.close()
    np.save(f"{args.save_path}/{args.data_mode}_box_file_id.npy", dataset.files)
    print("Finished.")
