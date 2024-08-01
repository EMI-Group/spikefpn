import os
import pathlib
import numpy as np
import tqdm
import argparse
from src.io.psee_loader import PSEELoader


def parse_args():
    parser = argparse.ArgumentParser(description="N-CARS framing")
    parser.add_argument("-rp", "--root_path", type=str, default="/dvs_dataset/N-CARS/original")
    parser.add_argument("-sp", "--save_path", type=str, default="/dvs_dataset/N-CARS")
    parser.add_argument("-dc", "--data_class", type=str, default="cars", help="cars or background")
    parser.add_argument("-dm", "--data_mode", type=str, default="train", help="train or test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DATA_PATH = f"{args.root_path}/n-cars_{args.data_mode}/{args.data_class}"
    file_names = [pathlib.Path(item).as_posix() for item in os.scandir(DATA_PATH) if item.is_file()]

    S = 10
    C = 1
    frame_inteval_ms = 10
    for file_name in tqdm.tqdm(file_names):
        video = PSEELoader(file_name) 
        # Acquire Events
        event_list = []
        while not video.done:
            # Load (t, x, y, p) events
            event = video.load_delta_t(frame_inteval_ms * 1e3)
            event_list.append(event)
        event_stream = np.concatenate(event_list)
        height = event_stream["y"].max() + 1
        width = event_stream["x"].max() + 1
    
        # Stacking Based on Time (SBT) Framing
        frames = np.zeros(
            shape=(len(event_list), height, width), 
            dtype=np.int32,
        )
        for index, event in enumerate(event_list):
            # If `event["p"][i]` is 0, set `event["p"][i]` to -1
            frames[index, event["y"], event["x"]] = 2 * event["p"].astype(np.int32) - 1
        frames = frames.reshape(-1, C, *frames.shape[1:])
    
        # Save Data 
        SAVE_PATH = f"{args.save_path}/SBT{frame_inteval_ms}ms_S{S}C{C}/{args.data_mode}_{args.data_class}"
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        np.save(f"{SAVE_PATH}/obj_{file_name.split("/")[-1].split("_")[1]}.npy", frames)
    
    print(f"{args.data_mode} dataset of `{args.data_class}` framing has finished.")
