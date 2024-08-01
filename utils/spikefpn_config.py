spikefpn_cfg = {
    # Network
    "backbone": "dlight",
    "train_size": 320,
    "val_size": 320,
    "random_size_range": [10, 19],
    # Anchor size
    # Input_size 304x240
    "anchor_size_gen1": [
        [14, 38], [28, 20],  [36, 27],
        [25, 64], [49, 34],  [66, 44],
        [85, 62], [107, 82], [138, 120]
    ],
    # Input_size 256
    "anchor_size_gen1_9": [
        [12, 33], [25, 17], [33, 24],
        [23, 58], [46, 31], [63, 42],
        [79, 58], [97, 78], [147, 112]
    ],
    "anchor_size_gen1_6": [
        [14, 38], [26, 18], [40, 28],
        [59, 44], [85, 63], [119, 97]
    ],
    "anchor_size_gen1_3": [[26, 23], [52, 37], [90, 69]],
    # Train
    "lr_epoch": (90, 120),
    "max_epoch": 150,
    "ignore_thresh": 0.5
}
