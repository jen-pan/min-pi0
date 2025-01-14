'''
Adapted from https://github.com/octo-models/octo/blob/main/examples/05_dataloading.ipynb
Using kwargs from https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py
'''
from data.dataset import make_interleaved_dataset
from data.oxe import make_oxe_dataset_kwargs_and_weights

class RLDSInterleavedDataset:
    def __init__(self, config, train=True):
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            config.dataset_mix,
            config.data_path,
            load_proprio=config.load_proprio,
            load_camera_views=("primary",),
        )
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights,
            train=train,
            split=config.get("split", None),
            shuffle_buffer_size=config.shuffle_buffer_size,
            batch_size=8,  # TODO: set in config
            balance_weights=True,
            traj_transform_kwargs=dict(
                # goal_relabeling_strategy="uniform",   # if we want to do goal relabeling
                window_size=config.window_size,  # history
                action_horizon=config.action_horizon, # future actions for action chunking
                subsample_length=100, # subsampling long trajectories improves shuffling
                skip_unlabeled=config.skip_unlabeled,  # skip ones without language annotation
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs={
                    "primary": dict(
                        random_resized_crop=dict(
                            scale=[0.8, 1.0],
                            ratio=[0.9, 1.1],
                        ),
                        random_brightness=[0.1],
                        random_contrast=[0.9, 1.1],
                        random_saturation=[0.9, 1.1],
                        random_hue=[0.05],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                    "wrist": dict(
                        random_brightness=[0.1],
                        random_contrast=[0.9, 1.1],
                        random_saturation=[0.9, 1.1],
                        random_hue=[0.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                },
                resize_size=dict(
                    primary=(224, 224),
                    wrist=(224, 224),
                ),
                num_parallel_calls=config.num_parallel_calls,
            ),
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
        )
        self.dataset = dataset
