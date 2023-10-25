import os 
import hashlib
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Tuple

import torch 
from torch.utils.data import TensorDataset, DataLoader

from zoology.config import DataConfig


@dataclass
class SyntheticData:
    """Simple dataclass which specifies the format that should be returned by
    the synthetic data generators.

    All tensors (train_inputs, train_labels, test_inputs, test_labels) should be
    have two axes and share the same second dimension length.

    Args:
        train_inputs (torch.Tensor): Training inputs of shape (num_train_examples, input_seq_len)
        train_labels (torch.Tensor): Training labels of shape (num_train_examples, input_seq_len)
        test_inputs (torch.Tensor): Test inputs of shape (num_test_examples, input_seq_len)
        test_labels (torch.Tensor): Test labels of shape (num_test_examples, input_seq_len)
    """

    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    test_inputs: torch.Tensor
    test_labels: torch.Tensor

    def check_shapes(
        self,
        num_train_examples: int,
        num_test_examples: int,
        input_seq_len: int,
    ):
        """Check that the shapes are correct
        this is useful to catch bugs in the data generation code because
        downstream errors due to incorrectly shaped can be tricky to debug.
        """
        if self.train_labels.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_labels shape is {self.train_labels.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.train_labels.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_labels shape is {self.train_labels.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.test_inputs.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_inputs shape is {self.test_inputs.shape} but should be {(num_test_examples, input_seq_len)}"
            )

        if self.test_labels.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_labels shape is {self.test_labels.shape} but should be {(num_test_examples, input_seq_len)}"
            )


def prepare_data(config: DataConfig) -> Tuple[DataLoader]:
    """
    Prepares the data for training and testing.
    This function checks if a cache directory is available and if the data is already 
    cached. If the data is cached, it loads the data from the cache. If not, it 
    generates the data using the provided configuration. The generated data is then 
    saved to the cache for future use. The function also checks if the shapes of the 
    data are correct. Finally, it prepares the data loaders for training and testing.
    
    Args: 
        config (DataConfig): The configuration object containing all the necessary parameters to prepare the data.
    Returns: 
        Tuple[DataLoader, DataLoader]: A tuple containing the training and testing data loaders.
    Raises: 
        ValueError: If the shapes of the data are not correct.
    Example: 
        >>> config = DataConfig(…) 
        >>> train_dl, test_dl = prepare_data(config) 
    """
    if config.cache_dir is not None:
        try:
            Path(config.cache_dir).mkdir(exist_ok=True, parents=True)
        except:
            print(f"Could not create cache directory {config.cache_dir}")
            config.cache_dir = None
    cache_path = _get_cache_path(config)
    # check cache
    if config.cache_dir is not None and os.path.exists(cache_path) and not config.force_cache:
        # load from cache
        print(f"Loading data from on-disk cache at {cache_path}...") 
        # SE 09-12-23: there's some sporadic issue in torch load that gives
        # RuntimeError: PytorchStreamReader failed reading file data/2: file read failed
        MAX_RETRIES = 10
        for _ in range(MAX_RETRIES):
            try:
                data = SyntheticData(**torch.load(cache_path))
                break
            except RuntimeError as e:
                print(e)
    else:
        print(f"Generating dataset...") 
        builder = config.builder.instantiate()

        # generate data
        data: SyntheticData = builder(
            num_test_examples=config.num_test_examples,
            num_train_examples=config.num_train_examples,
            input_seq_len=config.input_seq_len,
            vocab_size=config.vocab_size,
            seed=config.seed,
        )

        if config.cache_dir is not None:
            print(f"Saving dataset to on-disk cache at {cache_path}...") 
            torch.save(asdict(data), cache_path)
    data.check_shapes(
        num_train_examples=config.num_train_examples, 
        num_test_examples=config.num_test_examples,
        input_seq_len=config.input_seq_len,
    )

    
    # pass x and y to dataloader each offset by one
    train_dl = DataLoader(
        TensorDataset(data.train_inputs, data.train_labels),
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(data.test_inputs, data.test_inputs),
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
    )

    return train_dl, test_dl


def _get_cache_path(config: DataConfig):
    if config.cache_dir is None:
        return None
    config_hash = hashlib.md5(
        json.dumps(config.dict(), sort_keys=True).encode()
    ).hexdigest()

    return os.path.join(
        config.cache_dir,
        f"data_{config_hash}.pt",
    )
