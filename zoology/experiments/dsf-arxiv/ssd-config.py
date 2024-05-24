from zoology.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

import numpy as np

VOCAB_SIZE = 8_192

ID = "2024-05-20"


configs = []
for input_seq_len, num_kv_pairs in [
    (128, 8),
    (256, 16),
    (64, 4),
    (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 32
    elif input_seq_len == 512:
        batch_size = 64
    elif input_seq_len == 256:
        batch_size = 128
    elif input_seq_len == 128:
        batch_size = 256
    else:
        batch_size = 512


    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
        cache_dir="", # TODO: add a directory to cache your data!
    )

    for d_model in [
        64,
        128, 
        256, 
        512,
    ]:
        for d_qk in [
            32,
            64,
            128,
            256,
        ]:
            for lr in np.logspace(-4, -2, 4):
                
                MIXERS = {
                    "s6-2": dict(
                        name="s6-2",
                        kwargs={}
                    ),
                }

                for sequence_mixer in [
                    # "sm-attention",
                    # "lin-attention",
                    # "norm-attention",
                    # "mamba-attention",
                    # "mamba-s6",
                    # "s6",
                    "s6-2",
                ]:

                    if 's6' in sequence_mixer:
                        block_type = "MambaBlock"
                    else:
                        block_type = "TransformerBlock"

                    model = ModelConfig(
                        d_model=d_model,
                        d_qk=d_qk,
                        n_layers=2,
                        block_type=block_type,
                        max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                        vocab_size=VOCAB_SIZE,
                        sequence_mixer=MIXERS[sequence_mixer],
                    )
                    config = TrainConfig(
                        model=model,
                        data=data,
                        learning_rate=lr,
                        max_epochs=64,
                        sweep_id=f"run{ID}-seqlen{input_seq_len}-kv{num_kv_pairs}",
                        run_id=f"{sequence_mixer}-dmodel{d_model}-dqk{d_qk}-lr{lr}",
                        #TODO: add your wandb information here
                        logger=LoggerConfig(
                            key = "",
                            project_name="",
                            entity=""
                        )

                    )
                    configs.append(config)
