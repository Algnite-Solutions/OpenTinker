# VLM Single-Turn Math (Geometry3K)

**Author:** Siqi Zhu

This example demonstrates training a vision-language model to solve geometry problems from the Geometry3K dataset.

## Prerequisites

1. Complete the [Installation](../README.md#-installation) steps
2. Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the Geo3K Environment (Client Side)

```bash
python opentinker/environment/geo3k/geo3k_server.py --port <env_port>
```


## Step 3: Generate Training Data

```bash
python opentinker/data_preprocess/geo3k.py \
    --local_save_dir=data/geo3k
```

## Step 4: Run Training

```bash
python opentinker/client/geo3k_rl.py \
    tokenizer_path=Qwen/Qwen3-VL-4B-Instruct \
    processor_path=Qwen/Qwen3-VL-4B-Instruct \
    batch_size=16 \
    val_batch_size=32 \
    data_path=data/geo3k/train.parquet \
    val_data_path=data/geo3k/test.parquet \
    num_epochs=1 \
    save_freq=1000 \
    test_freq=5 \
    scheduler_url=http://$SCHEDULER_IP:8000 \
    interaction.config.env_port=8001 \
    interaction.config.env_host=$ENVIRONMENT_IP
```

## Performance

See [wandb run](https://wandb.ai/zsqzz/Open-Tinker/runs/aidfc2y1?nw=nwuserzhusq20) for training metrics and results.
