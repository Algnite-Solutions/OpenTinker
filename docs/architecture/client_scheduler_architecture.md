# OpenTinker Client-Scheduler-Server Architecture

This document explains the traffic flow and communication patterns between the Client, Scheduler, and Training Server components in OpenTinker.

## Overview

OpenTinker uses a three-tier architecture for distributed RL training:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Client      │────▶│    Scheduler    │────▶│ Training Server │
│  (math_rl.py)   │     │ (job_scheduler) │     │(http_training)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
   User Code              Job Queue &              Ray Workers
   Data Loading           GPU Allocation          PPO Training
   Environment            Process Management      Model Updates
```

## Component Responsibilities

### 1. Client (`opentinker/client/`)
- Loads training data and creates dataloaders
- Defines the training environment (e.g., MathGame)
- Submits jobs to the scheduler
- Sends training batches to the allocated server
- Handles lifecycle management (cleanup on exit)

### 2. Scheduler (`opentinker/scheduler/job_scheduler.py`)
- Manages GPU resource allocation across jobs
- Maintains a job queue for pending requests
- Spawns HTTP Training Server processes
- Handles user authentication (optional)
- Provides REST API for job management

### 3. Training Server (`opentinker/server/http_training_server.py`)
- Runs PPO training logic with Ray workers
- Receives batches via HTTP from the client
- Manages actor, critic, and reference policy workers
- Computes rewards and updates model weights
- Saves checkpoints

## Traffic Flow

### Phase 1: Job Submission

```
Client                          Scheduler
   │                               │
   │  POST /submit_job             │
   │  {config, num_gpus, ...}      │
   │──────────────────────────────▶│
   │                               │ 1. Allocate GPUs
   │                               │ 2. Find available port
   │                               │ 3. Launch training server
   │                               │ 4. Wait for health check
   │  {job_id, server_url, status} │
   │◀──────────────────────────────│
   │                               │
```

### Phase 2: Server Configuration

```
Client                          Training Server
   │                               │
   │  POST /set_generation_config  │
   │  {temperature, top_p, ...}    │
   │──────────────────────────────▶│
   │                               │
   │  POST /upload_reward_function │
   │  {function_name, source_code} │
   │──────────────────────────────▶│
   │                               │
   │  POST /set_config             │
   │  {config_overrides}           │
   │──────────────────────────────▶│
   │                               │
   │  POST /init_workers           │
   │  {total_steps}                │
   │──────────────────────────────▶│ Initialize Ray workers
   │                               │
```

### Phase 3: Training Loop

```
Client                          Training Server
   │                               │
   │  for batch in dataloader:     │
   │                               │
   │    POST /train_step           │
   │    {batch_data (serialized)}  │
   │──────────────────────────────▶│ 1. Deserialize batch
   │                               │ 2. Generate rollouts
   │                               │ 3. Compute rewards
   │                               │ 4. PPO update
   │    {metrics, global_steps}    │
   │◀──────────────────────────────│
   │                               │
   │    POST /validate (periodic)  │
   │    {batch_data}               │
   │──────────────────────────────▶│
   │    {metrics, samples}         │
   │◀──────────────────────────────│
   │                               │
   │    POST /save_checkpoint      │
   │    (periodic)                 │
   │──────────────────────────────▶│
   │    {checkpoint_dir}           │
   │◀──────────────────────────────│
```

### Phase 4: Job Completion

```
Client                          Scheduler
   │                               │
   │  POST /complete_job/{job_id}  │
   │──────────────────────────────▶│ 1. Kill server process
   │                               │ 2. Kill Ray actors
   │                               │ 3. Release GPUs & port
   │  {status: COMPLETED}          │ 4. Schedule next job
   │◀──────────────────────────────│
```

## Data Serialization

Training batches are serialized as `DataProto` objects for HTTP transmission:

```python
# Serialization (Client → Server)
{
    "batch": {
        "input_ids": {"__type__": "torch.Tensor", "__data__": "<base64>", ...},
        "attention_mask": {"__type__": "torch.Tensor", ...},
        ...
    },
    "non_tensor_batch": {
        "raw_prompts": {"__type__": "numpy.ndarray", "__dtype__": "object", ...},
        ...
    },
    "meta_info": {...}
}
```

Supported types:
- `torch.Tensor` → base64-encoded numpy bytes
- `numpy.ndarray` → base64-encoded bytes (object arrays as lists)
- `PIL.Image` → base64-encoded PNG

## Authentication

When `enable_auth: true` in scheduler config:

```
Authorization: Bearer <api_key>
```

Register a new user:
```bash
curl -X POST "http://scheduler:8765/register?username=myuser"
# Returns: {"api_key": "otk_...", ...}
```

## Error Handling & Retries

The `HTTPTrainingClient` implements exponential backoff:

```python
retry_delay = 5.0  # Initial delay
max_retries = 1000  # For long server initialization

for attempt in range(max_retries):
    try:
        response = session.post(url, json=data)
        return response.json()
    except (Timeout, ConnectionError):
        wait_time = min(retry_delay * (2 ** attempt), 20.0)
        time.sleep(wait_time)
```

## Lifecycle Management

The `SchedulerClientLifecycleManager` handles cleanup:

1. **Normal exit**: `atexit` handler calls `complete_job`
2. **SIGINT/SIGTERM**: Signal handler performs graceful shutdown
3. **Cleanup actions**:
   - Cancel/complete job on scheduler
   - Stop local reward server (if started)
   - Run custom cleanup callbacks

## Port Forwarding (SSH Tunnels)

For remote scheduler access, the client auto-detects SSH forwarding mode:

```python
# If scheduler_url contains 'localhost', rewrite server_url
# Original: http://10.0.0.5:38000
# Modified: http://localhost:38000 (requires SSH tunnel)
```

Set up tunnel:
```bash
ssh -L 8780:localhost:8780 -L 38000:localhost:38000 user@remote
```

## Inference Jobs

The scheduler also supports inference jobs (vLLM servers):

```
POST /submit_inference_job
{
    "model_path": "/path/to/model",
    "tensor_parallel_size": 2,
    "num_gpus": 2
}

Response:
{
    "job_id": "abc123",
    "status": "STARTING",
    "vllm_server_url": "http://10.0.0.5:38001"
}
```

Inference jobs launch vLLM servers that can be used for model evaluation or serving.
