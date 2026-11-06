# predictable-scaling

## Getting Start

Install Python 3.14 and uv, then run the commands below to download the dependencies:

```bash
uv sync
```

### Flock login

Since the training datasets will be loaded from [Flock](https://flock.cerulean.works), please login with the commands below:

```bash
uv run flock login
```

### Environment Variables

```
HF_TOKEN=<YOUR-HUGGINGFACE-TOKEN>
WANDB_API_KEY=<YOUR-WANDB-KEY>
WANDB_ENTITY=<YOUR-WANDB-PROJECT>

WANDB_BASE_URL=<>   # optional
```