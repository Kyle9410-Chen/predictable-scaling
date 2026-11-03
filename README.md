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

Create the `.env` file in the root with:

```bash
S3_ACCESS_KEY_ID=<your-access-key-id>
S3_SECRET_ACCESS_KEY=<your-secret-access-key>
```

- running with `.env` file:

```bash
uv run --env-file .env predictable-scaling
```
