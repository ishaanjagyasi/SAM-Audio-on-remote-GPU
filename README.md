# SAM Audio Source Separation on a Remote GPU 

This setup runs Meta/Facebook SAM Audio Large for text conditioned source separation on any remote GPU machine (cloud or on‑prem). The working script uses fp16 and chunked processing to avoid CUDA out of memory.

## Working

Given an input audio file and a text prompt (example: "female vocals"), the script produces two files.

- `<prefix>_target.wav`  The described source
- `<prefix>_residual.wav`  Everything else

## Files

- `sam_audio_optimised.py`  Final working script

## Requirements

- A remote GPU machine with NVIDIA drivers working (any provider or your own server)
- Python 3.11
- Git
- A Hugging Face account and access token
- A Python environment with these packages installed
  - torch
  - soundfile
  - sam_audio

Optional but useful
- torchaudio

## Hardware requirements (practical)

### GPU VRAM

These are practical fp16 or bf16 inference numbers reported by users in the official SAM Audio GitHub repo.

- SAM Audio Small: about 12 GB VRAM
- SAM Audio Base: about 14 GB VRAM
- SAM Audio Large: about 17 GB VRAM

For SAM Audio Large, plan for at least 24 GB VRAM for a stable experience, especially if you process longer audio and want headroom for peak allocations. Chunking long audio is required in practice to avoid OOM.

### System RAM

- Recommended: 32 GB RAM
- Minimum: 16 GB RAM (may be tight depending on container, caching, and audio length)

### Disk

- Allow at least 20 GB free for Hugging Face cache, model weights, and outputs.

The exact peak VRAM depends on audio duration and settings. Long clips can still OOM without chunking, even on very large GPUs, so keep chunking enabled.

- A remote GPU machine with NVIDIA drivers working (any provider or your own server)
- Python 3.11
- Git
- A Hugging Face account and access token
- A Python environment with these packages installed
  - torch
  - soundfile
  - sam_audio

Optional but useful
- torchaudio

## Setup order of operations

Run these steps on the remote GPU machine.

### 1. Create and activate a Python 3.11 venv

```bash
python3.11 -m venv ~/sam_env
source ~/sam_env/bin/activate
```

### 2. Install PyTorch for your CUDA version

Check your CUDA version first:

```bash
nvidia-smi
```

Then install PyTorch matching that CUDA version.

Example for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Example for CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install SAM Audio from GitHub

```bash
pip install git+https://github.com/facebookresearch/sam-audio.git
```

### 4. Pin Hugging Face dependencies (compatibility)

```bash
pip uninstall -y transformers huggingface_hub
pip install "huggingface_hub<1" "transformers<5"
```

### 5. Authenticate with Hugging Face

```bash
hf auth login
```

Paste your Hugging Face access token when prompted.

### 6. Request access to gated weights

The model `facebook/sam-audio-large` is gated on Hugging Face.

1. Go to the model page: `facebook/sam-audio-large`
2. Click Request access
3. Wait for approval from Meta/Facebook

### 7. First run downloads weights automatically

On first run, Hugging Face will download and cache the weights for:

- `facebook/sam-audio-large`

Later runs reuse the local cache.

## Verify GPU is available

On the remote GPU machine

```bash
nvidia-smi
```

You should see the RTX 5090 and no unexpected processes using VRAM.

## Environment setup (fresh machine on any provider)

Run these commands in this order on the Vast instance.

```bash
python3.11 -m venv ~/sam_env
source ~/sam_env/bin/activate

# Check CUDA version first
nvidia-smi

# Install PyTorch matching your CUDA version
# Example for CUDA 12.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Example for CUDA 12.4:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/facebookresearch/sam-audio.git

pip uninstall -y transformers huggingface_hub
pip install "huggingface_hub<1" "transformers<5"

hf auth login
```

---


## Run separation

Example

```bash
python sam_audio_optimised.py \
  --audio Audio_input.wav \
  --desc "female vocals" \
  --out female_vocals \
  --rerank 1
```

Outputs

- `female_vocals_target.wav`
- `female_vocals_residual.wav`

## Arguments

- `--audio`  Path to input audio file (wav recommended)
- `--desc`  Text description of the source (example: "female vocals", "drums")
- `--out`  Output file prefix
- `--rerank`  Must be 1 or higher for this SAM Audio implementation. Use 1 for lowest memory usage.

## Why chunking is used

Long audio can exceed GPU memory even on large GPUs. The script splits the input into fixed length chunks, runs separation per chunk, then concatenates outputs.

To change chunk length, edit this line inside `sam_audio_optimised.py`

```python
chunk_sec = 10
```

If you still see OOM errors, reduce `chunk_sec`.

## Transfer files between your local machine and the remote GPU machine

You can use `ssh` to connect and `scp` to copy files.

You need four pieces of information:

- User (example: root, ubuntu, ec2-user)
- Host (public IP or hostname)
- SSH port (often 22, sometimes custom)
- Remote path (a directory on the server, example: /workspace/ or ~/work/)

### SSH into the server

```bash
ssh -p <PORT> <USER>@<HOST>
```

If you use an SSH key:

```bash
ssh -i /path/to/key.pem -p <PORT> <USER>@<HOST>
```

### Find your remote path

After you SSH in:

```bash
pwd
```

Use the printed path as <REMOTE_PATH> in the commands below.

### Pull results to local Downloads (macOS)

```bash
scp -P <PORT> <USER>@<HOST>:<REMOTE_PATH>/female_vocals_target.wav ~/Downloads/
scp -P <PORT> <USER>@<HOST>:<REMOTE_PATH>/female_vocals_residual.wav ~/Downloads/
```

### Push script from local to server

```bash
scp -P <PORT> sam_audio_optimised.py <USER>@<HOST>:<REMOTE_PATH>/
```

If you use an SSH key, add `-i /path/to/key.pem` to the scp commands.

### Optional: copy an entire folder

```bash
scp -P <PORT> -r ./my_project_folder <USER>@<HOST>:<REMOTE_PATH>/
```

## Notes

- The script prints `dtype before cuda` and `dtype after cuda` to confirm fp16 is active.
- You may see warnings about pynvml, timm, meshgrid, and RobertaModel initialization. These are expected and do not block inference.

