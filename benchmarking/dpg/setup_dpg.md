
## DPG Benchmark setup


### Janus

```bash
# Clone the repository
git clone https://github.com/deepseek-ai/Janus.git
cd Janus

# Create conda environment
conda create -p /.../janus_env python=3.10
conda activate /.../janus_env

# Install project and dependencies
pip install uv
uv pip install -e .
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 # for H100, CUDA 12.1
MAX_JOBS=128 uv pip install flash-attn --no-build-isolation
uv pip install --upgrade transformers
uv pip install omegaconf
```
  > CUDA 11.8 works with `module cuda/12.1.1`. Also fixes issues with `safetensors`.

Modify the script to handle `NoneType` error in `modeling_utils.ALL_PARALLEL_STYLES` if need be.


### BLIP3o

```bash
# Create conda environment
conda create -p /.../blio python=3.11
conda activate /.../blio

# Install dependencies
pip install uv
uv pip install --upgrade pip wheel setuptools
uv pip install -r requirements.txt
uv pip install --reinstall torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
MAX_JOBS=128 uv pip install flash-attn --no-build-isolation
uv pip install omegaconf
```

### Showo

> **Note:** If NeoBabel environment is already installed, Showo runs out-of-the-box.


## Generation

Depending on the model, change .py file in `generate_sharded.sh`. 

```bash
sbatch generate_sharded.sh configs/Neobabel_dpg.yaml
```


To evaluate the generated outputs:

```bash
# Clone and setup the ELLA repo
git clone https://github.com/TencentQQGYLab/ELLA.git
cd ella
# Follow their setup instructions under 📊 DPG-Bench section in their readme.md
```