#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=neobabel-evals
#SBATCH --time=00:40:00
#SBATCH --output=eval_logs/score-evals-slurm-%j.out
#SBATCH --error=eval_logs/score-evals-slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=gcn45

hostname
module purge
source $HOME/.bashrc

module load 2023
module load CUDA/12.1.1
conda activate evalenv6 

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_directory> <output_file>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_FILE="$2"

echo "Evaluating from config: $INPUT_DIR"
echo "Output at : $OUTPUT_FILE"
python evaluation/evaluate_images.py \
    "$INPUT_DIR" \
    --outfile "$OUTPUT_FILE" \
    --model-path "object_detectors"

python evaluation/summary_scores.py "$OUTPUT_FILE"