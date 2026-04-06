#!/bin/bash
#SBATCH --job-name=pso_benchmark
#SBATCH --output=benchmarks/results/pso_results_%j.log
#SBATCH --error=benchmarks/error_logs/pso_error_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100            # Explicitly request an A100 GPU on Adroit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load anaconda3/2025.12
module load cudatoolkit/13.0
conda activate cutlass-dsl

export PYOPENCL_COMPILER_OUTPUT=1

# Route Triton cache to Adroit's scratch space to avoid home quota limits
# and to speed up multi-node/repeat compiles. 
export TRITON_CACHE_DIR="/scratch/network/$USER/.triton_cache"

# Tell Triton to print the winning autotune configuration to stdout
export TRITON_PRINT_AUTOTUNING=1

echo "Starting Benchmark on $(hostname)"
flash-pso-baseline
