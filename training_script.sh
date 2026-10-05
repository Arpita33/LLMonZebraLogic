#!/bin/bash
#SBATCH --account=asaparov
#SBATCH --partition=a100-40gb
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --job-name rl-qwen4b_8gpu
#SBATCH --mem=256G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

echo "=== Submitted Script Header ==="
cat <<'SLURM_HEADER'
#!/bin/bash
#SBATCH --account=asaparov
#SBATCH --partition=a100-40gb
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --job-name rl-qwen4b_newreward
#SBATCH --mem=256G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
SLURM_HEADER
echo "==============================="
#memory leak - > ssh - > nvdia-smi - top
echo "New reward training with Qwen3 4B, using 1000 puzzles(N_Max=4, M_Max=4) of max difficulty 15 conflicts."
echo " Setting the reward score of partial match to 0.1 * (matched/total)"


###more code here

