#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                        # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=03:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N train_classifier_single                     # 🏷 作业名称
#PBS -o logs/pegasus/train_classifier_out.log            # 📤 标准输出日志
#PBS -e logs/pegasus/train_classifier_err.log            # ❌ 错误输出日志

# === 切换到作业提交目录 ===
cd /home/SKIING/chenkaixu/code/Filter_PhaseMix_PyTorch

mkdir -p logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate # 确保先退出任何现有的 Conda 环境
source /home/SKIING/chenkaixu/code/med_atn/bin/activate

# === 可选：打印 GPU 状态 ===
nvidia-smi

NUM_WORKERS=$(nproc)
# 输出当前环境信息
echo "Current working directory: $(pwd)"
echo "Total CPU cores: $NUM_WORKERS, use $((NUM_WORKERS / 4)) for data loading"
echo "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"

# params 
root_path=/work/SKIING/chenkaixu/data/asd_dataset

# === 运行你的训练脚本（Hydra 参数可以加在后面）===
python -m project.main data.root_path=${root_path} train.backbone=3dcnn train.fold=5 train.filter=false train.temporal_mix=true