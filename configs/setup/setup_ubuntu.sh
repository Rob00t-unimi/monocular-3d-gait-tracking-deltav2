#!/bin/bash
# echo "Environment creation VGGT..."
# conda env create -f envs/vggt_env.yml
# echo "Additional requirements installation for VGGT..."
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate vggt_env
# pip install -e models/vggt
# conda deactivate

 echo "Environment creation SpaTrackerV2..."
 conda env create -f envs/spatrack_env.yml
 echo "Additional requirements installation for SpaTracker..."
 source "$(conda info --base)/etc/profile.d/conda.sh"
 conda activate spatrack_env
 pip install -r models/SpaTrackerV2/requirements.txt
 conda deactivate

echo "Environment creation Analysis..."
conda env create -f envs/analysis_env.yml

echo "Environment creation DELTA..."
conda env create -f envs/delta_env.yml
echo "Additional requirements installation for DELTA..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate delta_env
pip install pip==24.0
pip install -r models/DELTAv2/requirements.txt
pip install -U "ray[default]" viser
# Install xformers e pytorch3d richiesti da UniDepth
pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.24
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
conda deactivate

echo "Installation completed!"