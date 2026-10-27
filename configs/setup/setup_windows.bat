@echo off
@REM echo Environment creation VGGT...
@REM call conda env create -f envs/vggt_env.yml
@REM echo Additional requirements installation for VGGT...
@REM call conda activate vggt_env
@REM call pip install -e models\vggt
@REM call conda deactivate

echo Environment creation SpaTrackerV2...
call conda env create -f envs/spatrack_env.yml
echo Additional requirements installation for SpaTracker...
call conda activate spatrack_env
call pip install -r models\SpaTrackerV2\requirements.txt
call conda deactivate

 echo Environment creation Analysis...
 call conda env create -f envs/analysis_env.yml

 echo Environment creation DELTA...
 call conda env create -f envs/delta_env.yml
 echo Additional requirements installation for DELTA...
 call conda activate delta_env
 call pip install pip==24.0
 call pip install -r models\DELTAv2\requirements.txt
 call pip install -U "ray[default]" viser
 @REM Nota: xformers e pytorch3d possono richiedere i build-tools di VS 2019/2022
 call pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.24
 call pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
 call conda deactivate

 echo Installation completed!
pause