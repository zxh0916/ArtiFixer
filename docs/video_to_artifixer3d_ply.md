# 从视频生成 ArtiFixer3D+ PLY

本文记录本机从原始视频生成高质量 3D Gaussian PLY 的推荐流程。以 `video/104.mp4` 为例，最终输出：

```text
/home/dataset-assist-0/chl_ws/artifixer-data/scene_104/104_artifixer3d_final_960.ply
```

## 环境

推荐使用已配置好的 conda 环境：

```bash
export PYTHON=/home/dataset-assist-0/chl_ws/miniconda3/envs/artifixer/bin/python
export CUDA_HOME=/home/dataset-assist-0/chl_ws/miniconda3/envs/artifixer
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

本机验证过的关键版本：

```text
torch: 2.12.1+cu130
nvcc: 13.0.88
COLMAP: 系统 /usr/bin/colmap
```

不要混用不同来源的 CUDA 头文件。`nvcc`、CUDA headers、`cusparse/cublas/cusolver/nvrtc` 应保持为同一套 CUDA 13.0 conda dev 包。

需要提前准备：

```text
/home/dataset-assist-0/chl_ws/artifixer-data/checkpoints/artifixer-14b.pt
/home/dataset-assist-0/chl_ws/artifixer-data/Wan2.1-T2V-14B-Diffusers
HuggingFace cache: Ruicheng/moge-2-vitl-normal
```

## 一键脚本

端到端脚本位于：

```text
scripts/run_video_to_artifixer3d_ply.sh
```

默认输入为：

```text
/home/dataset-assist-0/chl_ws/video/104.mp4
```

后台执行：

```bash
cd /home/dataset-assist-0/chl_ws/ArtiFixer
nohup bash scripts/run_video_to_artifixer3d_ply.sh \
  > /home/dataset-assist-0/chl_ws/artifixer-data/logs/run_video_to_ply.nohup.log 2>&1 &
```

可用环境变量覆盖路径：

```bash
SCENE_ID=104 \
VIDEO=/path/to/video.mp4 \
WORKSPACE=/home/dataset-assist-0/chl_ws \
bash scripts/run_video_to_artifixer3d_ply.sh
```

脚本默认会复用已完成的中间产物；如果最终 PLY 已存在，会直接退出，避免覆盖。需要强制重跑时设置：

```bash
FORCE=1 bash scripts/run_video_to_artifixer3d_ply.sh
```

## 流程

脚本包含以下阶段：

1. 从原始视频按 `2fps` 抽帧。
2. 使用 COLMAP 做 SfM。
3. 用 `colmap image_undistorter` 转成 3DGRUT 可用的 PINHOLE 相机。
4. 使用 half-covisibility 选择训练视角。
5. 训练基础 3DGRUT 到 `20000` steps，并渲染/估计 metric scale。
6. 生成 960 长边的低分辨率 prepared 副本，同步缩放图片、render、opacity 和相机内参。
7. ArtiFixer 推理生成修复帧。
8. 使用优化 MCMC 参数蒸馏 ArtiFixer3D 到 `30000` steps。
9. 运行 ArtiFixer3D+ 推理。
10. 从 ArtiFixer3D checkpoint 导出最终 PLY。

## 优化参数

```text
long side: 960
num_views: 4
sink_size: 7
local_attn_size: 7
max_neighbors_per_encode: 1
base 3DGRUT steps: 20000
ArtiFixer3D steps: 30000
max_n_gaussians: 800000
opacity_threshold: 0.01
lambda_opacity: 0.02
lambda_scale: 0.03
relocate/add end_iteration: 20000
perturb noise_lr: 300000
```

对应配置：

```text
thirdparty/3DGRUT-ArtiFixer/configs/apps/colmap_3dgut_sparse_mcmc_lpips_optimized.yaml
```

如果该配置文件不存在，脚本会在运行时自动创建。

## 常见问题

- 原始 undistort 分辨率约 `899x1600`，ArtiFixer 14B 在 A100 80GB 上也可能 OOM。推荐使用脚本中的 960 长边路线。
- 低分辨率 prepared 不只缩放图片，还必须同步缩放 `transforms.json` 的 `w/h/fl_x/fl_y/cx/cy`，并保留 `sparse/0/{cameras.bin,images.bin,points3D.bin}`。
- MoGe 模型需要提前缓存；离线运行时保持 `HF_HUB_OFFLINE=1`。
- 如果最终 PLY 仍有明显漂浮高斯，优先检查 COLMAP sparse cloud 和输入中的天空、反光、动态物体区域，再考虑 mask 或 PLY 后处理剪枝。
