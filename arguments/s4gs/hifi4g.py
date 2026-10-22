# ATGS on HiFi4G for the S4GS comparison (docs/experiments/comparisons.md, review R-02 / A-02).
# Used by scripts/compare/run_atgs.sh:
#   python train_long.py -s <converted scene> -m <model>/ --frames_start_end 0 T --configs arguments/s4gs/hifi4g.py --base_path <init.ply>
#   python render.py     -s <converted scene> -m <model>/ --frames_start_end 0 T --configs arguments/s4gs/hifi4g.py --base_path <init.ply> \
#                        --iteration -1 --skip_train --skip_video
# The scene is converted by scripts/compare/convert_hifi4g_atgs.py (one directory per frame + a COLMAP text model), the initial
# anchors by scripts/compare/atgs_init_cloud.py.
#
# Base: arguments/vru/basketball.py, the authors' configuration for VRU Basketball with the hash-grid spacetime encoder
# (hash=True): iterations 100,000, one hash encoder per clip_size=20 frames (auto_encoder_levels: levels = ceil(T / 20)),
# encoder-balanced gradient accumulation, no densification (start_stat=150000000), anchor positions fixed (position_lr = 0),
# time_window_size 5, frame_interval 1. The README runs the same file for 250- and 1400-frame sequences. Everything not listed below is
# inherited unchanged; no hyper-parameter was tuned. Changed keys:
#   llffhold 10 -> 8             the held-out cameras are every 8th camera of COLMAP's images.bin order = utils/camera_split.py
#                                (docs/eval-protocol.md section 2); the converter names the images camPP by that order, which is the
#                                order ATGS indexes cameras in (see the converter's docstring)
#   downsample 2.0 -> 1.0        evaluate and train at the stored image resolution (HiFi4G: the 25% images, 934x525 .. 1057x554, every
#                                camera its own size). The authors halve their images; the protocol compares against the stored GT.
#                                The converter writes the cameras in the pixel units of the stored images, so FoV is right at 1.0.
#   white_background False -> True   HiFi4G's images (image_white_undistortion) have a white background; the rasterizer must render on
#                                white or the model has to paint the background. A property of the data, not a hyper-parameter.
#   base_path                    given on the command line (--base_path): the initial anchor cloud + its <stem>_time_list.npy. On the
#                                command line it wins over this file (utils/general_utils.py merge_hparams); render.py needs the same
#                                --base_path again because the lifetime table is not stored in the model.
# --iterations on the command line also wins over this file (the smoke tests use --iterations 2000).
#
# Not scaled with the sequence length (the authors do not either): iterations stay 100,000 whatever T. For Greeting (70 training
# cameras x 200 frames = 14,000 images, 10 hash encoders) that is 7 passes over the images, 10,000 visits per encoder.
# Memory: every encoder is a tiny-cuda-nn 4D hash grid of 63.46 M float32 parameters (253.9 MB; 16 levels x 8 features, 2^19 entries
# per level, hard-coded in scene/spacetime_hash.py, the hashmap_size below is not read). Weights + gradients + Adam moments take
# 4 x 0.254 GB x ceil(T/20) plus 2.1 GB for the dense 128^3 x 64 static voxel grid: T=120 -> 8.2 GB, 200 -> 12.3, 300 -> 17.4,
# 700 -> 37.7, 1400 -> 73.2, 1500 -> 78.3 GB; Adam's foreach step needs one more encoder-sized copy for a moment, and the rasterizer
# and MLP activations come on top. The authors used an 80 GB A100; a 3090 (24 GB) reaches about 300 frames.
_base_ = '../vru/basketball.py'

ModelParams = dict(
    base_path="",
    llffhold=8,
    downsample=1.0,
    white_background=True,
)
