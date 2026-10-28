# ATGS on Neural3DV (coffee_martini) for the S4GS comparison (docs/research-log.md 2026-10-28: beat MoDec-GS and ATGS on the same machine
# and protocol). Used by scripts/compare/run_atgs.sh:
#   python train_long.py -s <converted scene> -m <model>/ --frames_start_end 0 T --configs arguments/s4gs/neural3dv.py --base_path <init.ply>
#   python render.py     -s <converted scene> -m <model>/ --frames_start_end 0 T --configs arguments/s4gs/neural3dv.py --base_path <init.ply> \
#                        --iteration -1 --skip_train --skip_video
# The scene is converted by scripts/compare/convert_hifi4g_atgs.py --layout neural3dv (one directory per frame + a COLMAP text model of
# the 18 cameras), the initial anchors by scripts/compare/atgs_init_cloud.py. Same pipeline as arguments/s4gs/hifi4g.py.
#
# Base: arguments/s4gs/hifi4g.py (= arguments/vru/basketball.py of the authors, hash-grid spacetime encoder, 100,000 iterations, one hash
# encoder per 20 frames, no densification, anchor positions fixed); everything not listed below is inherited unchanged, in particular the
# iteration count (100,000, not scaled with T). No hyper-parameter was tuned. Differences from hifi4g.py (all of them data properties):
#   llffhold 8 -> 100            ATGS holds out range(0, #cameras, llffhold) of its camera list (scene/dataset_readers.py,
#                                readColmapSceneInfo), which cannot express "cam00 only" for any small llffhold. The converter puts cam00
#                                at position 0 (utils/camera_split.py: Neural3DV's one test camera is cam00; docs/eval-protocol.md
#                                section 2) and any llffhold >= #cameras then selects position 0 and nothing else. 100 is the most
#                                cameras the converter's two-digit file names allow, so it holds for any N3DV scene. No reader patch.
#   white_background True -> False   Neural3DV is a real scene: the rasterizer renders on black, as the S4GS Neural3DV protocol does.
#   downsample stays 1.0         the stored images are the undistorted 1340x1003 images and the cameras.txt written by the converter is
#                                in their pixel units (scale 1), so the field of view equals this repo's loader up to ATGS's floor() of
#                                the focal length (730.7 -> 730 px, 0.1% of the FoV).
#   base_path                    as in hifi4g.py: given on the command line (initial anchor cloud + its <stem>_time_list.npy), wins over this file.
# Memory: 300 frames -> 15 hash encoders, about 17.4 GB for weights, gradients and Adam moments plus activations; see hifi4g.py for the
# arithmetic. A 24 GB card is at its limit (hifi4g.py: "a 3090 reaches about 300 frames").
_base_ = '../vru/basketball.py'  # same base file as hifi4g.py; its three data keys are set again below

ModelParams = dict(
    base_path="",
    llffhold=100,
    downsample=1.0,
    white_background=False,
)
