_base_ = [
    "../_base_/datasets/soccernet/features_clips_CALF.py",  # dataset config
    "../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py" # trainer config
]

work_dir = "outputs/contextawarelossfunction/soccernet_resnetpca512"

dataset = dict(
    train=dict(features="ResNET_TF2_PCA512.npy"),
    valid=dict(features="ResNET_TF2_PCA512.npy"),
    test=dict(features="ResNET_TF2_PCA512.npy")
)
log_level = 'INFO'  # The level of logging

# model = dict(
#     neck=dict(
#         type='NetVLAD++',
#         vocab_size=64),
# )