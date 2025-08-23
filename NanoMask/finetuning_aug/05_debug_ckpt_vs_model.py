import os, sys, re, inspect, torch
from pprint import pprint

if len(sys.argv) < 4:
    print("Usage: python 05_debug_ckpt_vs_model.py <TASK_DIRNAME> <FOLD> <CKPT_PATH>")
    print("Example: python 05_debug_ckpt_vs_model.py Task200_CT2PET_FT 0 /platforms/radiomics/NanoMask/finetuning_aug/pretrained/Task006_CT2PET_fold0/model_final_checkpoint.model")
    sys.exit(1)

TASK, FOLD, CKPT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# Infer workspace root from TASK path layout you use
FT_ROOT = "/platforms/radiomics/NanoMask/finetuning_aug"
os.environ['nnUNet_raw_data_base'] = f"{FT_ROOT}/nnUNet_data/nnUNet_raw_data"
os.environ['nnUNet_preprocessed']   = f"{FT_ROOT}/nnUNet_data/pre_data"
os.environ['RESULTS_FOLDER']        = f"{FT_ROOT}/nnUNet_data/nnUNet"

# Build the same configuration nnUNet_train uses
from nnunet.run.default_configuration import get_default_configuration

net = '3d_fullres'
trainer_name = 'nnUNetTrainerV2'
plans_id = 'nnUNetPlansv2.1'

cfg = get_default_configuration(net, TASK, trainer_name, plans_id)
plans_file, output_folder, dataset_directory, batch_dice, stage, trainer_class = cfg

# Instantiate trainer with only the args it supports (version-safe)
sig = inspect.signature(trainer_class.__init__)
cand = dict(plans_file=plans_file, fold=FOLD, output_folder=output_folder,
            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage)
kwargs = {k:v for k,v in cand.items() if k in sig.parameters}
trainer = trainer_class(**kwargs)

# Initialize network without starting training
init_sig = inspect.signature(trainer.initialize)
try:
    trainer.initialize(False)  # common signature: initialize(training)
except TypeError:
    # try with keyword if required by your version
    try: trainer.initialize(training=False)
    except TypeError:
        # last resort: call with no args (older code)
        trainer.initialize()

net_sd = trainer.network.state_dict()

# Load checkpoint
ckpt = torch.load(CKPT, map_location='cpu')
state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

missing_in_model = []
shape_mismatch = []
extra_in_model = []

for k, w in state.items():
    if k not in net_sd:
        missing_in_model.append(k); continue
    if tuple(w.shape) != tuple(net_sd[k].shape):
        shape_mismatch.append((k, tuple(w.shape), tuple(net_sd[k].shape)))

for k in net_sd:
    if k not in state:
        extra_in_model.append(k)

print(f"# keys in ckpt: {len(state)}; in model: {len(net_sd)}")
print(f"Missing in model: {len(missing_in_model)} (first 20)")
print("\n".join(missing_in_model[:20]))
print(f"\nShape mismatches: {len(shape_mismatch)} (first 30)")
for k, s_ck, s_md in shape_mismatch[:30]:
    print(f"{k}: ckpt{s_ck} != model{s_md}")
print(f"\nExtra in model: {len(extra_in_model)} (first 20)")
print("\n".join(extra_in_model[:20]))
