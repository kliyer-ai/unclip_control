import sys
import os

assert len(sys.argv) == 3, "Args are wrong."

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), "Input model does not exist."
assert not os.path.exists(output_path), "Output filename already exists."
assert os.path.exists(os.path.dirname(output_path)), "Output path is not valid."

import torch
from share import *
from cldm.model import create_model


model = create_model(config_path="./models/cldm_v15_cross.yaml")

pretrained_weights = torch.load(input_path)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]

scratch_dict = model.state_dict()

target_dict = {}

print(scratch_dict.keys())

# log var won't be in the scratch_dict
# so it's not copied over from the pretrained weights (checkpoint)
for k in scratch_dict.keys():
    target_dict[k] = pretrained_weights[k].clone()


model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print("Done.")
