#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

image_path = "./data/DIOR/images_train_512"
mask_path = "./data/DIOR/label_train_512"

parent_path = Path(image_path).parent

image_names = sorted(os.listdir(image_path))
mask_names = sorted(os.listdir(mask_path))

with open(f"{parent_path}/prompt.json", "w") as f:
    for image_name, mask_name in zip(image_names, mask_names):
        if image_name.split(".")[-1] in ("png", "jpg"):
            data_dict = {"source": os.path.join(mask_path.split("/")[-1], mask_name), "target": os.path.join(image_path.split("/")[-1], image_name), "prompt": ""}
            f.write(json.dumps(data_dict)+"\n")
