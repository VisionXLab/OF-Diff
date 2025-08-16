#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

image_path  = Path("./data/DIOR/images_train_512")
mask_path   = Path("./data/DIOR/label_train_512")
prompt_path = Path("./data/DIOR/DIOR_prompt")

parent_path = image_path.parent  
out_json    = parent_path / "prompt.json"

valid_img_ext = {".png", ".jpg", ".jpeg"}

mask_dict = {p.stem: p.name for p in mask_path.iterdir() if p.suffix.lower() in valid_img_ext}
prompt_dict = {p.stem: p for p in prompt_path.iterdir() if p.suffix.lower() == ".txt"}

with out_json.open("w", encoding="utf-8") as f_out:
    for img_file in sorted(image_path.iterdir()):
        if img_file.suffix.lower() not in valid_img_ext:
            continue

        stem = img_file.stem

        if stem not in mask_dict:
            print(f"[WARN] no mask for {img_file.name}, skip.")
            continue

        if stem not in prompt_dict:
            print(f"[WARN] no prompt for {img_file.name}, skip.")
            continue

        with prompt_dict[stem].open("r", encoding="utf-8") as fp:
            for line in fp:
                prompt_line = line.strip()
                if prompt_line:
                    break
            else:
                print(f"[WARN] empty prompt in {prompt_dict[stem].name}, skip.")
                continue

        data_dict = {
            "source": f"{mask_path.name}/{mask_dict[stem]}",
            "target": f"{image_path.name}/{img_file.name}",
            "prompt": prompt_line
        }
        f_out.write(json.dumps(data_dict, ensure_ascii=False) + "\n")

print(f"Done. JSONL saved to {out_json}")
