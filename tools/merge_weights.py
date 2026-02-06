#!/usr/bin/env python

import os
import sys
import torch
import subprocess
import argparse
from pathlib import Path
import shutil
import glob
import tempfile

# Ensure project modules can be imported
sys.path.append(".")

# Import at module level
try:
    import share
    from cldm.model import load_state_dict
except ImportError as e:
    # Not exiting immediately as this might happen during script import, not execution
    # We'll check again in the main function
    pass

def main():
    parser = argparse.ArgumentParser(description="Merge Stable Diffusion weights with ControlNet training weights")
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained model checkpoint, e.g.: ./checkpoints/checkpoints_DIOR_train/model-step=10000.ckpt")
    parser.add_argument("--sd15_path", type=str, default="./models/control_sd15_ini.ckpt", help="Path to SD15 initialization model")
    parser.add_argument("--output_dir", type=str, help="Output directory, defaults to a 'merged' folder in the same directory as the checkpoint")
    parser.add_argument("--use_direct_load", action="store_true", help="Directly load model files instead of using zero_to_fp32.py")
    args = parser.parse_args()

    # Set paths
    checkpoint_path = args.checkpoint_path
    sd15_path = args.sd15_path
    
    # If no output directory is specified, create a 'merged' folder in the same directory as the checkpoint
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, "merged")
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = os.path.join(output_dir, "merged_pytorch_model.pth")
    
    # Find checkpoint subdirectory
    checkpoint_subdir = os.path.join(checkpoint_path, "checkpoint")
    if not os.path.isdir(checkpoint_subdir):
        print(f"Warning: checkpoint subdirectory not found in {checkpoint_path}")
        # Try using checkpoint_path directly
        checkpoint_subdir = checkpoint_path
    
    # Check if latest file exists
    latest_file = os.path.join(checkpoint_subdir, "latest")
    
    # Two modes: using zero_to_fp32.py or direct loading
    if args.use_direct_load or not os.path.exists(latest_file):
        print(f"Step 1: Preparing to load model files directly...")
        
        # Find all mp_rank files
        mp_rank_files = glob.glob(os.path.join(checkpoint_subdir, "mp_rank_*_model_states.pt"))
        
        if not mp_rank_files:
            print(f"Warning: No mp_rank files found in {checkpoint_subdir}")
            raise FileNotFoundError(f"Could not find model files in {checkpoint_subdir}")
        
        print(f"Found model file: {mp_rank_files[0]}")
        
        # Load model file directly
        print(f"Loading model file: {mp_rank_files[0]}")
        model_state_dict = torch.load(mp_rank_files[0], map_location="cpu")
        
        # Extract model weights
        if "module" in model_state_dict:
            sd15_with_control_state_dict = model_state_dict["module"]
        else:
            sd15_with_control_state_dict = model_state_dict
    else:
        print(f"Step 1: Using zero_to_fp32.py to convert checkpoint to fp32 weights...")
        
        # Find zero_to_fp32.py script
        zero_to_fp32_script = os.path.join(checkpoint_path, "zero_to_fp32.py")
        
        if not os.path.exists(zero_to_fp32_script):
            raise FileNotFoundError(f"Could not find zero_to_fp32.py file, please make sure it's in the {checkpoint_path} directory")
        
        print(f"Using zero_to_fp32.py script: {zero_to_fp32_script}")
        print(f"Checking latest file: {latest_file}")
        print(f"Latest file exists: {os.path.exists(latest_file)}")
        
        # Show latest file content
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                latest_content = f.read().strip()
            print(f"Latest file content: {latest_content}")
        
        # Use temporary directory instead of creating a fixed intermediate directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Execute zero_to_fp32.py
            cmd = [
                "python", 
                zero_to_fp32_script, 
                checkpoint_subdir,
                temp_dir
            ]
            
            print(f"Executing command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"zero_to_fp32.py execution failed, error code: {e.returncode}")
                print("Trying to load model files directly...")
                
                # Fall back to direct loading mode
                mp_rank_files = glob.glob(os.path.join(checkpoint_subdir, "mp_rank_*_model_states.pt"))
                
                if not mp_rank_files:
                    print(f"Warning: No mp_rank files found in {checkpoint_subdir}")
                    raise FileNotFoundError(f"Could not find model files in {checkpoint_subdir}")
                
                print(f"Found model file: {mp_rank_files[0]}")
                
                # Load model file directly
                print(f"Loading model file: {mp_rank_files[0]}")
                model_state_dict = torch.load(mp_rank_files[0], map_location="cpu")
                
                # Extract model weights
                if "module" in model_state_dict:
                    sd15_with_control_state_dict = model_state_dict["module"]
                else:
                    sd15_with_control_state_dict = model_state_dict
            else:
                # Check if pytorch_model.bin file was generated
                pytorch_model_bin = os.path.join(temp_dir, "pytorch_model.bin")
                if not os.path.exists(pytorch_model_bin):
                    # Try to find other possible filenames
                    bin_files = list(Path(temp_dir).glob("*.bin"))
                    if bin_files:
                        pytorch_model_bin = str(bin_files[0])
                        print(f"Found weights file: {pytorch_model_bin}")
                    else:
                        raise FileNotFoundError(f"Could not find weights file in {temp_dir}")
                
                # Load converted weights
                print(f"Loading ControlNet weights: {pytorch_model_bin}")
                sd15_with_control_state_dict = load_state_dict(pytorch_model_bin)
    
    print(f"Step 2: Merging SD15 weights with ControlNet weights...")
    
    # Check if necessary modules were successfully imported
    if 'load_state_dict' not in globals():
        print("Error: Could not import necessary modules. Please make sure to run this script from the ControlNet project root directory")
        sys.exit(1)
    
    # Load SD15 weights
    print(f"Loading SD15 weights: {sd15_path}")
    sd15_state_dict = load_state_dict(sd15_path)
    
    # Merge weights
    print("Merging weights...")
    final_state_dict = sd15_state_dict.copy()
    
    # Counter to track how many keys were replaced
    replaced_keys = 0
    
    for key, value in sd15_with_control_state_dict.items():
        if key in final_state_dict:
            print(f"Overriding {key} with ControlNet weights")
            final_state_dict[key] = value
            replaced_keys += 1
    
    print(f"Replaced a total of {replaced_keys} keys")
    
    # Save the result
    print(f"Saving merged model to: {output_path}")
    torch.save(final_state_dict, output_path)
    
    print('Done! Merged model saved to ' + output_path)

if __name__ == "__main__":
    main()
