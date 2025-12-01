#!/usr/bin/env python3
# dependency check
missing = []

try:
    from PIL import Image
except ImportError:
    missing.append("Pillow (import PIL)")

try:
    import imagehash
except ImportError:
    missing.append("imagehash")

try:
    import numpy
except ImportError:
    missing.append("numpy")

if missing:
    print("Missing Python dependencies:")
    for m in missing:
        print(" - " + m)
    print("Install with: pip install pillow imagehash numpy")
    raise SystemExit(1)
import os
import json
import shutil
import imagehash
from PIL import Image
from collections import defaultdict

def flatten_directory(source_dir, dest_dir):
    """Copy all PNG files from source_dir to dest_dir root, flattening structure."""
    print(f"Flattening {source_dir} to {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    files_copied = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)

                # Handle filename conflicts by adding counter
                counter = 1
                base_dest_path = dest_path
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_dest_path)
                    dest_path = f"{name}_{counter}{ext}"
                    counter += 1

                shutil.copy2(source_path, dest_path)
                files_copied += 1

    print(f"  Flattened {files_copied} files to {dest_dir}")
    return files_copied

def find_filename_matches(com_dir, glide_dir, glideout_dir):
    """Find matching filenames between COM and Glide, copy matches to GlideOut."""
    print("Finding filename matches between COM and Glide...")

    # Get all COM filenames
    com_files = set()
    for file in os.listdir(com_dir):
        if file.lower().endswith('.png'):
            com_files.add(file)

    print(f"  COM files: {len(com_files)}")

    # Find and copy matching Glide files
    matches_found = 0
    for root, dirs, files in os.walk(glide_dir):
        for file in files:
            if file.lower().endswith('.png') and file in com_files:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(glideout_dir, file)

                # Handle conflicts
                counter = 1
                base_dest_path = dest_path
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_dest_path)
                    dest_path = f"{name}_{counter}{ext}"
                    counter += 1

                shutil.copy2(source_path, dest_path)
                matches_found += 1

    print(f"  Found {matches_found} filename matches, copied to {glideout_dir}")
    return matches_found

def build_hash_mapping(glideout_dir, ng_dir, max_hamming=8):
    """Build mapping between GlideOut files and NG files by content hash with Hamming distance tolerance."""
    print(f"Building hash mapping between GlideOut and NG (max Hamming: {max_hamming})...")

    # Hash NG files (preserve directory structure)
    print("  Hashing NG files...")
    ng_hashes = {}
    for root, dirs, files in os.walk(ng_dir):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        file_hash = imagehash.phash(img)
                        rel_path = os.path.relpath(file_path, ng_dir)
                        ng_hashes[file_hash] = rel_path
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")

    print(f"    Hashed {len(ng_hashes)} NG files")

    # Hash GlideOut files and find matches with Hamming distance
    print("  Hashing GlideOut files and finding matches...")
    filename_mapping = {}
    matches_found = 0
    perfect_matches = 0
    near_matches = 0

    for file in os.listdir(glideout_dir):
        if file.lower().endswith('.png'):
            file_path = os.path.join(glideout_dir, file)
            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    glide_hash = imagehash.phash(img)

                    # Find best match within Hamming distance
                    best_match = None
                    best_distance = float('inf')

                    for ng_hash, ng_path in ng_hashes.items():
                        distance = int(glide_hash - ng_hash) # Hamming distance
                        if distance <= max_hamming and distance < best_distance:
                            best_match = ng_path
                            best_distance = distance

                    if best_match is not None:
                        filename_mapping[file] = best_match
                        matches_found += 1

                        if best_distance == 0:
                            perfect_matches += 1
                            print(f"    ✓ {file} -> {best_match} (perfect)")
                        else:
                            near_matches += 1
                            print(f"    ≈ {file} -> {best_match} (Hamming: {best_distance})")

            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

    print(f"  Found {matches_found} total matches:")
    print(f"    Perfect matches (Hamming 0): {perfect_matches}")
    print(f"    Near matches (Hamming 1-{max_hamming}): {near_matches}")
    return filename_mapping

def convert_com_files(com_dir, filename_mapping, comout_dir):
    """Convert COM files to NG naming convention and directory structure."""
    print("Converting COM files to NG structure...")

    converted_files = 0
    skipped_files = 0

    for file in os.listdir(com_dir):
        if file.lower().endswith('.png') and file in filename_mapping:
            ng_path = filename_mapping[file]
            source_path = os.path.join(com_dir, file)
            dest_path = os.path.join(comout_dir, ng_path)

            # Create destination directory
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy file
            shutil.copy2(source_path, dest_path)
            converted_files += 1
            print(f"  ✓ {file} -> {ng_path}")
        elif file.lower().endswith('.png'):
            skipped_files += 1
            if skipped_files <= 10:  # Show first 10 skipped
                print(f"  ✗ {file} -> NO MATCH")

    print(f"Converted {converted_files} files, skipped {skipped_files} files")
    return converted_files

def main():
    # Configuration - UPDATE THESE PATHS
    BASE_DIR = os.path.expanduser("~/wip")
    COM_DIR = os.path.join(BASE_DIR, "com")
    GLIDE_DIR = os.path.join(BASE_DIR, "glide")
    NG_DIR = os.path.join(BASE_DIR, "ng")

    # Output directories
    GLIDEOUT_DIR = os.path.join(BASE_DIR, "glideout")
    COMOUT_DIR = os.path.join(BASE_DIR, "comout")

    print("=== Texture Conversion Pipeline (Hamming Distance 8) ===")

    # Step 1: Flatten COM and Glide directories (if needed)
    com_flat_dir = COM_DIR  # Already flat per your setup
    glide_flat_dir = GLIDE_DIR  # Already flat per your setup

    # Step 2: Find filename matches and copy to GlideOut
    matches = find_filename_matches(com_flat_dir, glide_flat_dir, GLIDEOUT_DIR)
    if matches == 0:
        print("No filename matches found! Exiting.")
        return

    # Step 3: Build hash mapping between GlideOut and NG with Hamming distance 8
    filename_mapping = build_hash_mapping(GLIDEOUT_DIR, NG_DIR, max_hamming=8)
    if not filename_mapping:
        print("No content matches found! Exiting.")
        return

    # Step 4: Convert COM files to NG structure
    convert_com_files(com_flat_dir, filename_mapping, COMOUT_DIR)

    print(f"\n=== Conversion Complete ===")
    print(f"Output directory: {COMOUT_DIR}")

if __name__ == "__main__":
    main()
