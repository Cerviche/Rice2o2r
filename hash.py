#!/usr/bin/env python3
"""
Enhanced Texture Hash Generator for Next-Gen Pipeline
=====================================================

Step 1 of 3 in the enhanced Glide → SoH texture conversion pipeline.

This version is designed to work with ENHANCED versions of map.py and convert.py.
It provides richer data while maintaining backward compatibility.

Key Features:
-------------
1. Multiple hash algorithms (phash, dhash, ahash, whash)
2. Alpha channel hashing for transparent textures
3. MD5 checksums for exact duplicate detection
4. Duplicate analysis and reporting
5. Parallel processing for speed
6. Comprehensive metadata and statistics

Output Structure:
-----------------
{
  "metadata": {...},                     # Enhanced metadata
  "hashes": {"path": "phash"},           # Backward compatible
  "extended": {                          # Enhanced data for map.py
    "path": {
      "algorithms": {"phash": "...", "dhash": "...", ...},
      "md5": "checksum",
      "alpha_hash": "...",               # Optional
      "metadata": {...}
    }
  },
  "analysis": {...}                      # Duplicate analysis, statistics
}
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# DEPENDENCY CHECKING
# ---------------------------------------------------------------------

def check_dependencies():
    """Ensure all required packages are available."""
    missing = []

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    try:
        import imagehash
    except ImportError:
        missing.append("imagehash")

    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        for dep in missing:
            print(f"  ❌ {dep}")
        print("\nInstall with: pip install Pillow imagehash tqdm")
        print("=" * 60)
        return False

    return True

if not check_dependencies():
    sys.exit(1)

from PIL import Image
import imagehash
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Hash algorithms available
# phash: Primary algorithm (backward compatibility)
# Others: For conflict resolution in map.py
HASH_ALGORITHMS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
    "whash": imagehash.whash,
}

DEFAULT_HASH_SIZE = 16      # 16x16 = 256-bit hash
DEFAULT_NORMALIZE_SIZE = (256, 256)

# ---------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------

def scan_directory(directory):
    """Find all PNG files with basic validation."""
    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.lower().endswith('.png'):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, directory)

            # Quick PNG validation
            try:
                with open(full_path, 'rb') as f:
                    if f.read(8) == b'\x89PNG\r\n\x1a\n':
                        files.append((rel_path, full_path))
            except OSError:
                continue  # Skip unreadable files

    return files


def calculate_md5(filepath):
    """Compute MD5 checksum for exact file comparison."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def normalize_image(img, target_size=None, preserve_alpha=False):
    """
    Prepare image for consistent hashing.

    If preserve_alpha=True and image has alpha channel:
    - Keep RGBA mode (hash algorithms will use RGB only)
    - Alpha hash calculated separately
    """
    if target_size:
        try:
            img = img.resize(target_size, Image.Resampling.BICUBIC)
        except AttributeError:
            img = img.resize(target_size, Image.BICUBIC)

    if img.mode in ('P', 'PA'):
        img = img.convert('RGBA')

    if img.mode == '1':
        return img.convert('L')

    if img.mode == 'L':
        return img

    if img.mode == 'RGBA':
        if preserve_alpha:
            return img  # Keep RGBA for alpha hashing
        # Composite onto white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background

    if img.mode == 'RGB':
        return img

    return img.convert('RGB')


def compute_alpha_hash(img, hash_size):
    """Calculate hash of alpha channel only."""
    if img.mode != 'RGBA':
        return None

    alpha = img.split()[-1]  # Extract alpha channel
    return str(imagehash.phash(alpha, hash_size=hash_size))


def compute_all_hashes(img, hash_size):
    """Calculate all configured hash algorithms."""
    results = {}
    failures = {}

    for name, func in HASH_ALGORITHMS.items():
        try:
            results[name] = str(func(img, hash_size=hash_size))
        except Exception as e:
            failures[name] = str(e)

    return results, failures


def process_single_file(args):
    """Process one file (for parallel execution)."""
    rel_path, full_path, options = args

    try:
        with Image.open(full_path) as img:
            # Get basic metadata
            file_md5 = calculate_md5(full_path)
            metadata = {
                'original_mode': img.mode,
                'original_size': img.size,
                'original_format': img.format,
                'file_md5': file_md5,
                'file_size': os.path.getsize(full_path)
            }

            # Calculate alpha hash if requested
            alpha_hash = None
            if options.get('alpha_hash', False):
                alpha_hash = compute_alpha_hash(img, options['hash_size'])

            # Normalize and compute hashes
            normalized = normalize_image(
                img,
                target_size=options.get('normalize_size'),
                preserve_alpha=options.get('preserve_alpha', False)
            )

            hashes, failures = compute_all_hashes(normalized, options['hash_size'])

            if not hashes:
                return False, {
                    'file': rel_path,
                    'error': 'All hash algorithms failed',
                    'failures': failures,
                    'metadata': metadata
                }

            # ✅ Added success flag
            return True, {
                'file': rel_path,
                'filename': os.path.basename(rel_path),
                'hashes': hashes,
                'alpha_hash': alpha_hash,
                'failures': failures,
                'metadata': metadata,
                'success': True
            }

    except Exception as e:
        return False, {
            'file': rel_path,
            'error': str(e),
            'type': type(e).__name__
        }


# ---------------------------------------------------------------------
# INTERACTIVE WORKFLOW
# ---------------------------------------------------------------------

def collect_user_input():
    """Get configuration from user interactively."""
    print("\n" + "=" * 60)
    print("ENHANCED TEXTURE HASH GENERATOR")
    print("=" * 60)

    # Get directory
    while True:
        directory = input("\n📁 Enter texture directory: ").strip()
        if directory.startswith('~'):
            directory = os.path.expanduser(directory)

        if os.path.exists(directory):
            break
        print(f"❌ Directory not found: {directory}")

    # Get format
    print("\n🏷️  Texture format:")
    print("  1. Glide/Rice (community texture packs)")
    print("  2. SoH/o2r (game reference textures)")

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice == '1':
            format_type = 'glide'
            break
        elif choice == '2':
            format_type = 'soh'
            break

    # Get processing options
    print("\n⚙️  Processing options (Enter for defaults):")

    hash_size = input(f"  Hash size [{DEFAULT_HASH_SIZE}]: ").strip()
    hash_size = int(hash_size) if hash_size else DEFAULT_HASH_SIZE

    alpha_hash = input("  Enable alpha channel hashing? (y/N): ").lower().strip() == 'y'

    preserve_alpha = input("  Preserve alpha in preprocessing? (y/N): ").lower().strip() == 'y'

    normalize_size = DEFAULT_NORMALIZE_SIZE
    disable_norm = input("  Disable image resizing? (y/N): ").lower().strip() == 'y'
    if disable_norm:
        normalize_size = None

    parallel = input(f"  Use parallel processing? (Y/n): ").lower().strip() != 'n'

    return {
        'directory': directory,
        'format': format_type,
        'hash_size': hash_size,
        'alpha_hash': alpha_hash,
        'preserve_alpha': preserve_alpha,
        'normalize_size': normalize_size,
        'parallel': parallel
    }


def analyze_results(results):
    """Analyze hashing results for duplicates and patterns."""
    analysis = {
        'duplicates_by_md5': defaultdict(list),
        'duplicates_by_phash': defaultdict(list),
        'filename_conflicts': defaultdict(list),
        'statistics': Counter()
    }

    for rel_path, data in results.items():
        if not data.get('success'):
            continue

        # Group by MD5 (exact file duplicates)
        md5 = data['metadata']['file_md5']
        analysis['duplicates_by_md5'][md5].append(rel_path)

        # Group by phash (perceptual duplicates)
        phash = data['hashes'].get('phash')
        if phash:
            analysis['duplicates_by_phash'][phash].append(rel_path)

        # Check filename conflicts
        filename = os.path.basename(rel_path)
        analysis['filename_conflicts'][filename].append(rel_path)

        # Count alpha textures
        if data.get('alpha_hash'):
            analysis['statistics']['alpha_textures'] += 1

    return analysis


def main():
    """Main interactive workflow."""
    config = collect_user_input()

    # Scan files
    print(f"\n🔍 Scanning {config['directory']}...")
    files = scan_directory(config['directory'])

    if not files:
        print("❌ No PNG files found")
        return

    print(f"  Found {len(files)} PNG files")

    # Prepare processing options
    options = {
        'hash_size': config['hash_size'],
        'alpha_hash': config['alpha_hash'],
        'preserve_alpha': config['preserve_alpha'],
        'normalize_size': config['normalize_size']
    }

    # Process files
    print("\n🔨 Processing files...")
    results = {}
    errors = []
    stats = Counter()

    if config['parallel'] and len(files) > 10:
        # Parallel processing
        from multiprocessing import Pool
        cpu_count = os.cpu_count() or 4

        with Pool(cpu_count) as pool:
            tasks = [(rel, full, options) for rel, full in files]

            for success, data in tqdm(
                pool.imap_unordered(process_single_file, tasks),
                total=len(tasks),
                desc="Hashing"
            ):
                if success:
                    results[data['file']] = data
                    stats['success'] += 1
                else:
                    errors.append(data)
                    stats['failed'] += 1
    else:
        # Sequential processing
        for rel_path, full_path in tqdm(files, desc="Hashing"):
            success, data = process_single_file((rel_path, full_path, options))
            if success:
                results[data['file']] = data
                stats['success'] += 1
            else:
                errors.append(data)
                stats['failed'] += 1

    # Analyze results
    print("\n📊 Analyzing results...")
    analysis = analyze_results(results)

    # Generate output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{config['format']}_hashes_{timestamp}.json"

    output = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced',
            'format': config['format'],
            'directory': config['directory'],
            'total_files': len(files),
            'successful': stats['success'],
            'failed': stats['failed'],
            'processing_options': options
        },

        # Backward compatible section
        'hashes': {
            path: data['hashes']['phash']
            for path, data in results.items()
            if 'phash' in data['hashes']
        },

        # Enhanced section (for new map.py)
        'extended': {
            path: {
                'algorithms': data['hashes'],
                'alpha_hash': data.get('alpha_hash'),
                'metadata': data['metadata']
            }
            for path, data in results.items()
        },

        # Analysis section
        'analysis': {
            'duplicates': {
                'exact_files': {
                    md5: paths[:10]  # Limit to first 10
                    for md5, paths in analysis['duplicates_by_md5'].items()
                    if len(paths) > 1
                },
                'perceptual_matches': {
                    phash: paths[:10]
                    for phash, paths in analysis['duplicates_by_phash'].items()
                    if len(paths) > 1
                }
            },
            'statistics': {
                'total_files': len(files),
                'successful': stats['success'],
                'alpha_textures': analysis['statistics'].get('alpha_textures', 0),
                'exact_duplicates': sum(1 for paths in analysis['duplicates_by_md5'].values()
                                       if len(paths) > 1),
                'perceptual_duplicates': sum(1 for paths in analysis['duplicates_by_phash'].values()
                                           if len(paths) > 1)
            }
        },

        'errors': errors[:100]  # Limit errors
    }

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("HASHING COMPLETE")
    print("=" * 60)
    print(f"✅ Output: {output_file}")
    print(f"📊 Success: {stats['success']}/{len(files)} files")

    dup_count = output['analysis']['statistics']['perceptual_duplicates']
    if dup_count > 0:
        print(f"🔄 Perceptual duplicates found: {dup_count} sets")
        if config['format'] == 'soh':
            print("   This is normal for SoH textures (game reuses textures)")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run this for BOTH Glide and SoH textures")
    print("2. Use ENHANCED map.py for better matching")
    print("3. Enhanced map.py will use multiple algorithms")
    print("4. Enhanced convert.py will handle special cases")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
