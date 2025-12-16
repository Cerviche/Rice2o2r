#!/usr/bin/env python3
"""
Enhanced Texture Pack Converter for Next-Gen Pipeline
=====================================================

Step 3 of 3 in enhanced Glide → SoH texture conversion pipeline.

This version works with ENHANCED map.py output and provides:
1. Support for enhanced CSV format with confidence scores
2. Better validation and error handling
3. Quality filtering based on confidence scores
4. Enhanced reporting and statistics
5. Optional visual verification copies

Key Features:
-------------
- Loads enhanced CSV maps (with confidence scores, algorithm distances)
- Quality filtering: Can skip low-confidence matches
- Better duplicate handling with validation
- Comprehensive reporting with confidence distribution
- Optional copy of original Glide files for visual comparison
- Progress tracking with detailed statistics

Workflow:
---------
1. Load enhanced CSV map from map.py v2.0
2. Optionally filter by confidence/distance thresholds
3. Scan texture pack for PNG files
4. Copy matched textures to ALL SoH locations
5. Generate enhanced reports with confidence metrics
6. Provide quality analysis and recommendations
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Default quality thresholds
QUALITY_THRESHOLDS = {
    'high': {'max_distance': 5.0, 'min_confidence': 0.7},
    'medium': {'max_distance': 10.0, 'min_confidence': 0.4},
    'low': {'max_distance': 20.0, 'min_confidence': 0.0}
}

# Field mappings for backward compatibility
LEGACY_FIELDS = ['glide_filename', 'soh_primary_path', 'soh_all_paths',
                 'hamming_distance', 'glide_hash', 'duplicate_count']

ENHANCED_FIELDS = ['glide_path', 'glide_filename', 'soh_primary_path',
                   'soh_all_paths', 'weighted_distance', 'confidence',
                   'algorithm_distances', 'duplicate_count', 'has_alpha_match']

# ---------------------------------------------------------------------
# MAP LOADING AND VALIDATION
# ---------------------------------------------------------------------

def detect_map_version(map_file):
    """Detect if map is enhanced (v2.0) or legacy."""
    with open(map_file, 'r', encoding='utf-8') as f:
        # Read just the header
        reader = csv.reader(f)
        header = next(reader)

    # Check for enhanced fields
    if 'weighted_distance' in header and 'confidence' in header:
        return 'enhanced'
    elif 'hamming_distance' in header:
        return 'legacy'
    else:
        return 'unknown'


def load_enhanced_texture_map(map_file):
    """
    Load enhanced texture map with confidence scores.

    Returns:
        tuple: (mapping_dict, stats_dict, metadata)
    """
    print(f"📂 Loading enhanced texture map: {map_file.name}")

    mapping = {}
    stats = {
        'total_mappings': 0,
        'distance_distribution': Counter(),
        'confidence_distribution': Counter(),
        'duplicate_distribution': Counter(),
        'alpha_matches': 0,
        'total_duplicate_paths': 0
    }

    with open(map_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Handle both enhanced and legacy formats
            if 'glide_path' in row:
                glide_key = row['glide_path']
            else:
                glide_key = row['glide_filename']

            # Parse enhanced fields or use defaults
            match_info = {
                'soh_primary_path': row.get('soh_primary_path', ''),
                'soh_all_paths': row.get('soh_all_paths', ''),
                'glide_filename': row.get('glide_filename', glide_key)
            }

            # Enhanced fields
            if 'weighted_distance' in row and row['weighted_distance']:
                match_info['weighted_distance'] = float(row['weighted_distance'])
            elif 'hamming_distance' in row and row['hamming_distance']:
                match_info['weighted_distance'] = float(row['hamming_distance'])
            else:
                match_info['weighted_distance'] = 0.0

            if 'confidence' in row and row['confidence']:
                match_info['confidence'] = float(row['confidence'])
            else:
                match_info['confidence'] = 1.0

            if 'algorithm_distances' in row:
                match_info['algorithm_distances'] = row['algorithm_distances']

            if 'duplicate_count' in row and row['duplicate_count']:
                match_info['duplicate_count'] = int(row['duplicate_count'])
            else:
                # Count from soh_all_paths
                paths = match_info['soh_all_paths'].split('|') if match_info['soh_all_paths'] else []
                match_info['duplicate_count'] = len(paths) if paths else 1

            if 'has_alpha_match' in row and row['has_alpha_match']:
                match_info['has_alpha_match'] = row['has_alpha_match'].lower() == 'true'
            else:
                match_info['has_alpha_match'] = False

            mapping[glide_key] = match_info

            # Update statistics
            stats['total_mappings'] += 1

            # Bucketize distances and confidences
            distance_bucket = int(match_info['weighted_distance'])
            stats['distance_distribution'][distance_bucket] += 1

            confidence_bucket = round(match_info['confidence'], 1)
            stats['confidence_distribution'][confidence_bucket] += 1

            dup_count = match_info['duplicate_count']
            stats['duplicate_distribution'][dup_count] += 1

            if match_info['duplicate_count'] > 1:
                stats['total_duplicate_paths'] += (match_info['duplicate_count'] - 1)

            if match_info.get('has_alpha_match'):
                stats['alpha_matches'] += 1

    # Try to load metadata from JSON file
    json_file = Path(str(map_file).replace('.csv', '.json'))
    metadata = {}
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                # Merge statistics
                if 'statistics' in data:
                    for key, value in data['statistics'].items():
                        if key not in stats:
                            stats[key] = value
        except Exception as e:
            print(f"  ⚠ Could not load JSON metadata: {e}")

    # Print loading summary
    print(f"  ✓ Loaded {stats['total_mappings']} texture mappings")

    if stats['distance_distribution']:
        print("  Match quality (weighted distance):")
        total_matches = sum(stats['distance_distribution'].values())
        for distance in sorted(stats['distance_distribution'].keys()):
            count = stats['distance_distribution'][distance]
            percentage = (count / total_matches) * 100
            print(f"    Distance ≤{distance}: {count} ({percentage:.1f}%)")

    if stats['confidence_distribution']:
        print("  Confidence distribution:")
        for confidence in sorted(stats['confidence_distribution'].keys(), reverse=True):
            count = stats['confidence_distribution'][confidence]
            percentage = (count / stats['total_mappings']) * 100
            print(f"    Confidence ≥{confidence:.1f}: {count} ({percentage:.1f}%)")

    if stats['duplicate_distribution']:
        unique_matches = stats['duplicate_distribution'].get(1, 0)
        duplicate_matches = stats['total_mappings'] - unique_matches

        print(f"  Duplicate handling:")
        print(f"    Unique matches: {unique_matches}")
        print(f"    Duplicate matches: {duplicate_matches}")
        print(f"    Additional copies needed: {stats['total_duplicate_paths']}")

        if duplicate_matches > 0:
            avg_destinations = stats['total_duplicate_paths'] / duplicate_matches + 1
            print(f"    Average destinations: {avg_destinations:.2f}")

    if stats['alpha_matches'] > 0:
        print(f"  Alpha channel matches: {stats['alpha_matches']}")

    return mapping, stats, metadata


def filter_mappings_by_quality(mapping, quality_level='medium'):
    """
    Filter mappings based on quality thresholds.

    Args:
        mapping: Dictionary of mappings
        quality_level: 'high', 'medium', or 'low'

    Returns:
        tuple: (filtered_mapping, filter_stats)
    """
    if quality_level not in QUALITY_THRESHOLDS:
        quality_level = 'medium'

    thresholds = QUALITY_THRESHOLDS[quality_level]
    max_distance = thresholds['max_distance']
    min_confidence = thresholds['min_confidence']

    filtered = {}
    filter_stats = Counter()

    for glide_key, match_info in mapping.items():
        distance = match_info.get('weighted_distance', 0)
        confidence = match_info.get('confidence', 1.0)

        if distance <= max_distance and confidence >= min_confidence:
            filtered[glide_key] = match_info
            filter_stats['accepted'] += 1
        else:
            filter_stats['rejected'] += 1

            if distance > max_distance:
                filter_stats['rejected_distance'] += 1
            if confidence < min_confidence:
                filter_stats['rejected_confidence'] += 1

    return filtered, dict(filter_stats)


# ---------------------------------------------------------------------
# TEXTURE PACK PROCESSING
# ---------------------------------------------------------------------

def scan_texture_pack_with_metadata(directory):
    """
    Enhanced scan with file metadata collection.

    Returns:
        list of tuples: (relative_path, full_path, filename, file_size, modified_time)
    """
    texture_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)

                try:
                    file_stats = os.stat(full_path)
                    texture_files.append((
                        rel_path,
                        full_path,
                        file,
                        file_stats.st_size,
                        file_stats.st_mtime
                    ))
                except OSError:
                    # Skip unreadable files
                    continue

    return texture_files


def create_backup_copy(source_path, dest_dir, suffix='_original'):
    """
    Create a backup copy of original file for visual comparison.

    Args:
        source_path: Path to source file
        dest_dir: Destination directory for backup
        suffix: Suffix to add to filename

    Returns:
        str or None: Path to backup file, or None if failed
    """
    try:
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        backup_name = f"{name}{suffix}{ext}"
        backup_path = os.path.join(dest_dir, backup_name)

        # Create directory if needed
        os.makedirs(dest_dir, exist_ok=True)

        shutil.copy2(source_path, backup_path)
        return backup_path
    except Exception:
        return None


# ---------------------------------------------------------------------
# MAIN CONVERSION WORKFLOW
# ---------------------------------------------------------------------

def collect_conversion_configuration():
    """Get user configuration for conversion."""
    print("\n" + "=" * 60)
    print("ENHANCED TEXTURE PACK CONVERTER")
    print("=" * 60)

    # Find map files
    map_files = list(Path('.').glob('*texture_map_*.csv'))

    if not map_files:
        print("❌ No texture map files found")
        print("   Run enhanced map.py first to create a texture map")
        return None

    # Sort by modification time
    map_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Display available maps
    print("\n📁 Available texture maps (newest first):")
    display_count = min(5, len(map_files))

    for i, map_file in enumerate(map_files[:display_count], 1):
        size_kb = map_file.stat().st_size / 1024
        version = detect_map_version(map_file)
        print(f"  {i}. {map_file.name} ({size_kb:.0f} KB) [{version}]")

    if len(map_files) > display_count:
        print(f"  ... and {len(map_files) - display_count} more")

    # Select map
    print("\nOptions:")
    print("  # - Select by number (1-5)")
    print("  filename - Enter full filename")
    choice = input(f"\nSelect map (1-{display_count}) or enter filename: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(map_files):
        selected_map = map_files[int(choice) - 1]
    else:
        selected_map = Path(choice)
        if not selected_map.exists():
            print(f"❌ Map file not found: {choice}")
            return None

    # Get texture pack directory
    print("\n📦 Texture pack to convert:")
    texture_pack_dir = input("Enter texture pack directory: ").strip()

    if not os.path.exists(texture_pack_dir):
        print(f"❌ Directory not found: {texture_pack_dir}")
        return None

    # Quality filtering
    print("\n⚙️  Quality filtering:")
    print("  1. High quality (distance ≤5, confidence ≥0.7)")
    print("  2. Medium quality (distance ≤10, confidence ≥0.4) [recommended]")
    print("  3. Low quality (distance ≤20, confidence ≥0.0)")
    print("  4. No filtering (use all matches)")

    quality_choice = input("\nSelect quality level (1-4): ").strip()
    quality_levels = { '1': 'high', '2': 'medium', '3': 'low', '4': 'none' }
    quality_level = quality_levels.get(quality_choice, 'medium')

    # Additional options
    print("\n🔧 Additional options:")
    create_backups = input("Create backup copies of originals for comparison? (y/N): ").lower().strip() == 'y'
    overwrite_existing = input("Overwrite existing converted files? (y/N): ").lower().strip() == 'y'

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"converted_{timestamp}"
    output_dir = input(f"\n📁 Output directory [{default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output

    return {
        'map_file': selected_map,
        'texture_pack_dir': texture_pack_dir,
        'quality_level': quality_level,
        'create_backups': create_backups,
        'overwrite_existing': overwrite_existing,
        'output_dir': output_dir
    }


def main():
    """Enhanced conversion workflow."""
    config = collect_conversion_configuration()
    if not config:
        return

    print("\n" + "=" * 60)
    print("LOADING AND VALIDATING")
    print("=" * 60)

    # Load texture map
    mapping, map_stats, metadata = load_enhanced_texture_map(config['map_file'])

    # Apply quality filtering if requested
    if config['quality_level'] != 'none':
        print(f"\n🎯 Applying {config['quality_level']} quality filter...")
        mapping, filter_stats = filter_mappings_by_quality(mapping, config['quality_level'])

        print(f"  Accepted: {filter_stats.get('accepted', 0)}")
        print(f"  Rejected: {filter_stats.get('rejected', 0)}")

        if filter_stats.get('rejected_distance', 0) > 0:
            print(f"    Due to distance: {filter_stats['rejected_distance']}")
        if filter_stats.get('rejected_confidence', 0) > 0:
            print(f"    Due to confidence: {filter_stats['rejected_confidence']}")

    # Scan texture pack
    print(f"\n🔍 Scanning texture pack: {config['texture_pack_dir']}")
    texture_files = scan_texture_pack_with_metadata(config['texture_pack_dir'])

    if not texture_files:
        print("❌ No PNG files found in texture pack")
        return

    print(f"  Found {len(texture_files)} PNG files")

    # Set up output directories
    output_config = {
        'main_output': config['output_dir'],
        'backup_output': os.path.join(config['output_dir'], 'originals') if config['create_backups'] else None,
        'csv_report': f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'json_report': f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    }

    # Create output directories
    os.makedirs(output_config['main_output'], exist_ok=True)
    if output_config['backup_output']:
        os.makedirs(output_config['backup_output'], exist_ok=True)

    print(f"\n📁 Output configuration:")
    print(f"  Main output: {output_config['main_output']}")
    if output_config['backup_output']:
        print(f"  Backup copies: {output_config['backup_output']}")
    print(f"  CSV report: {output_config['csv_report']}")
    print(f"  JSON report: {output_config['json_report']}")

    if not config['overwrite_existing']:
        print(f"  ⚠ Will skip existing files")

    print("\n" + "=" * 60)
    print("CONVERTING TEXTURES")
    print("=" * 60)

    # Initialize tracking
    conversion_stats = {
        'total_textures': len(texture_files),
        'converted': 0,
        'missing': 0,
        'errors': 0,
        'skipped_existing': 0,
        'total_copies_made': 0,
        'duplicate_copies': 0,
        'distance_distribution': Counter(),
        'confidence_distribution': Counter(),
        'duplicate_distribution': Counter(),
        'alpha_matches': 0,
        'backup_copies': 0,
        'converted_details': [],
        'missing_details': [],
        'error_details': []
    }

    # Process files
    for rel_path, full_path, filename, file_size, mod_time in tqdm(
        texture_files, desc="Converting", unit="files"
    ):
        try:
            # Try different keys for lookup
            lookup_keys = [filename, rel_path]
            match_info = None

            for key in lookup_keys:
                if key in mapping:
                    match_info = mapping[key]
                    break

            if match_info:
                # Get all destination paths
                if match_info['soh_all_paths']:
                    all_soh_paths = match_info['soh_all_paths'].split('|')
                else:
                    all_soh_paths = [match_info['soh_primary_path']]

                duplicate_count = len(all_soh_paths)
                weighted_distance = match_info.get('weighted_distance', 0)
                confidence = match_info.get('confidence', 1.0)
                has_alpha = match_info.get('has_alpha_match', False)

                # Create backup copy if requested
                backup_path = None
                if config['create_backups'] and output_config['backup_output']:
                    backup_path = create_backup_copy(
                        full_path, output_config['backup_output']
                    )
                    if backup_path:
                        conversion_stats['backup_copies'] += 1

                # Copy to all destinations
                copied_paths = []
                skipped_paths = []

                for soh_path in all_soh_paths:
                    dest_path = os.path.join(output_config['main_output'], soh_path)
                    dest_dir = os.path.dirname(dest_path)

                    # Create destination directory
                    os.makedirs(dest_dir, exist_ok=True)

                    # Check if file already exists
                    if os.path.exists(dest_path) and not config['overwrite_existing']:
                        skipped_paths.append(dest_path)
                        continue

                    # Copy file
                    shutil.copy2(full_path, dest_path)
                    copied_paths.append(dest_path)

                # Update statistics
                conversion_stats['converted'] += 1
                conversion_stats['total_copies_made'] += len(copied_paths)
                conversion_stats['duplicate_copies'] += (duplicate_count - 1)
                conversion_stats['skipped_existing'] += len(skipped_paths)

                # Update distributions
                distance_bucket = int(weighted_distance)
                conversion_stats['distance_distribution'][distance_bucket] += 1

                confidence_bucket = round(confidence, 1)
                conversion_stats['confidence_distribution'][confidence_bucket] += 1

                conversion_stats['duplicate_distribution'][duplicate_count] += 1

                if has_alpha:
                    conversion_stats['alpha_matches'] += 1

                # Record conversion details
                conversion_details = {
                    'original_path': rel_path,
                    'filename': filename,
                    'soh_primary_path': all_soh_paths[0],
                    'soh_all_paths': '|'.join(all_soh_paths),
                    'weighted_distance': weighted_distance,
                    'confidence': confidence,
                    'algorithm_distances': match_info.get('algorithm_distances', ''),
                    'duplicate_count': duplicate_count,
                    'additional_copies': duplicate_count - 1,
                    'has_alpha_match': has_alpha,
                    'source_file': full_path,
                    'copied_destinations': '|'.join(copied_paths),
                    'skipped_destinations': '|'.join(skipped_paths),
                    'backup_copy': backup_path or '',
                    'status': 'CONVERTED',
                    'error': '',
                    'notes': f"Skipped {len(skipped_paths)} existing files" if skipped_paths else ''
                }
                conversion_stats['converted_details'].append(conversion_details)

            else:
                # No mapping found
                conversion_stats['missing'] += 1

                missing_details = {
                    'original_path': rel_path,
                    'filename': filename,
                    'soh_primary_path': '',
                    'soh_all_paths': '',
                    'weighted_distance': '',
                    'confidence': '',
                    'algorithm_distances': '',
                    'duplicate_count': '',
                    'additional_copies': '',
                    'has_alpha_match': '',
                    'source_file': full_path,
                    'copied_destinations': '',
                    'skipped_destinations': '',
                    'backup_copy': '',
                    'status': 'MISSING',
                    'error': '',
                    'notes': 'No match found in texture map'
                }
                conversion_stats['missing_details'].append(missing_details)

        except Exception as e:
            # Handle conversion errors
            conversion_stats['errors'] += 1

            error_details = {
                'original_path': rel_path,
                'filename': filename,
                'soh_primary_path': '',
                'soh_all_paths': '',
                'weighted_distance': '',
                'confidence': '',
                'algorithm_distances': '',
                'duplicate_count': '',
                'additional_copies': '',
                'has_alpha_match': '',
                'source_file': full_path,
                'copied_destinations': '',
                'skipped_destinations': '',
                'backup_copy': '',
                'status': 'ERROR',
                'error': str(e),
                'notes': str(e)
            }
            conversion_stats['error_details'].append(error_details)

    # Generate reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    # Save CSV report
    csv_fields = [
        'original_path', 'filename', 'soh_primary_path', 'soh_all_paths',
        'weighted_distance', 'confidence', 'algorithm_distances',
        'duplicate_count', 'additional_copies', 'has_alpha_match',
        'source_file', 'copied_destinations', 'skipped_destinations',
        'backup_copy', 'status', 'error', 'notes'
    ]

    print(f"💾 Saving CSV report: {output_config['csv_report']}")
    with open(output_config['csv_report'], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        all_records = (
            conversion_stats['converted_details'] +
            conversion_stats['missing_details'] +
            conversion_stats['error_details']
        )

        for record in all_records:
            writer.writerow(record)

    # Save JSON report
    print(f"💾 Saving JSON report: {output_config['json_report']}")
    report_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '2.0-enhanced',
            'map_file': config['map_file'].name,
            'texture_pack': config['texture_pack_dir'],
            'output_directory': output_config['main_output'],
            'quality_level': config['quality_level'],
            'create_backups': config['create_backups'],
            'overwrite_existing': config['overwrite_existing']
        },
        'map_statistics': {
            'total_mappings': map_stats['total_mappings'],
            'distance_distribution': dict(map_stats['distance_distribution']),
            'confidence_distribution': dict(map_stats['confidence_distribution']),
            'duplicate_distribution': dict(map_stats['duplicate_distribution']),
            'total_duplicate_paths': map_stats['total_duplicate_paths'],
            'alpha_matches': map_stats['alpha_matches']
        },
        'conversion_statistics': {
            'total_textures': conversion_stats['total_textures'],
            'converted': conversion_stats['converted'],
            'missing': conversion_stats['missing'],
            'errors': conversion_stats['errors'],
            'skipped_existing': conversion_stats['skipped_existing'],
            'total_copies_made': conversion_stats['total_copies_made'],
            'duplicate_copies': conversion_stats['duplicate_copies'],
            'backup_copies': conversion_stats['backup_copies'],
            'distance_distribution': dict(conversion_stats['distance_distribution']),
            'confidence_distribution': dict(conversion_stats['confidence_distribution']),
            'duplicate_distribution': dict(conversion_stats['duplicate_distribution']),
            'alpha_matches': conversion_stats['alpha_matches'],
            'conversion_rate': (
                conversion_stats['converted'] / conversion_stats['total_textures'] * 100
                if conversion_stats['total_textures'] > 0 else 0
            )
        },
        'quality_analysis': {
            'average_distance': (
                sum(dist * count for dist, count in conversion_stats['distance_distribution'].items()) /
                conversion_stats['converted'] if conversion_stats['converted'] > 0 else 0
            ),
            'average_confidence': (
                sum(conf * count for conf, count in conversion_stats['confidence_distribution'].items()) /
                conversion_stats['converted'] if conversion_stats['converted'] > 0 else 0
            ),
            'average_duplicates': (
                conversion_stats['total_copies_made'] / conversion_stats['converted']
                if conversion_stats['converted'] > 0 else 0
            )
        }
    }

    with open(output_config['json_report'], 'w', encoding='utf-8') as jsonfile:
        json.dump(report_data, jsonfile, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"\n📊 Conversion Summary:")
    print(f"   Texture pack files: {conversion_stats['total_textures']}")
    print(f"   Successfully converted: {conversion_stats['converted']}")
    print(f"   No mapping found: {conversion_stats['missing']}")
    print(f"   Errors: {conversion_stats['errors']}")

    if conversion_stats['skipped_existing'] > 0:
        print(f"   Skipped existing files: {conversion_stats['skipped_existing']}")

    conversion_rate = (
        conversion_stats['converted'] / conversion_stats['total_textures'] * 100
        if conversion_stats['total_textures'] > 0 else 0
    )
    print(f"   Conversion rate: {conversion_rate:.1f}%")

    if conversion_stats['converted'] > 0:
        print(f"\n🎯 Match Quality:")

        avg_distance = report_data['quality_analysis']['average_distance']
        avg_confidence = report_data['quality_analysis']['average_confidence']
        avg_duplicates = report_data['quality_analysis']['average_duplicates']

        print(f"   Average distance: {avg_distance:.2f}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Average destinations: {avg_duplicates:.2f}")

        print(f"\n📈 Distribution:")

        if conversion_stats['distance_distribution']:
            print(f"   Distance distribution:")
            for dist in sorted(conversion_stats['distance_distribution'].keys()):
                count = conversion_stats['distance_distribution'][dist]
                percent = count / conversion_stats['converted'] * 100
                print(f"     ≤{dist}: {count} ({percent:.1f}%)")

        if conversion_stats['confidence_distribution']:
            print(f"   Confidence distribution:")
            for conf in sorted(conversion_stats['confidence_distribution'].keys(), reverse=True):
                count = conversion_stats['confidence_distribution'][conf]
                percent = count / conversion_stats['converted'] * 100
                print(f"     ≥{conf:.1f}: {count} ({percent:.1f}%)")

        if conversion_stats['duplicate_copies'] > 0:
            print(f"\n🔄 Duplicate handling:")
            print(f"   Additional copies: {conversion_stats['duplicate_copies']}")
            print(f"   Total file operations: {conversion_stats['total_copies_made']}")

        if conversion_stats['alpha_matches'] > 0:
            print(f"\n🎭 Alpha channel matches: {conversion_stats['alpha_matches']}")

        if conversion_stats['backup_copies'] > 0:
            print(f"\n💾 Backup copies created: {conversion_stats['backup_copies']}")

    print(f"\n📁 Output files:")
    print(f"   Converted textures: {output_config['main_output']}")
    if output_config['backup_output']:
        print(f"   Backup copies: {output_config['backup_output']}")
    print(f"   CSV report: {output_config['csv_report']}")
    print(f"   JSON report: {output_config['json_report']}")

    print(f"\n✅ IMPORTANT:")
    print(f"   Original texture pack files are UNMODIFIED")
    print(f"   Converted textures are ready for use")

    if conversion_stats['missing'] > 0:
        print(f"\n⚠️  Note: {conversion_stats['missing']} textures had no mapping")
        print(f"   These may be unique to this texture pack")
        print(f"   Check the CSV report for details")

    print("\n" + "=" * 60)


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
