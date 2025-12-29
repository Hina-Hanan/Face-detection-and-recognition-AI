import os
import shutil
import pandas as pd
import random
from pathlib import Path

DATA_DIR = "data/raw"
TRAIN_DB = "data/train_db"
TEST_SET = "data/test_set"
CACHE_FILE = "data/train_embeddings_cache.pkl"

def scan_and_filter_dataset():
    """Scan dataset and filter to persons with 2+ images"""
    print("Scanning dataset...")
    records = []
    
    for person in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        
        images = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) >= 2:  # Only persons with 2+ images
            for img in images:
                records.append({
                    "person": person,
                    "img_path": os.path.join(person_dir, img),
                    "img_name": img
                })
    
    df = pd.DataFrame(records)
    print(f"\nDataset Statistics:")
    print(f"  Total persons with 2+ images: {df['person'].nunique()}")
    print(f"  Total images: {len(df)}")
    
    return df

def select_25_percent(df):
    """Randomly select 25% of persons"""
    unique_persons = df['person'].unique().tolist()
    total_persons = len(unique_persons)
    target_count = max(1, int(total_persons * 0.25))  # At least 1 person
    
    print(f"\nSelecting 25% of persons...")
    print(f"  Total persons: {total_persons}")
    print(f"  Target: {target_count} persons (25%)")
    
    # Set seed for reproducibility
    random.seed(42)
    selected_persons = random.sample(unique_persons, target_count)
    
    filtered_df = df[df['person'].isin(selected_persons)].copy()
    
    print(f"  Selected {len(selected_persons)} persons")
    print(f"  Selected images: {len(filtered_df)}")
    
    return filtered_df, selected_persons

def delete_unselected_persons(selected_persons):
    """Delete person folders that are not in selected list"""
    print(f"\nDeleting unselected persons from {DATA_DIR}...")
    
    deleted_count = 0
    for person in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person)
        if os.path.isdir(person_dir) and person not in selected_persons:
            shutil.rmtree(person_dir)
            deleted_count += 1
    
    print(f"  Deleted {deleted_count} person folders")

def cleanup_processed_data():
    """Delete train_db, test_set, cache files, and metadata"""
    print(f"\nCleaning up processed data...")
    
    cleanup_items = [
        (TRAIN_DB, "Training database"),
        (TEST_SET, "Test set"),
        (CACHE_FILE, "Embeddings cache"),
        ("data/train_metadata.csv", "Train metadata"),
        ("data/test_metadata.csv", "Test metadata"),
    ]
    
    for path, name in cleanup_items:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    # More robust deletion for Windows
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, 0o777)
                        func(path)
                    shutil.rmtree(path, onerror=handle_remove_readonly)
                else:
                    os.remove(path)
                print(f"  Deleted {name}")
            except Exception as e:
                print(f"  Warning: Could not delete {name}: {e}")
        else:
            print(f"  {name} not found (already clean)")

def main():
    print("=" * 60)
    print("Dataset Filtering Script")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Keep only persons with 2+ images")
    print("  2. Randomly select 25% of those persons")
    print("  3. Delete all other persons from data/raw")
    print("  4. Clean up processed data (train_db, test_set, cache)")
    print("=" * 60)
    print("\nWARNING: This will DELETE data. Starting in 2 seconds...")
    import time
    time.sleep(2)
    
    # Step 1: Scan and filter to 2+ images
    df = scan_and_filter_dataset()
    
    if len(df) == 0:
        print("No persons with 2+ images found!")
        return
    
    # Step 2: Select 25%
    filtered_df, selected_persons = select_25_percent(df)
    
    # Step 3: Delete unselected persons
    delete_unselected_persons(selected_persons)
    
    # Step 4: Cleanup processed data
    cleanup_processed_data()
    
    print("\n" + "=" * 60)
    print("Dataset filtering complete!")
    print("=" * 60)
    print(f"\nRemaining dataset:")
    print(f"  Persons: {len(selected_persons)}")
    print(f"  Images: {len(filtered_df)}")
    print(f"\nNext steps:")
    print(f"  1. Run: python src/01_dataset_prep.py")
    print(f"  2. Then: python src/02_evaluate.py")

if __name__ == "__main__":
    main()

