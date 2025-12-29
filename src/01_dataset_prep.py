import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = "data/raw"
TRAIN_DB = "data/train_db"
TEST_SET = "data/test_set"

def scan_dataset():
    """Scan folder structure: person_name/image.jpg"""
    records = []
    for person in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        images = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            records.append({
                "person": person,
                "img_path": os.path.join(person_dir, img),
                "img_name": img
            })
    df = pd.DataFrame(records)
    print(f"Total images: {len(df)}")
    print(f"Persons with 1 image: {(df.groupby('person').size() == 1).sum()}")
    print(f"Persons with 2+ images: {(df.groupby('person').size() >= 2).sum()}")
    return df

def split_dataset(df):
    """80/20 split, respecting single-image persons"""
    single_img_persons = df[df.groupby('person')['person'].transform('size') == 1]
    multi_img_persons = df[df.groupby('person')['person'].transform('size') >= 2]
    
    # Single image persons → all to train (for db coverage)
    train_single = single_img_persons
    
    # Multi-image: 80/20 split
    train_multi_list, test_list = [], []
    for person, group in multi_img_persons.groupby('person'):
        if len(group) >= 2:
            train_g, test_g = train_test_split(group, test_size=0.2, random_state=42)
            train_multi_list.append(train_g)
            test_list.append(test_g)
    
    train_df = pd.concat([train_single] + train_multi_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    print(f"Train: {len(train_df)} images, Test: {len(test_df)} images")
    return train_df, test_df

def copy_files(df, target_base, prefix=""):
    """Copy images maintaining person folders"""
    os.makedirs(target_base, exist_ok=True)
    for _, row in df.iterrows():
        person_dir = os.path.join(target_base, row['person'])
        os.makedirs(person_dir, exist_ok=True)
        target_path = os.path.join(person_dir, f"{prefix}{row['img_name']}")
        shutil.copy2(row['img_path'], target_path)

def augment_train(train_df):
    """Augment training images (rotation, flip, brightness)"""
    aug_dir = TRAIN_DB
    aug_count = 0
    
    for _, row in train_df.iterrows():
        img_path = row['img_path']
        person = row['person']
        img_name = row['img_name']
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        person_dir = os.path.join(aug_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Original
        shutil.copy2(img_path, os.path.join(person_dir, img_name))
        
        # Flip
        flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(person_dir, f"flip_{img_name}"), flip)
        aug_count += 1
        
        # Rotate +5°
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
        rot = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(os.path.join(person_dir, f"rot5_{img_name}"), rot)
        aug_count += 1
        
        # Brightness
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        cv2.imwrite(os.path.join(person_dir, f"bright_{img_name}"), bright)
        aug_count += 1
    
    print(f"Created {aug_count} augmented images")

if __name__ == "__main__":
    df = scan_dataset()
    train_df, test_df = split_dataset(df)
    
    # Copy test set
    copy_files(test_df, TEST_SET)
    
    # Copy + augment train set (creates DeepFace-ready db)
    copy_files(train_df, TRAIN_DB)
    augment_train(train_df)
    
    # Save metadata
    train_df.to_csv("data/train_metadata.csv", index=False)
    test_df.to_csv("data/test_metadata.csv", index=False)
    print("✅ Dataset preparation complete!")
