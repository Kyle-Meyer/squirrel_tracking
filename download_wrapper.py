#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd
from src.downloader import ImageDownloader
from src.squirrel_trainer import SquirrelYOLOTrainer

class DirectTrainingDownloader:
    def __init__(self):
        # Initialize the trainer to set up project structure
        self.trainer = SquirrelYOLOTrainer()
        
        # Set up paths for all splits
        self.base_path = Path("yolo_squirrel_project/dataset")
        self.splits = {
            'train': {
                'images': self.base_path / "train" / "images",
                'labels': self.base_path / "train" / "labels"
            },
            'val': {
                'images': self.base_path / "val" / "images", 
                'labels': self.base_path / "val" / "labels"
            },
            'test': {
                'images': self.base_path / "test" / "images",
                'labels': self.base_path / "test" / "labels"
            }
        }
        
    def setup_dirs(self, split='train'):
        # Project structure is already created by SquirrelYOLOTrainer
        # Just ensure our specific directories exist
        images_dir = self.splits[split]['images']
        labels_dir = self.splits[split]['labels']
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure classes.txt exists
        classes_file = labels_dir / "classes.txt"
        if not classes_file.exists():
            classes_file.write_text("squirrel\n")
            print(f"✅ Created classes.txt in {labels_dir}")
        
        print(f"✅ {split.upper()} directories ready:")
        print(f"   📁 Images: {images_dir}")
        print(f"   📁 Labels: {labels_dir}")
        
        return images_dir, labels_dir
        
    def get_existing_image_ids(self, split='train'):
        """Get existing image IDs only from the specified split folder"""
        existing_ids = set()
        
        images_dir = self.splits[split]['images']
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                if img_file.name.startswith("squirrel_"):
                    img_id = img_file.stem.replace("squirrel_", "")
                    existing_ids.add(img_id)
                    
        print(f"📊 Found {len(existing_ids)} existing image IDs in {split} set")
        return existing_ids
        
    def download_to_split(self, csv_file, split='train', max_images=0, skip_existing=True):
        
        print(f"🔍 DOWNLOADING TO {split.upper()} FOLDER FROM: {csv_file}")
        print("="*60)
        
        if split not in self.splits:
            print(f"❌ Invalid split '{split}'. Must be one of: {list(self.splits.keys())}")
            return False
        
        if not Path(csv_file).exists():
            print(f"❌ CSV file not found: {csv_file}")
            return False
            
        # Setup directories for the specified split
        images_dir, labels_dir = self.setup_dirs(split)
        
        # Load CSV and check structure
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 CSV loaded: {len(df)} total records")
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
            
        # Check required columns
        if 'image_url' not in df.columns:
            print("❌ CSV must have 'image_url' column")
            print(f"📊 Available columns: {list(df.columns)}")
            return False
            
        if 'id' not in df.columns:
            print("❌ CSV must have 'id' column")
            print(f"📊 Available columns: {list(df.columns)}")
            return False
            
        # First filter for images with URLs
        df_with_urls = df.dropna(subset=['image_url'])
        
        # Show dataset info
        print(f"📊 Dataset info:")
        print(f"   - Total observations: {len(df)}")
        print(f"   - With image URLs: {len(df_with_urls)}")
        
        if 'common_name' in df.columns:
            species_counts = df['common_name'].value_counts().head(5)
            print(f"   - Top species: {species_counts.to_dict()}")
            
        if 'quality_grade' in df.columns:
            quality_counts = df['quality_grade'].value_counts()
            print(f"   - Quality grades: {quality_counts.to_dict()}")
            
        # Filter out images we already have (if requested)
        if skip_existing:
            existing_images = self.get_existing_image_ids(split)
            initial_count = len(df_with_urls)
            
            # Debug: Show some info about what we're filtering
            print(f"📊 Checking {initial_count} records against {len(existing_images)} existing images in {split}")
            
            df_with_urls = df_with_urls[~df_with_urls['id'].astype(str).isin(existing_images)]
            filtered_count = initial_count - len(df_with_urls)
            print(f"📊 Filtered out {filtered_count} existing images")
            
            # Debug: If we have very few remaining, show what they are
            if len(df_with_urls) < 10:
                print(f"📊 Remaining records:")
                for _, row in df_with_urls.iterrows():
                    has_url = pd.notna(row.get('image_url', None)) and row.get('image_url', '') != ''
                    print(f"   ID: {row['id']}, Has URL: {has_url}")
                    
        else:
            print(f"📊 Skipping existing image check (--force mode)")
            
        # Limit to max_images (but allow downloading all if max_images is 0 or large)
        if max_images > 0 and len(df_with_urls) > max_images:
            # Prioritize research grade if available
            if 'quality_grade' in df_with_urls.columns:
                research_grade = df_with_urls[df_with_urls['quality_grade'] == 'research']
                if len(research_grade) >= max_images:
                    df_with_urls = research_grade.sample(n=max_images, random_state=42)
                    print(f"📊 Selected {max_images} research-grade images")
                else:
                    df_with_urls = df_with_urls.sample(n=max_images, random_state=42)
                    print(f"📊 Randomly selected {max_images} images")
            else:
                df_with_urls = df_with_urls.sample(n=max_images, random_state=42)
                print(f"📊 Randomly selected {max_images} images")
        else:
            print(f"📊 Will attempt to download all {len(df_with_urls)} images")
            
        print(f"📊 Final selection: {len(df_with_urls)} records ready for download")
        
        if len(df_with_urls) == 0:
            print("❌ No images to download!")
            return False
            
        # Use the existing downloader but point to the split folder
        downloader = ImageDownloader(
            aCsvFile=csv_file, 
            aOutputFolder=str(images_dir)
        )
        
        # Create temporary CSV with only the images we want
        temp_csv = Path("temp_download.csv")
        df_with_urls.to_csv(temp_csv, index=False)
        
        # Update downloader to use filtered CSV
        downloader.mCsvFile = str(temp_csv)
        
        print(f"🚀 Starting download of {len(df_with_urls)} images to {split} folder...")
        downloader.downloadAllImages()
        
        # Cleanup temp file
        temp_csv.unlink(missing_ok=True)
        
        # Create empty annotation files for new images
        self.create_empty_labels(split)
        
        # Create dataset.yaml file
        self.trainer.createDataSetYaml()
        
        # Report results
        downloaded_images = list(images_dir.glob("*.jpg"))
        print(f"\n✅ Download complete!")
        print(f"📊 Total images in {split} folder: {len(downloaded_images)}")
        print(f"📁 Full project structure created at: yolo_squirrel_project/")
        
        return len(downloaded_images) > 0
        
    def create_empty_labels(self, split='train'):
        images_dir = self.splits[split]['images']
        labels_dir = self.splits[split]['labels']
        
        if not images_dir.exists():
            return 0
            
        created_count = 0
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                label_file.touch()  # Create empty file
                created_count += 1
                
        if created_count > 0:
            print(f"✅ Created {created_count} empty annotation files in {split}")
        return created_count
        
    def show_training_stats(self):
        print(f"\n📊 DATASET STATISTICS")
        print("="*50)
        
        for split in ['train', 'val', 'test']:
            images_dir = self.splits[split]['images']
            labels_dir = self.splits[split]['labels']
            
            if not images_dir.exists():
                print(f"📈 {split.upper()}: No directory found")
                continue
                
            images = list(images_dir.glob("*.jpg"))
            annotated = 0
            
            if labels_dir.exists():
                for img in images:
                    label_file = labels_dir / f"{img.stem}.txt"
                    if label_file.exists() and label_file.stat().st_size > 0:
                        annotated += 1
                        
            coverage = annotated/max(1,len(images))*100
            print(f"📈 {split.upper()}: {len(images)} images, {annotated} annotated ({coverage:.1f}%)")
        
        # Overall annotation guidance
        train_images = list(self.splits['train']['images'].glob("*.jpg")) if self.splits['train']['images'].exists() else []
        if len(train_images) > 0:
            train_labels_dir = self.splits['train']['labels']
            train_annotated = 0
            if train_labels_dir.exists():
                for img in train_images:
                    label_file = train_labels_dir / f"{img.stem}.txt"
                    if label_file.exists() and label_file.stat().st_size > 0:
                        train_annotated += 1
            
            if train_annotated == 0:
                print(f"\n💡 READY FOR ANNOTATION:")
                print(f"📁 Images to annotate: {self.splits['train']['images']}")
                print(f"📁 Save annotations to: {self.splits['train']['labels']}")
                print(f"🏷️  Use LabelImg or similar tool")
            else:
                print(f"\n🎯 ANNOTATION PROGRESS:")
                print(f"✅ Annotated: {train_annotated}")
                print(f"⏳ Remaining: {len(train_images) - train_annotated}")

def main():
    print("🐿️  DIRECT TRAINING FOLDER DOWNLOADER")
    print("="*50)
    
    downloader = DirectTrainingDownloader()
    
    if len(sys.argv) < 2:
        print("📋 USAGE:")
        print("  python download_wrapper.py download_all <csv_file> [--force]")
        print("  python download_wrapper.py download <csv_file> <max_images> [split]")
        print("  python download_wrapper.py stats")
        print("  python download_wrapper.py prepare [split]")
        print("")
        print("📋 EXAMPLES:")
        print("  python download_wrapper.py download_all observations603652.csv")
        print("  python download_wrapper.py download_all observations603652.csv --force")
        print("  python download_wrapper.py download observations603652.csv 100")
        print("  python download_wrapper.py download observations603652.csv 50 val")
        print("  python download_wrapper.py download observations603652.csv 25 test")
        print("")
        print("💡 --force flag will re-download existing images")
        print("💡 split can be: train (default), val, or test")
        print("💡 Images download to yolo_squirrel_project/dataset/<split>/images/")
        print("💡 Empty label files created in yolo_squirrel_project/dataset/<split>/labels/")
        print("💡 Full project structure will be created automatically!")
        print("")
        downloader.show_training_stats()
        return
        
    command = sys.argv[1].lower()
    
    if command == "download_all":
        if len(sys.argv) < 3:
            print("❌ Please specify CSV file!")
            print("Usage: python download_wrapper.py download_all observations603652.csv")
            print("       python download_wrapper.py download_all observations603652.csv --force")
            return
            
        csv_file = sys.argv[2]
        skip_existing = True
        
        # Check for --force flag
        if len(sys.argv) > 3 and sys.argv[3] == "--force":
            skip_existing = False
            print(f"🚀 FORCE DOWNLOADING ALL IMAGES (including existing ones)")
        else:
            print(f"🚀 DOWNLOADING ALL IMAGES TO TRAINING FOLDER")
            
        print(f"📁 This will create the full yolo_squirrel_project structure")
        
        success = downloader.download_to_split(csv_file, split='train', max_images=0, skip_existing=skip_existing)
        if success:
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Use LabelImg to annotate images")
            print(f"2. Images: {downloader.splits['train']['images']}")
            print(f"3. Save labels: {downloader.splits['train']['labels']}")
            print(f"4. Train model: python main.py")
            
    elif command == "download":
        if len(sys.argv) < 4:
            print("❌ Please specify CSV file and max images!")
            print("Usage: python download_wrapper.py download observations603652.csv 100 [split]")
            return
            
        csv_file = sys.argv[2]
        max_images = int(sys.argv[3])
        split = sys.argv[4] if len(sys.argv) > 4 else 'train'
        
        if split not in ['train', 'val', 'test']:
            print(f"❌ Invalid split '{split}'. Must be: train, val, or test")
            return
            
        print(f"🚀 DOWNLOADING UP TO {max_images} IMAGES TO {split.upper()} FOLDER")
        print(f"📁 This will create the full yolo_squirrel_project structure")
        
        success = downloader.download_to_split(csv_file, split=split, max_images=max_images)
        if success:
            print(f"\n📋 NEXT STEPS:")
            if split == 'train':
                print(f"1. Use LabelImg to annotate images")
                print(f"2. Images: {downloader.splits[split]['images']}")
                print(f"3. Save labels: {downloader.splits[split]['labels']}")
                print(f"4. Train model: python main.py")
            else:
                print(f"1. Use LabelImg to annotate {split} images")
                print(f"2. Images: {downloader.splits[split]['images']}")
                print(f"3. Save labels: {downloader.splits[split]['labels']}")
                print(f"4. These will be used for {split} during training")
            
    elif command == "prepare":
        split = sys.argv[2] if len(sys.argv) > 2 else 'train'
        if split not in ['train', 'val', 'test']:
            print(f"❌ Invalid split '{split}'. Must be: train, val, or test")
            return
            
        count = downloader.create_empty_labels(split)
        print(f"✅ Ready for annotation in {split} set!")
            
    elif command == "download":
        if len(sys.argv) < 4:
            print("❌ Please specify CSV file and max images!")
            print("Usage: python download_wrapper.py download observations603652.csv 100")
            return
            
        csv_file = sys.argv[2]
        max_images = int(sys.argv[3])
        print(f"🚀 DOWNLOADING UP TO {max_images} IMAGES TO TRAINING FOLDER")
        print(f"📁 This will create the full yolo_squirrel_project structure")
        
        success = downloader.download_to_training(csv_file, max_images)
        if success:
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Use LabelImg to annotate images")
            print(f"2. Images: {downloader.train_images_dir}")
            print(f"3. Save labels: {downloader.train_labels_dir}")
            print(f"4. Train model: python main.py")
            
    elif command == "prepare":
        count = downloader.create_empty_labels()
        print(f"✅ Ready for annotation!")
        
    elif command == "stats":
        downloader.show_training_stats()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: download_all, download, stats, prepare")

if __name__ == "__main__":
    main()
