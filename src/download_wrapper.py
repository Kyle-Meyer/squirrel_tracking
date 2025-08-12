#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.downloader import ImageDownloader
from src.squirrel_trainer import SquirrelYOLOTrainer

kProjectBasePath = "../yolo_squirrel_project/dataset"
kClassesFileName = "classes.txt"
kClassContent = "squirrel\n"
kTempCsvName = "../temp_download.csv"
kRandomSeed = 42
kImageExtension = "*.jpg"
kSquirrelPrefix = "squirrel_"

kValidSplits = ['train', 'val', 'test']
kRequiredColumns = ['image_url', 'id']
kQualityGradeColumn = 'quality_grade'
kCommonNameColumn = 'common_name'
kResearchGrade = 'research'

class DirectTrainingDownloader:
    def __init__(self):
        # Initialize the trainer to set up project structure
        self.mTrainer = SquirrelYOLOTrainer()
        
        # Set up paths for all splits (relative to src/ directory)
        self.mBasePath = Path(__file__).parent.parent / "yolo_squirrel_project" / "dataset"
        self.mSplits = {
            'train': {
                'images': self.mBasePath / "train" / "images",
                'labels': self.mBasePath / "train" / "labels"
            },
            'val': {
                'images': self.mBasePath / "val" / "images", 
                'labels': self.mBasePath / "val" / "labels"
            },
            'test': {
                'images': self.mBasePath / "test" / "images",
                'labels': self.mBasePath / "test" / "labels"
            }
        }
        
    def setupDirs(self, aSplit='train'):
        # Project structure is already created by SquirrelYOLOTrainer
        # Just ensure our specific directories exist
        tImagesDir = self.mSplits[aSplit]['images']
        tLabelsDir = self.mSplits[aSplit]['labels']
        
        tImagesDir.mkdir(parents=True, exist_ok=True)
        tLabelsDir.mkdir(parents=True, exist_ok=True)
        
        # Ensure classes.txt exists
        tClassesFile = tLabelsDir / kClassesFileName
        if not tClassesFile.exists():
            tClassesFile.write_text(kClassContent)
            print(f"Created {kClassesFileName} in {tLabelsDir}")
        
        print(f"{aSplit.upper()} directories ready:")
        print(f"   Images: {tImagesDir}")
        print(f"   Labels: {tLabelsDir}")
        
        return tImagesDir, tLabelsDir
        
    def getExistingImageIds(self, aSplit='train'):
        """Get existing image IDs only from the specified split folder"""
        tExistingIds = set()
        
        tImagesDir = self.mSplits[aSplit]['images']
        if tImagesDir.exists():
            for tImgFile in tImagesDir.glob(kImageExtension):
                if tImgFile.name.startswith(kSquirrelPrefix):
                    tImgId = tImgFile.stem.replace(kSquirrelPrefix, "")
                    tExistingIds.add(tImgId)
                    
        print(f"Found {len(tExistingIds)} existing image IDs in {aSplit} set")
        return tExistingIds
        
    def downloadToSplit(self, aCsvFile, aSplit='train', aMaxImages=0, aSkipExisting=True):
        
        print(f"DOWNLOADING TO {aSplit.upper()} FOLDER FROM: {aCsvFile}")
        print("="*60)
        
        if aSplit not in kValidSplits:
            print(f"ERROR: Invalid split '{aSplit}'. Must be one of: {kValidSplits}")
            return False
        
        # Convert relative path to absolute from src/ directory
        tCsvPath = Path(__file__).parent.parent / aCsvFile
        if not tCsvPath.exists():
            print(f"ERROR: CSV file not found: {tCsvPath}")
            return False
            
        # Setup directories for the specified split
        tImagesDir, tLabelsDir = self.setupDirs(aSplit)
        
        # Load CSV and check structure
        try:
            tDf = pd.read_csv(tCsvPath)
            print(f"CSV loaded: {len(tDf)} total records")
        except Exception as tException:
            print(f"ERROR: Error reading CSV: {tException}")
            return False
            
        # Check required columns
        for tColumn in kRequiredColumns:
            if tColumn not in tDf.columns:
                print(f"ERROR: CSV must have '{tColumn}' column")
                print(f"Available columns: {list(tDf.columns)}")
                return False
            
        # First filter for images with URLs
        tDfWithUrls = tDf.dropna(subset=[kRequiredColumns[0]])
        
        # Show dataset info
        print(f"Dataset info:")
        print(f"   - Total observations: {len(tDf)}")
        print(f"   - With image URLs: {len(tDfWithUrls)}")
        
        if kCommonNameColumn in tDf.columns:
            tSpeciesCounts = tDf[kCommonNameColumn].value_counts().head(5)
            print(f"   - Top species: {tSpeciesCounts.to_dict()}")
            
        if kQualityGradeColumn in tDf.columns:
            tQualityCounts = tDf[kQualityGradeColumn].value_counts()
            print(f"   - Quality grades: {tQualityCounts.to_dict()}")
            
        # Filter out images we already have (if requested)
        if aSkipExisting:
            tExistingImages = self.getExistingImageIds(aSplit)
            tInitialCount = len(tDfWithUrls)
            
            # Debug: Show some info about what we're filtering
            print(f"Checking {tInitialCount} records against {len(tExistingImages)} existing images in {aSplit}")
            
            tDfWithUrls = tDfWithUrls[~tDfWithUrls[kRequiredColumns[1]].astype(str).isin(tExistingImages)]
            tFilteredCount = tInitialCount - len(tDfWithUrls)
            print(f"Filtered out {tFilteredCount} existing images")
            
            # Debug: If we have very few remaining, show what they are
            if len(tDfWithUrls) < 10:
                print(f"Remaining records:")
                for _, tRow in tDfWithUrls.iterrows():
                    tHasUrl = pd.notna(tRow.get(kRequiredColumns[0], None)) and tRow.get(kRequiredColumns[0], '') != ''
                    print(f"   ID: {tRow[kRequiredColumns[1]]}, Has URL: {tHasUrl}")
                    
        else:
            print(f"Skipping existing image check (--force mode)")
            
        # Limit to aMaxImages (but allow downloading all if aMaxImages is 0 or large)
        if aMaxImages > 0 and len(tDfWithUrls) > aMaxImages:
            # Prioritize research grade if available
            if kQualityGradeColumn in tDfWithUrls.columns:
                tResearchGrade = tDfWithUrls[tDfWithUrls[kQualityGradeColumn] == kResearchGrade]
                if len(tResearchGrade) >= aMaxImages:
                    tDfWithUrls = tResearchGrade.sample(n=aMaxImages, random_state=kRandomSeed)
                    print(f"Selected {aMaxImages} research-grade images")
                else:
                    tDfWithUrls = tDfWithUrls.sample(n=aMaxImages, random_state=kRandomSeed)
                    print(f"Randomly selected {aMaxImages} images")
            else:
                tDfWithUrls = tDfWithUrls.sample(n=aMaxImages, random_state=kRandomSeed)
                print(f"Randomly selected {aMaxImages} images")
        else:
            print(f"Will attempt to download all {len(tDfWithUrls)} images")
            
        print(f"Final selection: {len(tDfWithUrls)} records ready for download")
        
        if len(tDfWithUrls) == 0:
            print("ERROR: No images to download!")
            return False
            
        # Use the existing downloader but point to the split folder
        tDownloader = ImageDownloader(
            aCsvFile=str(tCsvPath), 
            aOutputFolder=str(tImagesDir)
        )
        
        # Create temporary CSV with only the images we want (in parent directory)
        tTempCsv = Path(__file__).parent.parent / "temp_download.csv"
        tDfWithUrls.to_csv(tTempCsv, index=False)
        
        # Update downloader to use filtered CSV
        tDownloader.mCsvFile = str(tTempCsv)
        
        print(f"Starting download of {len(tDfWithUrls)} images to {aSplit} folder...")
        tDownloader.downloadAllImages()
        
        # Cleanup temp file
        tTempCsv.unlink(missing_ok=True)
        
        # Create empty annotation files for new images
        self.createEmptyLabels(aSplit)
        
        # Create dataset.yaml file
        self.mTrainer.createDataSetYaml()
        
        # Report results
        tDownloadedImages = list(tImagesDir.glob(kImageExtension))
        print(f"\nDownload complete!")
        print(f"Total images in {aSplit} folder: {len(tDownloadedImages)}")
        print(f"Full project structure created at: yolo_squirrel_project/")
        
        return len(tDownloadedImages) > 0
        
    def createEmptyLabels(self, aSplit='train'):
        tImagesDir = self.mSplits[aSplit]['images']
        tLabelsDir = self.mSplits[aSplit]['labels']
        
        if not tImagesDir.exists():
            return 0
            
        tCreatedCount = 0
        for tImgFile in tImagesDir.glob(kImageExtension):
            tLabelFile = tLabelsDir / f"{tImgFile.stem}.txt"
            if not tLabelFile.exists():
                tLabelFile.touch()  # Create empty file
                tCreatedCount += 1
                
        if tCreatedCount > 0:
            print(f"Created {tCreatedCount} empty annotation files in {aSplit}")
        return tCreatedCount
        
    def showTrainingStats(self):
        print(f"\nDATASET STATISTICS")
        print("="*50)
        
        for tSplit in kValidSplits:
            tImagesDir = self.mSplits[tSplit]['images']
            tLabelsDir = self.mSplits[tSplit]['labels']
            
            if not tImagesDir.exists():
                print(f"{tSplit.upper()}: No directory found")
                continue
                
            tImages = list(tImagesDir.glob(kImageExtension))
            tAnnotated = 0
            
            if tLabelsDir.exists():
                for tImg in tImages:
                    tLabelFile = tLabelsDir / f"{tImg.stem}.txt"
                    if tLabelFile.exists() and tLabelFile.stat().st_size > 0:
                        tAnnotated += 1
                        
            tCoverage = tAnnotated/max(1,len(tImages))*100
            print(f"{tSplit.upper()}: {len(tImages)} images, {tAnnotated} annotated ({tCoverage:.1f}%)")
        
        # Overall annotation guidance
        tTrainImages = list(self.mSplits['train']['images'].glob(kImageExtension)) if self.mSplits['train']['images'].exists() else []
        if len(tTrainImages) > 0:
            tTrainLabelsDir = self.mSplits['train']['labels']
            tTrainAnnotated = 0
            if tTrainLabelsDir.exists():
                for tImg in tTrainImages:
                    tLabelFile = tTrainLabelsDir / f"{tImg.stem}.txt"
                    if tLabelFile.exists() and tLabelFile.stat().st_size > 0:
                        tTrainAnnotated += 1
            
            if tTrainAnnotated == 0:
                print(f"\nREADY FOR ANNOTATION:")
                print(f"Images to annotate: {self.mSplits['train']['images']}")
                print(f"Save annotations to: {self.mSplits['train']['labels']}")
                print(f"Use LabelImg or similar tool")
            else:
                print(f"\nANNOTATION PROGRESS:")
                print(f"Annotated: {tTrainAnnotated}")
                print(f"Remaining: {len(tTrainImages) - tTrainAnnotated}")

def main():
    print("DIRECT TRAINING FOLDER DOWNLOADER")
    print("="*50)
    
    tDownloader = DirectTrainingDownloader()
    
    if len(sys.argv) < 2:
        print("USAGE:")
        print("  python src/download_wrapper.py download_all <csv_file> [--force]")
        print("  python src/download_wrapper.py download <csv_file> <max_images> [split]")
        print("  python src/download_wrapper.py stats")
        print("  python src/download_wrapper.py prepare [split]")
        print("")
        print("EXAMPLES:")
        print("  python src/download_wrapper.py download_all observations603652.csv")
        print("  python src/download_wrapper.py download_all observations603652.csv --force")
        print("  python src/download_wrapper.py download observations603652.csv 100")
        print("  python src/download_wrapper.py download observations603652.csv 50 val")
        print("  python src/download_wrapper.py download observations603652.csv 25 test")
        print("")
        print("NOTE: --force flag will re-download existing images")
        print("NOTE: split can be: train (default), val, or test")
        print("NOTE: Images download to yolo_squirrel_project/dataset/<split>/images/")
        print("NOTE: Empty label files created in yolo_squirrel_project/dataset/<split>/labels/")
        print("NOTE: Full project structure will be created automatically!")
        print("")
        tDownloader.showTrainingStats()
        return
        
    tCommand = sys.argv[1].lower()
    
    if tCommand == "download_all":
        if len(sys.argv) < 3:
            print("ERROR: Please specify CSV file!")
            print("Usage: python src/download_wrapper.py download_all observations603652.csv")
            print("       python src/download_wrapper.py download_all observations603652.csv --force")
            return
            
        tCsvFile = sys.argv[2]
        tSkipExisting = True
        
        # Check for --force flag
        if len(sys.argv) > 3 and sys.argv[3] == "--force":
            tSkipExisting = False
            print(f"FORCE DOWNLOADING ALL IMAGES (including existing ones)")
        else:
            print(f"DOWNLOADING ALL IMAGES TO TRAINING FOLDER")
            
        print(f"This will create the full yolo_squirrel_project structure")
        
        tSuccess = tDownloader.downloadToSplit(tCsvFile, aSplit='train', aMaxImages=0, aSkipExisting=tSkipExisting)
        if tSuccess:
            print(f"\nNEXT STEPS:")
            print(f"1. Use LabelImg to annotate images")
            print(f"2. Images: {tDownloader.mSplits['train']['images']}")
            print(f"3. Save labels: {tDownloader.mSplits['train']['labels']}")
            print(f"4. Train model: python src/main.py")
            
    elif tCommand == "download":
        if len(sys.argv) < 4:
            print("ERROR: Please specify CSV file and max images!")
            print("Usage: python src/download_wrapper.py download observations603652.csv 100 [split]")
            return
            
        tCsvFile = sys.argv[2]
        tMaxImages = int(sys.argv[3])
        tSplit = sys.argv[4] if len(sys.argv) > 4 else 'train'
        
        if tSplit not in kValidSplits:
            print(f"ERROR: Invalid split '{tSplit}'. Must be: {', '.join(kValidSplits)}")
            return
            
        print(f"DOWNLOADING UP TO {tMaxImages} IMAGES TO {tSplit.upper()} FOLDER")
        print(f"This will create the full yolo_squirrel_project structure")
        
        tSuccess = tDownloader.downloadToSplit(tCsvFile, aSplit=tSplit, aMaxImages=tMaxImages)
        if tSuccess:
            print(f"\nNEXT STEPS:")
            if tSplit == 'train':
                print(f"1. Use LabelImg to annotate images")
                print(f"2. Images: {tDownloader.mSplits[tSplit]['images']}")
                print(f"3. Save labels: {tDownloader.mSplits[tSplit]['labels']}")
                print(f"4. Train model: python src/main.py")
            else:
                print(f"1. Use LabelImg to annotate {tSplit} images")
                print(f"2. Images: {tDownloader.mSplits[tSplit]['images']}")
                print(f"3. Save labels: {tDownloader.mSplits[tSplit]['labels']}")
                print(f"4. These will be used for {tSplit} during training")
            
    elif tCommand == "prepare":
        tSplit = sys.argv[2] if len(sys.argv) > 2 else 'train'
        if tSplit not in kValidSplits:
            print(f"ERROR: Invalid split '{tSplit}'. Must be: {', '.join(kValidSplits)}")
            return
            
        tCount = tDownloader.createEmptyLabels(tSplit)
        print(f"Ready for annotation in {tSplit} set!")
            
    elif tCommand == "stats":
        tDownloader.showTrainingStats()
        
    else:
        print(f"ERROR: Unknown command: {tCommand}")
        print("Valid commands: download_all, download, stats, prepare")

if __name__ == "__main__":
    main()
