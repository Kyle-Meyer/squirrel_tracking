import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.downloader import ImageDownloader
from src.squirrel_trainer import SquirrelYOLOTrainer
import os

kProjectPath = "../yolo_squirrel_project"
kRunsPath = "runs"
kWeightsPath = "weights"
kBestModelFile = "best.pt"
kLastModelFile = "last.pt"
kSquirrelDetectionPrefix = "squirrel_detection"
kPredictionsPrefix = "predictions"

kDefaultDataSetPath = "../resources/squirrel_base_images"
kDefaultModelVariant = "yolov8n.pt"
kDefaultEpochs = 50
kDefaultImageSize = 640
kDefaultBatchSize = 8
kDefaultDevice = 'cpu'
kDefaultPatience = 20
kDefaultSavePeriod = 10
kContinuedTrainingPatience = 30
kSamplePredictionCount = 5

def findLatestModel(aProjectPath=None):
    if aProjectPath is None:
        # Get absolute path from src/ directory
        aProjectPath = Path(__file__).parent.parent / "yolo_squirrel_project"
    else:
        aProjectPath = Path(aProjectPath)
    
    tRunsPath = aProjectPath / kRunsPath 
    
    if not tRunsPath.exists():
        print("No previous training runs found.")
        return None
    
    # Look for training directories
    tTrainDirs = [tDir for tDir in tRunsPath.iterdir() if tDir.is_dir() and kSquirrelDetectionPrefix in tDir.name]
    
    if not tTrainDirs:
        print("No squirrel detection training runs found.")
        return None
    
    # Sort by modification time to get the latest
    tLatestDir = max(tTrainDirs, key=lambda x: x.stat().st_mtime)
    
    # Look for best.pt and last.pt
    tBestModel = tLatestDir / kWeightsPath / kBestModelFile
    tLastModel = tLatestDir / kWeightsPath / kLastModelFile
    
    if tBestModel.exists():
        print(f"Found best model: {tBestModel}")
        return str(tBestModel)
    elif tLastModel.exists():
        print(f"Found last model: {tLastModel}")
        return str(tLastModel)
    else:
        print(f"No model weights found in {tLatestDir}")
        return None

def evaluateModelPerformance(aTrainer, aModelPath=None):
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Check annotation status first
    print("\nDataset Statistics:")
    tStats = aTrainer.checkAnnotations()
    tTotalAnnotated = sum(tS['annotated'] for tS in tStats.values())
    tTotalImages = sum(tS['total'] for tS in tStats.values())
    
    print(f"Total images: {tTotalImages}")
    print(f"Annotated images: {tTotalAnnotated}")
    print(f"Annotation coverage: {tTotalAnnotated/tTotalImages*100:.1f}%")
    
    if tTotalAnnotated == 0:
        print("ERROR: No annotated images found! Cannot evaluate model.")
        return None
    
    # Run validation on test set
    print("\nRunning Model Validation...")
    try:
        tResults = aTrainer.evaluateModel(aModelPath)
        
        print("\nModel Performance Metrics:")
        if hasattr(tResults, 'box'):
            tMetrics = tResults.box
            print(f"  • mAP50: {tMetrics.map50:.3f} (mean Average Precision at IoU=0.5)")
            print(f"  • mAP50-95: {tMetrics.map:.3f} (mAP across IoU 0.5-0.95)")
            print(f"  • Precision: {tMetrics.mp:.3f}")
            print(f"  • Recall: {tMetrics.mr:.3f}")
            
            # Performance interpretation
            print("\nPerformance Interpretation:")
            if tMetrics.map50 > 0.8:
                print("  EXCELLENT: Model is detecting squirrels very well.")
            elif tMetrics.map50 > 0.6:
                print("  GOOD: Model is working well with room for improvement.")
            elif tMetrics.map50 > 0.4:
                print("  MODERATE: Consider more training or data.")
            else:
                print("  POOR: More training data or different approach needed.")
                
        else:
            print("  WARNING: Detailed metrics not available, but validation completed.")
            
    except Exception as tException:
        print(f"ERROR: Evaluation failed: {tException}")
        return None
    
    return tResults

def testSamplePredictions(aTrainer, aModelPath=None):
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS TEST")
    print("="*60)
    
    # Get some test images (relative to src/ directory)
    tTestImagesDir = Path(__file__).parent.parent / "yolo_squirrel_project" / "dataset" / "test" / "images"
    
    if not tTestImagesDir.exists():
        print("ERROR: No test images directory found.")
        return
    
    tTestImages = list(tTestImagesDir.glob("*.jpg"))[:kSamplePredictionCount]  # Test on first 5 images
    
    if not tTestImages:
        print("ERROR: No test images found.")
        return
    
    print(f"Testing on {len(tTestImages)} sample images...")
    
    try:
        for tI, tImgPath in enumerate(tTestImages):
            print(f"\nTesting image {tI+1}: {tImgPath.name}")
            tResults = aTrainer.predictSample(str(tImgPath), aModelPath, aSaveResults=True)
            
            # Analyze results
            if tResults and len(tResults) > 0:
                tResult = tResults[0]
                if hasattr(tResult, 'boxes') and tResult.boxes is not None:
                    tNumDetections = len(tResult.boxes)
                    if tNumDetections > 0:
                        tConfidences = tResult.boxes.conf.cpu().numpy()
                        tMaxConf = max(tConfidences)
                        tAvgConf = sum(tConfidences) / len(tConfidences)
                        print(f"  SUCCESS: Found {tNumDetections} squirrel(s)")
                        print(f"  Confidence - Max: {tMaxConf:.3f}, Avg: {tAvgConf:.3f}")
                    else:
                        print("  NO DETECTION: No squirrels detected")
                else:
                    print("  NO DETECTION: No detections")
            else:
                print("  ERROR: Prediction failed")
                
    except Exception as tException:
        print(f"ERROR: Sample prediction failed: {tException}")

def main():
    print("SQUIRREL DETECTION - CONTINUED TRAINING & EVALUATION")
    print("="*70)
    
    # Initialize trainer with paths relative to src/ directory
    tTrainer = SquirrelYOLOTrainer(
        aDataSetPath=kDefaultDataSetPath,
        aModelVariant=kDefaultModelVariant  # Will be overridden if continuing training
    )
    
    # Check for existing models
    tLatestModel = findLatestModel()
    
    print("\nTRAINING OPTIONS:")
    print("1. Continue training from latest model")
    print("2. Start fresh training")
    print("3. Evaluate existing model only")
    print("4. Test sample predictions")
    
    tChoice = input("\nEnter your choice (1-4): ").strip()
    
    if tChoice == "1":
        if tLatestModel:
            print(f"\nContinuing training from: {tLatestModel}")
            # Load the existing model
            tTrainer.mModelVariant = tLatestModel
            tTrainer.loadModel()
            
            # Continue training with more epochs
            print("\nStarting continued training...")
            tResults = tTrainer.trainModel(
                epochs=kDefaultEpochs,           # Additional epochs
                imgsz=kDefaultImageSize,
                batch=kDefaultBatchSize,         # Adjust based on your GPU memory
                device=kDefaultDevice,          # 'cuda' if you have GPU, 'cpu' otherwise
                patience=kDefaultPatience,       # Early stopping patience
                save_period=kDefaultSavePeriod  # Save checkpoint every 10 epochs
            )
            print("SUCCESS: Continued training completed!")
            
            # Evaluate the updated model
            evaluateModelPerformance(tTrainer)
            
        else:
            print("ERROR: No existing model found. Starting fresh training...")
            tChoice = "2"
    
    if tChoice == "2":
        print("\nStarting fresh training...")
        
        # Check annotations first
        tStats = tTrainer.checkAnnotations()
        tTotalAnnotated = sum(tS['annotated'] for tS in tStats.values())
        
        if tTotalAnnotated == 0:
            print("ERROR: No annotated images found! Please annotate your images first.")
            tTrainer.getAnnotationGuide()
            return
            
        print(f"SUCCESS: Found {tTotalAnnotated} annotated images. Starting training...")
        
        tResults = tTrainer.trainModel(
            epochs=kDefaultEpochs,          # Full training epochs
            imgsz=kDefaultImageSize,
            batch=kDefaultBatchSize,
            device=kDefaultDevice,
            patience=kContinuedTrainingPatience
        )
        print("SUCCESS: Training completed!")
        
        # Evaluate the new model
        evaluateModelPerformance(tTrainer)
    
    elif tChoice == "3":
        if tLatestModel:
            print(f"\nEvaluating model: {tLatestModel}")
            evaluateModelPerformance(tTrainer, tLatestModel)
            testSamplePredictions(tTrainer, tLatestModel)
        else:
            print("ERROR: No trained model found to evaluate.")
    
    elif tChoice == "4":
        if tLatestModel:
            print(f"\nTesting predictions with: {tLatestModel}")
            testSamplePredictions(tTrainer, tLatestModel)
        else:
            print("ERROR: No trained model found for testing.")
    
    else:
        print("ERROR: Invalid choice. Please run again and select 1-4.")
    
    print("\n" + "="*70)
    print("SQUIRREL DETECTION SESSION COMPLETE!")
    print("="*70)
    
    # Show where results are saved (relative to project root)
    print("\nResults saved in:")
    print(f"  • Training runs: yolo_squirrel_project/runs/detect/")
    print(f"  • Model weights: yolo_squirrel_project/runs/detect/{kSquirrelDetectionPrefix}*/weights/")
    print(f"  • Predictions: yolo_squirrel_project/runs/detect/{kPredictionsPrefix}*/")

if __name__ == "__main__":
    main()
