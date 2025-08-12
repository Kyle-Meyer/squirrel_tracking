import argparse
from ultralytics import YOLO
from pathlib import Path
import json

kDefaultModelPath = "best_model.pt"
kDefaultImagesDir = "sample_images/"
kOutputFileName = "evaluation_results.json"
kImageExtension = "*.jpg"

def evaluateModel(aImagesDir, aModelPath=kDefaultModelPath):
    """Evaluate model on sample images"""
    
    tModel = YOLO(aModelPath)
    tImagesDir = Path(aImagesDir)
    
    tResultsSummary = {
        "model": aModelPath,
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "average_confidence": 0.0,
        "results": []
    }
    
    tAllConfidences = []
    
    for tImgFile in tImagesDir.glob(kImageExtension):
        tResults = tModel.predict(source=str(tImgFile), verbose=False)
        
        tImageResult = {
            "image": tImgFile.name,
            "detections": 0,
            "confidences": []
        }
        
        for tResult in tResults:
            if tResult.boxes is not None:
                tConfidences = tResult.boxes.conf.cpu().numpy().tolist()
                tImageResult["detections"] = len(tConfidences)
                tImageResult["confidences"] = tConfidences
                tAllConfidences.extend(tConfidences)
                
                if len(tConfidences) > 0:
                    tResultsSummary["images_with_detections"] += 1
        
        tResultsSummary["results"].append(tImageResult)
        tResultsSummary["total_images"] += 1
    
    # Calculate summary statistics
    tResultsSummary["total_detections"] = len(tAllConfidences)
    if tAllConfidences:
        tResultsSummary["average_confidence"] = sum(tAllConfidences) / len(tAllConfidences)
    
    # Save results
    with open(kOutputFileName, "w") as tFile:
        json.dump(tResultsSummary, tFile, indent=2)
    
    # Print summary
    print("="*50)
    print("SQUIRREL DETECTION MODEL EVALUATION")
    print("="*50)
    print(f"Model: {aModelPath}")
    print(f"Images tested: {tResultsSummary['total_images']}")
    print(f"Images with detections: {tResultsSummary['images_with_detections']}")
    print(f"Total detections: {tResultsSummary['total_detections']}")
    print(f"Average confidence: {tResultsSummary['average_confidence']:.3f}")
    
    tDetectionRate = tResultsSummary['images_with_detections']/tResultsSummary['total_images']*100 if tResultsSummary['total_images'] > 0 else 0
    print(f"Detection rate: {tDetectionRate:.1f}%")
    print(f"\nDetailed results saved to: {kOutputFileName}")

if __name__ == "__main__":
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--images", default=kDefaultImagesDir, help="Directory with test images")
    tParser.add_argument("--model", default=kDefaultModelPath, help="Path to model file")
    tArgs = tParser.parse_args()
    
    evaluateModel(tArgs.images, tArgs.model)
