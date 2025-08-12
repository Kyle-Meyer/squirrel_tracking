import argparse
from ultralytics import YOLO
import cv2
from pathlib import Path

kDefaultModelPath = "best_model.pt"
kDefaultConfThreshold = 0.5
kDemoResultsProject = "demo_results"
kDemoResultsName = "predictions"

def runDemo(aImagePath, aModelPath=kDefaultModelPath):
    """Run squirrel detection on a single image"""
    
    # Load model
    tModel = YOLO(aModelPath)
    
    # Run prediction
    tResults = tModel.predict(
        source=aImagePath,
        save=True,
        project=kDemoResultsProject,
        name=kDemoResultsName,
        conf=kDefaultConfThreshold
    )
    
    # Print results
    for tResult in tResults:
        if tResult.boxes is not None:
            tNumSquirrels = len(tResult.boxes)
            tConfidences = tResult.boxes.conf.cpu().numpy()
            print(f"Found {tNumSquirrels} squirrel(s)")
            for tI, tConf in enumerate(tConfidences):
                print(f"  Squirrel {tI+1}: {tConf:.3f} confidence")
        else:
            print("No squirrels detected")
    
    print(f"Results saved to: {kDemoResultsProject}/{kDemoResultsName}/")

if __name__ == "__main__":
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--image", required=True, help="Path to image file")
    tParser.add_argument("--model", default=kDefaultModelPath, help="Path to model file")
    tArgs = tParser.parse_args()
    
    runDemo(tArgs.image, tArgs.model)
