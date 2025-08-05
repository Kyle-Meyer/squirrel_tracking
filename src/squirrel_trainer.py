import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO 
import cv2 
import numpy as np 
from sklearn.model_selection import train_test_split 
import logging

class SquirrelYOLOTrainer:
    def __init__(self, aDataSetPath="resources/squirrel_base_images", aModelVariant='yolov8n.pt'):
        self.mDataSetPath = Path(aDataSetPath)
        self.mModelVariant = aModelVariant
        self.mProjectRoot = Path("yolo_squirrel_project")
        self.mYoloDataSetPath = self.mProjectRoot / "dataset"

        self._setupProjectStructure()
        
        logging.basicConfig(level=logging.INFO)
        self.mLogger = logging.getLogger(__name__)

        self.mModel = None 

    def _setupProjectStructure(self):
        tDirectories = [
                self.mProjectRoot,
                self.mYoloDataSetPath,
                self.mYoloDataSetPath / "train" / "images",
                self.mYoloDataSetPath / "train" / "labels",
                self.mYoloDataSetPath / "val" / "images",
                self.mYoloDataSetPath / "val" / "labels",
                self.mYoloDataSetPath / "test" / "images",
                self.mYoloDataSetPath / "test" / "labels",
                self.mYoloDataSetPath / "runs",
                self.mYoloDataSetPath / "models",
        ]

        for tDirectory in tDirectories:
            tDirectory.mkdir(parents=True, exist_ok=True)

        self.mLogger.info(f"Project structure created at {self.mProjectRoot}")

    def prepareDataSet(self, aTrainSplit=0.7, aValSplit=0.2, aTestSplit=0.1):
        if not self.mDataSetPath.exists():
            raise FileNotFoundError(f"Dataset path {self.mDataSetPath} does not exist")
        
        tImageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        tImageFiles = []

        for tExt in tImageExtensions:
            tImageFiles.extend(list(self.mDataSetPath.glob(f"{tExt}")))
            tImageFiles.extend(list(self.mDataSetPath.glob(f"{tExt.upper()}")))

        if not tImageFiles:
            raise ValueError(f"No Images found in {self.mDataSetPath}")

        self.mLogger.info(f"found {len(tImageFiles)} images")

        #split 
        tTrainFiles, tTempFiles = train_test_split(tImageFiles, test_size=(1-aTrainSplit), random_state=12)
        tValFiles, tTestFiles = train_test_split(tTempFiles, test_size=(aTestSplit/(aValSplit+aTestSplit)), random_state=12)

        tSplits = {
                'train': tTrainFiles,
                'val': tValFiles,
                'test': tTestFiles
        }

        for tSplitName, tFiles in tSplits.items():
            self.mLogger.info(f"Processing {tSplitName} set: {len(tFiles)} images")

            for tImgFile in tFiles:
                tDstImg = self.mYoloDataSetPath / tSplitName / "images" / tImgFile.name 
                shutil.copy2(tImgFile, tDstImg)

                tLabelName = tImgFile.stem + ".txt"
                tLabelFile = self.mYoloDataSetPath / tSplitName / "labels" / tLabelName
                tLabelFile.touch()

            self.mLogger.info("Dataset preparation complete")
            self.mLogger.warning("Remember to annotate the images for training")

            return len(tTrainFiles), len(tValFiles), len(tTestFiles)

    def createDataSetYaml(self):
        tYamlContent = {
                'path': str(self.mYoloDataSetPath.absolute()),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': 1,    #number of classes
                'names': ['squirrel']
        }

        tYamlFile = self.mYoloDataSetPath / "dataset.yaml"
        with open(tYamlFile, 'w') as tFile:
            yaml.dump(tYamlContent, tFile, default_flow_style=False)

        self.mLogger.info(f"Dataset YAMEL created {tYamlFile}")
        return tYamlFile

    def checkAnnotations(self):
        tSplits = ['train', 'val', 'test']
        tAnnotationStats = {}

        for tSplit in tSplits:
            tLabelDir = self.mYoloDataSetPath / tSplit / "labels"
            tLabelFiles = list(tLabelDir.glob("*.txt"))

            tAnnotatedCount = 0 
            for tLabelFile in tLabelFiles:
                if tLabelFile.stat().st_size > 0: #only non empty files 
                    tAnnotatedCount += 1 

            tAnnotationStats[tSplit] = {
                'total': len(tLabelFiles),
                'annotated': tAnnotatedCount,
                'empty': len(tLabelFiles) - tAnnotatedCount
            }
        self.mLogger.info("annotation status: ")
        for tSplit, tStats in tAnnotationStats.items():
            self.mLogger.info(f"    {tSplit}: {tStats['annotated']}/{tStats['total']} annotated")

        return tAnnotationStats
    
    def loadModel(self):
        self.mModel = YOLO(self.mModelVariant)
        self.mLogger.info(f"Loaded {self.mModelVariant} model")
        return self.mModel

    def trainModel(self, aEpochs=100, aImgsz=640, aBatch=16, aDevice='auto', **aKwargs):
        if self.mModel is None:
            self.loadModel()

        tYamlFile = self.mYoloDataSetPath / "dataset.yaml"
        if not tYamlFile.exists():
            self.createDataSetYaml()
        
        tStats = self.checkAnnotations()
        tTotalAnnotated = sum(tS['annotated'] for tS in tStats.values())

        if totalAnnotated == 0:
            raise ValueError("No annotated images found, please annotate them first")

        self.mLogger.info(f"Starting training {tTotalAnnotated} annotated images")

        tTrainArgs = {
            'data': str(tYamlFile),
            'epochs': aEpochs,
            'imgsz': aImgsz,
            'batch': aBatch,
            'device': aDevice,
            'project': str(self.mProjectRoot / "runs"),
            'name': 'squirrel_detection',
            'save_period': 10, #save checkpoint every 10 epochs 
            'patience': 50, #early stopping patience 
            **aKwargs
        }

        tResults = self.mModel.train(**tTrainArgs)

        self.mLogger.info("Training Completed!")
        return tResults

    def evaluateModel(self, aModelPath=None):
        if aModelPath:
            tModel = YOLO(aModelPath)
        elif self.mModel:
            tModel = self.mModel 
        else:
            raise ValueError("No model available for evaluation")

        tYamlFile = self.mYoloDataSetPath / "dataset.yaml"

        tResults = tModel.val(
            data=str(tYamlFile),
            split='test',
            project=str(self.mProjectRoot / "runs"),
            name='evaluation'
        )

        return tResults

    def predictSample(self, aImagePath, aModelPath=None, aSaveResults=True):
        if aModelPath:
            tModel = YOLO(aModelPath)
        elif self.mModel:
            tModel = self.mModel
        else:
            raise ValueError("No model available for prediction")
        
        tResults = tModel.predict(
            source=aImagePath,
            save=aSaveResults,
            project=str(self.mProjectRoot / "runs"),
            name='predictions'
        )
        
        return tResults

    
    def getAnnotationGuide(self):
        kGuide = """
        ANNOTATION GUIDE FOR SQUIRREL DETECTION:
        
        1. Use tools like LabelImg, Roboflow, or CVAT for annotation
        2. Each image needs a corresponding .txt file with the same name
        3. YOLO format: class_id center_x center_y width height (all normalized 0-1)
        4. For squirrels, class_id = 0
        5. Example annotation line: 0 0.5 0.5 0.3 0.4
        
        Annotation Tips:
        - Include the entire squirrel body in bounding box
        - Be consistent with partially visible squirrels
        - Annotate squirrels in all poses (sitting, climbing, jumping)
        - Include squirrels at different distances
        
        Recommended tools:
        - LabelImg: https://github.com/tzutalin/labelImg
        - Roboflow: https://roboflow.com (online, has auto-annotation)
        - CVAT: https://github.com/openvinotoolkit/cvat
        """
        
        print(kGuide)
        return kGuide
