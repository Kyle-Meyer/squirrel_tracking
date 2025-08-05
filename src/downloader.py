import pandas as pd 
import requests 
import os
import time 
from pathlib import Path 

class ImageDownloader:
    def __init__(self, aCsvFile='squirrel_base_sets.csv', aOutputFolder='resources/squirrel_base_images'):
        self.mCsvFile = aCsvFile
        self.mOutputFolder = aOutputFolder
        self.mDownload = 0
        self.mFailed = 0

    def downloadAllImages(self):
        os.makedirs(self.mOutputFolder, exist_ok=True)

        print("loading csv file")
        tDataFrame = pd.read_csv(self.mCsvFile)

        tDataFrameWithImages = tDataFrame.dropna(subset=['image_url'])
        tTotalImages = len(tDataFrameWithImages)
        print(f"found {tTotalImages} images to download")

        for tIndex, tRow in tDataFrameWithImages.iterrows():
            self._downloadSingleImage(tRow, tIndex)

        print(f"\n=== Download Complete ===")
        print(f"Successfully downloaded: {self.mDownload} images")
        print(f"Failed downloads: {self.mFailed}")
        print(f"Images saved to: {self.mOutputFolder}/")

    def _downloadSingleImage(self, aRow, aIndex):
        tImageUrl = aRow['image_url']
        tObservationId = aRow['id']

        tFileName = f"squirrel_{tObservationId}.jpg"
        tFilePath = os.path.join(self.mOutputFolder, tFileName)

        if os.path.exists(tFilePath):
            print(f"Skipping {tFilePath}")
            self.mDownload += 1
            return 

        try:
            print(f"Downloading {self.mDownload + self.mFailed + 1}: {tFileName}")
            tHeaders = {'User-Agent': 'Mozilla/5.0 (compatible; SquirrelBot/1.0)'}
            tResponse = requests.get(tImageUrl, headers=tHeaders, timeout=10)
            tResponse.raise_for_status()

            #save it 
            with open(tFilePath, 'wb') as tFile:
                tFile.write(tResponse.content)

            self.mDownload += 1 
            print(f"✓ {tFileName}")
        except Exception as tException:
            print(f"✗ Failed {tFileName}: {tException}")
            self.mFailed += 1

        #be nice to the server, he does it for free 
        time.sleep(0.3)
