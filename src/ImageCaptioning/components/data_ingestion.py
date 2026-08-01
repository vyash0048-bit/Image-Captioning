import os
import sys
import urllib.request as request
import zipfile
from ImageCaptioning.logger import logger
from ImageCaptioning.exception import CustomException
from ImageCaptioning.utils.common import get_size
from ImageCaptioning.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.info(f"Downloading data from {self.config.source_URL}...")
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logger.info(f"{filename} downloaded! with following info: \n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  
        except Exception as e:
            raise CustomException(e, sys)

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            logger.info(f"Extracting zip file to {unzip_path}...")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted data to {unzip_path}")
        except Exception as e:
            raise CustomException(e, sys)
