import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
from ImageCaptioning.config.configuration import ConfigurationManager
from ImageCaptioning.components.vocabulary import Vocabulary
from ImageCaptioning.components.data_loader import get_loader
from ImageCaptioning.logger import logger
from ImageCaptioning.exception import CustomException
import sys
import nltk

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            
            # Download NLTK tokenizer if not already downloaded
            nltk.download('punkt')
            nltk.download('punkt_tab')
            
            logger.info("Building vocabulary from annotations...")
            
            import torchvision.transforms as transforms
            transform_train = transforms.Compose([ 
                transforms.Resize(256),                          # smaller edge of image resized to 256
                transforms.RandomCrop(224),                      # get 224x224 crop from random location
                transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))])
                                     
            # This will create vocab.pkl if vocab_from_file is False
            data_loader = get_loader(transform=transform_train,
                                     mode='train',
                                     batch_size=1,
                                     vocab_threshold=4, 
                                     vocab_file=str(data_transformation_config.vocab_file),
                                     vocab_from_file=False)
            logger.info("Vocabulary built successfully.")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
