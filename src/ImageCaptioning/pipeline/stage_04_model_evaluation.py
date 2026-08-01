import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
from ImageCaptioning.config.configuration import ConfigurationManager
from ImageCaptioning.components.model_evaluation import ModelEvaluation
from ImageCaptioning.logger import logger
from ImageCaptioning.exception import CustomException
import sys

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
