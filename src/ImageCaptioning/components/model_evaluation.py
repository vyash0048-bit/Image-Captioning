import sys
import torch
import json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu
from ImageCaptioning.logger import logger
from ImageCaptioning.exception import CustomException
from ImageCaptioning.entity.config_entity import ModelEvaluationConfig
from ImageCaptioning.components.model import EncoderCNN, DecoderRNN
from ImageCaptioning.components.data_loader import get_loader
import torchvision.transforms as transforms
import pickle
import os
import urllib.parse
import mlflow
import mlflow.pytorch
import dagshub

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        try:
            logger.info("Initializing Data Loader for Evaluation...")
            transform_test = transforms.Compose([ 
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))])

            data_loader = get_loader(transform=transform_test,
                                     mode='train',
                                     batch_size=1,
                                     vocab_threshold=4,
                                     vocab_file=str(self.config.vocab_file),
                                     vocab_from_file=True)
            
            vocab_size = len(data_loader.dataset.vocab)
            vocab = data_loader.dataset.vocab

            logger.info("Initializing models for evaluation...")
            embed_size = 256
            hidden_size = 512
            
            encoder = EncoderCNN(embed_size)
            decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Loading weights from {self.config.model_path}")
            if Path(self.config.model_path).exists():
                checkpoint = torch.load(self.config.model_path, map_location=device)
                encoder.load_state_dict(checkpoint['encoder'])
                decoder.load_state_dict(checkpoint['decoder'])
                
            encoder.to(device)
            decoder.to(device)
            encoder.eval()
            decoder.eval()

            logger.info("Starting evaluation loop...")
            references = []
            hypotheses = []
            
            with torch.no_grad():
                for i_step in range(1, 10):
                    images, captions = next(iter(data_loader))
                    images = images.to(device)
                    
                    features = encoder(images)
                    output = decoder.sample(features.unsqueeze(1))
                    
                    pred_sentence = [vocab.idx2word[idx] for idx in output]
                    hypotheses.append(pred_sentence)
                    
                    ref_sentence = []
                    for idx in captions[0].numpy():
                        word = vocab.idx2word[idx]
                        if word == '<end>':
                            break
                        if word != '<start>':
                            ref_sentence.append(word)
                    references.append([ref_sentence])
            
            bleu_score = corpus_bleu(references, hypotheses)
            logger.info(f"Computed BLEU score: {bleu_score}")
            
            metrics = {"bleu_score": bleu_score}
            with open(self.config.metrics_file, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"Metrics saved to {self.config.metrics_file}")
            
            logger.info("Logging to MLflow on DagsHub...")
            os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vyash0048/Image-Captioning.mlflow"
            dagshub.init(repo_owner='vyash0048', repo_name='Image-Captioning', mlflow=True)

            with mlflow.start_run():
                mlflow.log_params({"embed_size": embed_size, "hidden_size": hidden_size})
                mlflow.log_metric("bleu_score", bleu_score)
                
                tracking_url_type_store = urllib.parse.urlparse(mlflow.get_tracking_uri()).scheme
                
                if tracking_url_type_store != "file":
                    mlflow.pytorch.log_model(encoder, "encoder_model", registered_model_name="ResNet50_Encoder")
                    mlflow.pytorch.log_model(decoder, "decoder_model", registered_model_name="LSTM_Decoder")
                else:
                    mlflow.pytorch.log_model(encoder, "encoder_model")
                    mlflow.pytorch.log_model(decoder, "decoder_model")
            logger.info("MLflow logging complete.")
            
        except Exception as e:
            raise CustomException(e, sys)
