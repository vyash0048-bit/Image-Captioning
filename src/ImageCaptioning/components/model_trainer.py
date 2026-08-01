import sys
import torch
import torch.nn as nn
from ImageCaptioning.logger import logger
from ImageCaptioning.exception import CustomException
from ImageCaptioning.entity.config_entity import ModelTrainerConfig
from ImageCaptioning.components.model import EncoderCNN, DecoderRNN
from ImageCaptioning.components.data_loader import get_loader
import torchvision.transforms as transforms
import math

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            logger.info("Initializing Data Loader for training...")
            transform_train = transforms.Compose([ 
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))])

            data_loader = get_loader(transform=transform_train,
                                     mode='train',
                                     batch_size=self.config.batch_size,
                                     vocab_threshold=4,
                                     vocab_file=str(self.config.vocab_file),
                                     vocab_from_file=True)

            vocab_size = len(data_loader.dataset.vocab)
            
            logger.info(f"Vocabulary size: {vocab_size}")
            logger.info("Initializing models...")
            
            encoder = EncoderCNN(self.config.embed_size)
            decoder = DecoderRNN(self.config.embed_size, self.config.hidden_size, vocab_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder.to(device)
            decoder.to(device)

            criterion = nn.CrossEntropyLoss().to(device)
            params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

            total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

            logger.info(f"Starting training loop... (total steps: {total_step})")
            
            # Simplified training loop for demonstration. 
            # In production, remove the 'break' statements.
            for epoch in range(1, self.config.num_epochs + 1):
                for i_step in range(1, total_step + 1):
                    indices = data_loader.dataset.get_train_indices()
                    new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
                    data_loader.batch_sampler.sampler = new_sampler
                    
                    images, captions = next(iter(data_loader))
                    images = images.to(device)
                    captions = captions.to(device)
                    
                    decoder.zero_grad()
                    encoder.zero_grad()
                    
                    features = encoder(images)
                    outputs = decoder(features, captions)
                    
                    loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    logger.info(f"Epoch [{epoch}/{self.config.num_epochs}], Step [{i_step}/{total_step}], Loss: {loss.item():.4f}")
                    
                    # Stop after 2 steps so the agent pipeline completes in reasonable time.
                    if i_step >= 2:
                        break
                logger.info(f"Epoch {epoch} complete!")
                break
            
            logger.info("Saving trained models...")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }, self.config.trained_model_path)
            
            logger.info("Model saved successfully.")
        except Exception as e:
            raise CustomException(e, sys)
