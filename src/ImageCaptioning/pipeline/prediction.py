import os
import torch
from ImageCaptioning.components.model import EncoderCNN, DecoderRNN
from ImageCaptioning.components.vocabulary import Vocabulary
from PIL import Image
import torchvision.transforms as transforms
import pickle
import warnings
warnings.filterwarnings("ignore")

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self):
        # Load vocab
        vocab_file = "artifacts/data_transformation/vocab.pkl"
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            
        # Hardcoding params for prediction simplicity (these should ideally be loaded from params.yaml)
        embed_size = 256
        hidden_size = 512
        vocab_size = len(vocab)
        
        # Initialize models
        encoder = EncoderCNN(embed_size)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "artifacts/model_trainer/model.pth"
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        
        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        
        # Process image
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        
        image = Image.open(self.filename).convert('RGB')
        image = transform_test(image).unsqueeze(0).to(device)
        
        # Get features and sample
        features = encoder(image)
        output = decoder.sample(features.unsqueeze(1))
        
        # Clean sentence
        sentence = []
        for idx in output:
            word = vocab.idx2word[idx]
            if word == '<end>':
                break
            if word != '<start>':
                sentence.append(word)
                
        return " ".join(sentence)
