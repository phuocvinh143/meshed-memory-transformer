from data.dataset import Flickr8k
from data import ImageDetectionsField, TextField
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from PIL import Image
import torchvision.transforms.functional as TF

import torch


features_path = 'flickr8k_en_features.hdf5'
annotation_folder = '/Users/phuocvinh143/Git/bottom-up-attention/data/flickr8k_vn'

# Pipeline for image regions
image_field = ImageDetectionsField(detections_path=features_path, max_detections=50, load_in_tmp=False)

# Pipeline for text
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                        remove_punctuation=True, nopoints=False)

# Create the dataset
dataset = Flickr8k(image_field, text_field, '/Users/phuocvinh143/Git/bottom-up-attention/data/flickr8k_vn/Images', annotation_folder, annotation_folder)
train_dataset, val_dataset, test_dataset = dataset.splits
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

device = torch.device('cuda')

# Model and dataloaders
encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                    attention_module_kwargs={'m': 40})
decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

data = torch.load('saved_models/m2_transformer_best.pth')
model.load_state_dict(data['state_dict'])

image = Image.open('images/m2.png')
x = TF.to_tensor(image)

with torch.no_grad():
    out, _ = model.beam_search(x, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)

caps_gen = text_field.decode(out, join_words=False)