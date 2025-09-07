from math import log
import os
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Latent, generate_pad_mask
from models.encoder_transformer import GVTransformerEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP


class IQ(nn.Module):
    """Information Maximization question generation.
    """

    def __init__(self, latent_transformer, vocab, args, num_att_layers=2):
        super(IQ, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab.word2idx)
        self.latent_transformer = latent_transformer
        self.args = args

        self.embedding = self.embedder()

        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(args)

        # self.latent_layer = Latent(args)
        # self.latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)

        self.decoder = GVTransformerDecoder(
            self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        # Setup image reconstruction.
        self.image_reconstructor = MLP(
            args.hidden_dim, args.pwffn_dim, args.hidden_dim,
            num_layers=num_att_layers)
