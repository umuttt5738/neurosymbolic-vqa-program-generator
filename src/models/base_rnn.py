import torch.nn as nn
from typing import Literal

RNN_CELL_TYPES = Literal["lstm", "gru"]


class BaseRNN(nn.Module):
    """
    A base RNN module to be inherited by Encoder and Decoder models.
    It holds common properties and validates the RNN cell type.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_size: int,
        input_dropout_prob: float,
        rnn_dropout_prob: float,
        num_layers: int,
        rnn_cell: RNN_CELL_TYPES = "lstm",
    ):
        """
        Initializes the BaseRNN.

        Args:
            vocab_size (int): The size of the vocabulary.
            max_seq_len (int): Maximum length of a sequence.
            hidden_size (int): The size of the RNN hidden state.
            input_dropout_prob (float): Dropout probability for the input embeddings.
            rnn_dropout_prob (float): Dropout probability for the RNN layers.
            num_layers (int): The number of RNN layers.
            rnn_cell (str): The type of RNN cell to use ('lstm' or 'gru').
        """
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout_prob = input_dropout_prob
        self.rnn_dropout_prob = rnn_dropout_prob

        # Validate and select RNN cell
        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN Cell: {rnn_cell}. Must be 'lstm' or 'gru'.")

        self.input_dropout = nn.Dropout(p=input_dropout_prob)

    def forward(self, *args, **kwargs):
        """
        Forward pass (to be implemented by subclasses).
        """
        raise NotImplementedError("This is a base class. Subclasses must implement forward().")
