import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Any

from src.utils.logger import log
from src.vocabulary import load_vocab


class ClevrQuestionDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed CLEVR questions from an H5 file.
    """

    def __init__(self, question_h5_path: str, vocab_json_path: str, max_samples: Optional[int] = None):
        """
        Args:
            question_h5_path (str): Path to the H5 file.
            vocab_json_path (str): Path to the vocabulary JSON file.
            max_samples (int, optional): Max number of samples to load.
        """
        self.question_h5_path = Path(question_h5_path)
        self.vocab_json_path = Path(vocab_json_path)
        self.max_samples = max_samples

        log.info(f"Loading dataset from: {self.question_h5_path}")
        try:
            with h5py.File(self.question_h5_path, 'r') as question_h5:
                # Load questions and image indices
                self.questions = torch.LongTensor(
                    np.asarray(question_h5['questions'], dtype=np.int64)
                )
                self.image_idxs = torch.LongTensor(
                    np.asarray(question_h5['image_idxs'], dtype=np.int64)
                )
                
                # Load programs if they exist
                if 'programs' in question_h5:
                    self.programs = torch.LongTensor(
                        np.asarray(question_h5['programs'], dtype=np.int64)
                    )
                else:
                    self.programs = None
                    log.warning("No 'programs' found in H5 file.")

                # Load answers if they exist
                if 'answers' in question_h5:
                    self.answers = torch.LongTensor(
                        np.asarray(question_h5['answers'], dtype=np.int64)
                    )
                else:
                    self.answers = None
                    log.warning("No 'answers' found in H5 file.")

        except (IOError, OSError, KeyError) as e:
            log.error(f"Failed to load H5 file: {e}")
            raise

        # Load vocabulary
        self.vocab = load_vocab(self.vocab_json_path)
        self.dataset_size = len(self.questions)

        if self.max_samples:
            self.dataset_size = min(self.max_samples, self.dataset_size)
            log.info(f"Using a subset of {self.dataset_size} samples.")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - question: The tokenized question tensor.
            - program: The tokenized program tensor (or -1 if not present).
            - answer: The answer index (or -1 if not present).
            - image_idx: The index of the corresponding image/scene.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self)})")

        question = self.questions[idx]
        image_idx = self.image_idxs[idx]
        
        program = -1
        if self.programs is not None:
            program = self.programs[idx]
            
        answer = -1
        if self.answers is not None:
            answer = self.answers[idx]

        return question, program, answer, image_idx


def get_dataloader(
    h5_path: str,
    vocab_json_path: str,
    batch_size: int,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the CLEVR dataset.

    Args:
        h5_path (str): Path to the H5 file.
        vocab_json_path (str): Path to the vocabulary JSON file.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.
        max_samples (int, optional): Max number of samples to use.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    log.info(f"Creating DataLoader for {h5_path}...")
    dataset = ClevrQuestionDataset(
        question_h5_path=h5_path,
        vocab_json_path=vocab_json_path,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    log.info(f"DataLoader created with {len(dataset)} samples.")
    return dataloader
