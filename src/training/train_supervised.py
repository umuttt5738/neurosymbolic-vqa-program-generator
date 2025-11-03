import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.utils.logger import log
from src.evaluation.eval_model import evaluate_model


class TrainerSupervised:
    """
    Handles the supervised training loop for a Seq2Seq model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        executor,
        vocab,
        device,
        learning_rate=1e-4,
        num_iters=100000,
        log_interval=100,
        val_interval=1000,
        model_save_path="models/supervised_model.pth",
    ):
        """
        Initializes the Supervised Trainer.

        Args:
            model (nn.Module): The Seq2Seq model (LSTM or Transformer).
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            executor (ClevrExecutor): The symbolic program executor.
            vocab (dict): The loaded vocabulary.
            device (torch.device): The device to run training on.
            learning_rate (float): The learning rate for the optimizer.
            num_iters (int): Total number of training iterations.
            log_interval (int): How often to log training loss.
            val_interval (int): How often to run validation.
            model_save_path (str): Path to save the best model.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.executor = executor
        self.vocab = vocab
        self.device = device

        # Hyperparameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.model_save_path = model_save_path

        # Get the padding token index for the loss function
        self.pad_idx = vocab["program_token_to_idx"]["<NULL>"]

        # Loss Function: Cross Entropy Loss, ignoring padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_accuracy = -1.0
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


    def _run_train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()  # Set model to training mode
        loop = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Supervised]")

        for batch in loop:
            if self.global_step >= self.num_iters:
                log.info("Reached target iterations. Stopping training.")
                return False  # Signal to stop training

            # Unpack batch and move to device
            questions, programs, _, _ = batch
            questions = questions.to(self.device)
            programs = programs.to(self.device)

            # Prepare inputs and targets for teacher-forcing
            # Input to decoder: <START>, token1, token2, ...
            decoder_inputs = programs[:, :-1]
            # Target for loss: token1, token2, ..., <END>
            decoder_targets = programs[:, 1:]

            # Forward pass
            self.optimizer.zero_grad()
            # The model should return raw logits
            logits = self.model(questions, decoder_inputs)

            # Calculate loss
            # Reshape for CrossEntropyLoss:
            # Logits: [B, S, V] -> [B*S, V]
            # Targets: [B, S] -> [B*S]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), 
                decoder_targets.reshape(-1)
            )

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            # --- Logging ---
            if self.global_step % self.log_interval == 0:
                # Perplexity is e^(loss)
                perplexity = torch.exp(loss)
                log.info(
                    f"[Step {self.global_step}/{self.num_iters}] "
                    f"Loss: {loss.item():.4f}, "
                    f"Perplexity: {perplexity.item():.2f}"
                )
                loop.set_postfix(loss=loss.item())

            # --- Validation ---
            if self.global_step % self.val_interval == 0:
                self._run_validation()

        return True # Signal to continue training

    def _run_validation(self):
        """Runs validation and saves the model if it's the best one."""
        log.info(f"--- Running validation at step {self.global_step} ---")
        # Use the dedicated evaluation function
        val_accuracy = evaluate_model(
            self.model,
            self.val_loader,
            self.executor,
            self.vocab,
            self.device,
            split="val",
        )
        
        # Log validation accuracy
        log.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")

        if val_accuracy > self.best_val_accuracy:
            log.info(f"New best validation accuracy! Saving model to {self.model_save_path}")
            self.best_val_accuracy = val_accuracy
            torch.save(self.model.state_dict(), self.model_save_path)
        
        # Set model back to training mode
        self.model.train()

    def train(self):
        """Main training entry point."""
        log.info("Starting supervised training...")
        log.info(f"  Device: {self.device}")
        log.info(f"  Num iterations: {self.num_iters}")
        log.info(f"  Model save path: {self.model_save_path}")

        while self.global_step < self.num_iters:
            self.epoch += 1
            log.info(f"--- Starting Epoch {self.epoch} ---")
            
            should_continue = self._run_train_epoch()
            if not should_continue:
                break

        log.info("Supervised training complete.")
        log.info(f"Best validation accuracy: {self.best_val_accuracy*100:.2f}%")
        log.info(f"Best model saved to {self.model_save_path}")
