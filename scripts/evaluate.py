"""
Script for evaluating a trained model on the validation or test set.

This script loads a trained model checkpoint, runs it on the specified dataset,
and prints the final program execution accuracy.

Example:
python scripts/evaluate.py \
    --model_type lstm \
    --model_path models/supervised_lstm.pth \
    --data_h5_path data/dataH5Files/clevr_val_questions.h5 \
    --split val
"""

import argparse
import torch
from pathlib import Path

import src.config as config
from src.utils.logger import setup_logger, log
from src.vocabulary import load_vocab
from src.executor import ClevrExecutor
from src.data_loader import get_dataloader
from src.models import LstmSeq2Seq, TransformerSeq2Seq
from src.evaluation.eval_model import evaluate_model


def get_model(args, vocab):
    """Instantiates and returns the correct model based on args."""
    log.info(f"Initializing model: {args.model_type}")
    
    if args.model_type == 'lstm':
        model = LstmSeq2Seq(
            vocab=vocab,
            word_vec_dim=config.LSTM_WORD_VEC_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            input_dropout_prob=config.LSTM_INPUT_DROPOUT,
            rnn_dropout_prob=config.LSTM_RNN_DROPOUT,
            bidirectional_encoder=True,
            use_attention=True
        )
    elif args.model_type == 'transformer':
        model = TransformerSeq2Seq(
            vocab=vocab,
            d_model=config.TR_D_MODEL,
            nhead=config.TR_NHEAD,
            num_encoder_layers=config.TR_NUM_ENCODER_LAYERS,
            num_decoder_layers=config.TR_NUM_DECODER_LAYERS,
            dim_feedforward=config.TR_DIM_FEEDFORWARD,
            dropout=config.TR_DROPOUT,
            max_seq_len=max(config.MAX_QUESTION_LEN, config.MAX_PROGRAM_LEN)
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
    return model


def main(args):
    # --- Setup ---
    setup_logger(config.LOG_FILE)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available. Falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # --- Load Data and Utils ---
    log.info(f"Loading vocab, executor, and data loader for split: {args.split}...")
    vocab = load_vocab(config.VOCAB_JSON_FILE)
    
    # Select the correct scene JSON based on the split
    scene_json = config.TRAIN_SCENES_JSON if args.split == 'train' else \
                 config.VAL_SCENES_JSON if args.split == 'val' else \
                 config.TEST_SCENES_JSON
    
    if not scene_json.exists():
        log.error(f"Scene JSON file not found for split '{args.split}': {scene_json}")
        # Use val scenes as a fallback if test scenes are missing
        if args.split == 'test' and config.VAL_SCENES_JSON.exists():
            log.warning(f"Using val scenes as fallback: {config.VAL_SCENES_JSON}")
            scene_json = config.VAL_SCENES_JSON
        else:
            raise FileNotFoundError(f"No scene data found for split: {args.split}")

    executor = ClevrExecutor(
        train_scene_json=config.TRAIN_SCENES_JSON,
        val_scene_json=scene_json,  # Use the selected scene file for 'val' split in executor
        vocab_json=config.VOCAB_JSON_FILE
    )
    
    eval_loader = get_dataloader(
        h5_path=args.data_h5_path,
        vocab_json_path=config.VOCAB_JSON_FILE,
        batch_size=args.batch_size,
        shuffle=False  # No shuffling for evaluation
    )

    # --- Initialize Model ---
    model = get_model(args, vocab)
    model.to(device)
    
    # --- Load Weights ---
    if not args.model_path.exists():
        log.error(f"Model checkpoint not found: {args.model_path}")
        raise FileNotFoundError(f"File not found: {args.model_path}")
        
    log.info(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # --- Run Evaluation ---
    log.info(f"Starting evaluation on {args.data_h5_path}...")
    
    model.eval()
    with torch.no_grad():
        accuracy = evaluate_model(
            model=model,
            data_loader=eval_loader,
            executor=executor,
            vocab=vocab,
            device=device,
            split=args.split  # Pass the correct split to the executor
        )
        
    log.info("--- Evaluation Complete ---")
    log.info(f"  Model:       {args.model_path}")
    log.info(f"  Data Split:  {args.data_h5_path} ({args.split})")
    log.info(f"  Accuracy:    {accuracy * 100:.2f}%")
    log.info("-----------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VQA model.")
    
    # --- Required Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'],
                        help="The type of model architecture to evaluate.")
    parser.add_argument('--model_path', type=Path, required=True,
                        help="Path to the saved .pth model checkpoint.")
    parser.add_argument('--data_h5_path', type=Path, required=True,
                        help="Path to the H5 data file to evaluate (e.g., clevr_val_questions.h5).")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help="The data split to evaluate (must match the H5 file).")
                        
    # --- Optional Arguments ---
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help="Batch size for evaluation.")
    parser.add_argument('--device', type=str, default=config.DEVICE, choices=['cuda', 'cpu'],
                        help="Device to run evaluation on.")
    
    args = parser.parse_args()
    main(args)
