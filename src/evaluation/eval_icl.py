import random
from typing import List, Dict, Any, Callable
from transformers import pipeline
from tqdm import tqdm
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)

from src.executor import ClevrExecutor
from src.utils import program_utils
from src.utils.logger import log

# Define a smoothing function for BLEU score calculation
# This helps with short sentences that might have 0-grams
BLEU_SMOOTHER = SmoothingFunction().method1


def _format_example(question_data: Dict[str, Any]) -> str:
    """
    Formats a single question-program pair for the few-shot prompt.
    Converts the program list to a prefix string.
    """
    question_text = question_data["question"]
    
    # Convert program list to prefix notation, then to string
    program_list = question_data["program"]
    program_prefix_list = program_utils.list_to_prefix(program_list)
    program_str = program_utils.list_to_str(program_prefix_list)
    
    # Add <START> and <END> tokens as expected by the models
    formatted_program = f"<START> {program_str} <END>"
    
    return f"Question: {question_text}\nProgram: {formatted_program}"


def get_few_shot_examples(
    train_questions: List[Dict[str, Any]], num_examples: int
) -> str:
    """
    Selects random examples from the training data to build the few-shot context.
    """
    if num_examples == 0:
        return ""
        
    # Ensure we don't try to sample more than we have
    num_to_sample = min(num_examples, len(train_questions))
    
    # Randomly select examples
    try:
        selected_examples = random.sample(train_questions, num_to_sample)
    except ValueError as e:
        log.error(f"Error sampling examples: {e}")
        return ""

    # Format each example and join them
    example_strings = [_format_example(ex) for ex in selected_examples]
    return "\n\n".join(example_strings)


def generate_program_with_llm(
    pipe: Callable, system_prompt: str, user_question: str
) -> str:
    """
    Queries the LLM pipeline with the formatted system and user prompts.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    try:
        outputs = pipe(
            messages,
            max_new_tokens=128,  # Limit output length
            do_sample=False,  # Use greedy decoding for consistency
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        # Extract the assistant's reply
        reply = outputs[0]["generated_text"][-1]["content"]
        return reply
    except Exception as e:
        log.error(f"Error during LLM inference: {e}")
        return ""


def parse_program_from_llm_output(raw_output: str) -> str:
    """
    Extracts the program string from the LLM's raw output.
    Looks for content between <START> and <END> tokens.
    """
    start_tag = "<START>"
    end_tag = "<END>"
    
    start_index = raw_output.find(start_tag)
    end_index = raw_output.find(end_tag)

    if start_index != -1 and end_index != -1 and end_index > start_index:
        # Extract content between tags
        program_str = raw_output[start_index + len(start_tag) : end_index].strip()
        return program_str
    else:
        # Fallback: try to find "Program:" and take the rest
        fallback_key = "Program:"
        fallback_index = raw_output.rfind(fallback_key)
        if fallback_index != -1:
            program_str = raw_output[fallback_index + len(fallback_key) :].strip()
            # Clean up if it still includes the tags
            program_str = program_str.replace(start_tag, "").replace(end_tag, "").strip()
            return program_str
            
        log.warning(f"Could not parse program from LLM output: {raw_output}")
        return ""


def evaluate_icl_executor(
    pipe: Callable,
    train_questions: List[Dict[str, Any]],
    test_questions: List[Dict[str, Any]],
    executor: ClevrExecutor,
    vocab: dict,
    num_shots_list: List[int],
    num_test_samples: int,
    split: str = "val",
) -> Dict[int, float]:
    """
    Evaluates LLM-generated programs using the symbolic executor.
    """
    log.info("Starting ICL evaluation with executor...")
    base_system_prompt = (
        "You are an AI assistant. You must translate natural language "
        "questions into a structured sequence of program functions. "
        "The program must start with <START> and end with <END>."
    )
    
    results = {}
    
    for num_shots in num_shots_list:
        log.info(f"--- Evaluating with {num_shots} shots ---")
        few_shot_context = get_few_shot_examples(train_questions, num_shots)
        system_prompt = f"{base_system_prompt}\n\n{few_shot_context}".strip()
        
        total_correct = 0
        test_sample_set = test_questions[:num_test_samples]

        for test_q in tqdm(test_sample_set, desc=f"ICL ({num_shots}-shot)"):
            llm_output = generate_program_with_llm(
                pipe, system_prompt, test_q["question"]
            )
            program_str = parse_program_from_llm_output(llm_output)

            # Tokenize and encode the string program
            tokens = program_str.split()
            token_indices = [
                vocab["program_token_to_idx"].get(
                    t, vocab["program_token_to_idx"]["<UNK>"]
                )
                for t in tokens
            ]
            
            # Get ground truth answer
            gt_answer_str = test_q["answer"]
            
            # Run executor
            pred_answer_str = executor.run(
                token_indices, test_q["image_index"], split=split
            )
            
            if pred_answer_str == gt_answer_str:
                total_correct += 1

        accuracy = total_correct / num_test_samples
        log.info(f"Accuracy ({num_shots}-shot): {accuracy*100:.2f}%")
        results[num_shots] = accuracy
        
    return results


def evaluate_icl_bleu(
    pipe: Callable,
    train_questions: List[Dict[str, Any]],
    test_questions: List[Dict[str, Any]],
    num_shots_list: List[int],
    num_test_samples: int,
) -> Dict[int, float]:
    """
    Evaluates LLM-generated programs using BLEU score against the
    ground truth program.
    """
    log.info("Starting ICL evaluation with BLEU score...")
    base_system_prompt = (
        "You are an AI assistant. You must translate natural language "
        "questions into a structured sequence of program functions. "
        "The program must start with <START> and end with <END>."
    )
    
    results = {}

    for num_shots in num_shots_list:
        log.info(f"--- Evaluating with {num_shots} shots ---")
        few_shot_context = get_few_shot_examples(train_questions, num_shots)
        system_prompt = f"{base_system_prompt}\n\n{few_shot_context}".strip()
        
        total_bleu_score = 0.0
        test_sample_set = test_questions[:num_test_samples]

        for test_q in tqdm(test_sample_set, desc=f"ICL BLEU ({num_shots}-shot)"):
            llm_output = generate_program_with_llm(
                pipe, system_prompt, test_q["question"]
            )
            
            # Get candidate (predicted) program
            candidate_str = parse_program_from_llm_output(llm_output)
            candidate_tokens = candidate_str.split()
            
            # Get reference (ground truth) program
            gt_program_list = program_utils.list_to_prefix(test_q["program"])
            gt_program_str = program_utils.list_to_str(gt_program_list)
            reference_tokens = [gt_program_str.split()]  # BLEU expects list of lists

            # Calculate sentence BLEU score
            score = sentence_bleu(
                reference_tokens, candidate_tokens, smoothing_function=BLEU_SMOOTHER
            )
            total_bleu_score += score
        
        avg_bleu = total_bleu_score / num_test_samples
        log.info(f"Average BLEU ({num_shots}-shot): {avg_bleu:.4f}")
        results[num_shots] = avg_bleu
        
    return results
