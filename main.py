from config import (
    MODEL_NAME,
    DATASET_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    DEVICE_MAP,
    TORCH_DTYPE,
    HF_TOKEN,
)
from dataset.loader import load_dataset
from prompting.prompt_builder import CountdownPromptBuilder
from inference.llama_infer import LlamaInference
from utils.parsing import pretty_print_result


def main():
    # 1. Load dataset
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        raise ValueError("Dataset is empty.")

    # For now, use one dataset sample exactly as you requested
    sample = dataset[0]

    # 2. Convert raw row into prompt-ready example
    prompt_builder = CountdownPromptBuilder()
    prompt_example = prompt_builder.build_example_for_prompt(sample)

    # 3. Load model/tokenizer
    infer_engine = LlamaInference(
        model_name=MODEL_NAME,
        hf_token=HF_TOKEN,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
    )

    # 4. Build prompt using tokenizer chat template
    prompt_data = prompt_builder.generate_prompt(infer_engine.tokenizer, prompt_example)

    # 5. Generate model completion
    generation_result = infer_engine.generate(
        prompt=prompt_data["prompt"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
    )

    # 6. Show everything clearly
    pretty_print_result(sample, prompt_data, generation_result)


if __name__ == "__main__":
    main()