import re
from typing import Optional

def extract_tag_content(text: str, tag_name: str) -> Optional[str]:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
    
def pretty_print_result(sample, prompt_data, generation_result):
    print("=" * 100)
    print("SAMPLE INFO")
    print("-" * 100)
    print(f"difficulty_n : {sample['difficulty_n']}")
    print(f"split        : {sample['split']}")
    print(f"sample_idx   : {sample['sample_idx']}")
    print(f"target       : {sample['target']}")
    print(f"numbers      : {sample['numbers']}")

    print("\n" + "=" * 100)
    print("PROMPT")
    print("-" * 100)
    print(prompt_data["prompt"])

    print("\n" + "=" * 100)
    print("MODEL COMPLETION")
    print("-" * 100)
    print(generation_result["completion"])

    think_text = extract_tag_content(generation_result["completion"], "think")
    answer_text = extract_tag_content(generation_result["completion"], "answer")

    print("\n" + "=" * 100)
    print("PARSED OUTPUT")
    print("-" * 100)
    print(f"think  : {think_text}")
    print(f"answer : {answer_text}")
    print("=" * 100)
    
    