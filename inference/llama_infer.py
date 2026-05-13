import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
from peft import PeftModel


class LlamaInference:
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        tokenizer_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.tokenizer_name = tokenizer_name or model_name
        self.adapter_path = adapter_path

        print(f"[DEBUG] adapter_path={self.adapter_path}", flush=True)
        print(f"[DEBUG] model_name={self.model_name}", flush=True)
        print(f"[DEBUG] tokenizer_name={self.tokenizer_name}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            token=self.hf_token,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if torch_dtype == "auto":
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.dtype = dtype

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if self.adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
            )
        else:
            self.model = base_model

        self.model.eval()

    def _get_input_device(self) -> torch.device:
        """
        Safer than relying on self.model.device directly.
        Works better with PEFT and device_map='auto'.
        """
        return next(self.model.parameters()).device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]

        completion = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return {
            "completion": completion,
            "full_text": full_text,
        }

        
