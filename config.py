MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


TRAIN_DATASET_PATHS = [
    "../data/countdown_n2_train.parquet",
    "/mmfs1/scratch/jacks.local/pkhanal2568/yuv_workshop_newnew_3b/local_data/countdown_n3_train.parquet",
    "/mmfs1/scratch/jacks.local/pkhanal2568/yuv_workshop_newnew_3b/local_data/countdown_n4_train.parquet",
    "/mmfs1/scratch/jacks.local/pkhanal2568/yuv_workshop_newnew_3b/local_data/countdown_n5_train.parquet",
]


PROBE_DATASET_PATHS = {
    "n2": "../data/countdown_n2_probe.parquet",
    "n3": "../data/countdown_n3_probe.parquet",
    "n4": "../data/countdown_n4_probe.parquet",
    "n5": "../data/countdown_n5_probe.parquet",
}

EVAL_DATASET_PATHS = {
    "n2": "../data/countdown_n2_holdout.parquet",
    "n3": "../data/countdown_n3_holdout.parquet",
    "n4": "../data/countdown_n4_holdout.parquet",
    "n5": "../data/countdown_n5_holdout.parquet",
}


MAX_NEW_TOKENS = 192
TEMPERATURE = 0.7
TOP_P = 1.0
DO_SAMPLE = True
DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"

HF_TOKEN = ""


USE_LORA =True
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

