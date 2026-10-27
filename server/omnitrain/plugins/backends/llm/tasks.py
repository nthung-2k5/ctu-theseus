"""What the language-model backend trains and the defaults it trains with (pure data, no torch)."""

# Free text in, free text out: the tasks whose snapshot has inline-text input columns and one inline-text target.
LLM_TASKS = frozenset({"text_generation", "summarization", "sequence_to_sequence", "question_answering"})

# LoRA trains a few small matrices beside the frozen weights, so a high learning rate is the norm (full fine-tuning
# would use 1e-5), and an epoch is cheap enough to run a few.
DEFAULTS = {
    "epochs": 3,
    "batch": 4,
    "learning_rate": 2e-4,
    "patience": 2,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    # The longest prompt plus answer, in tokens. A row that is longer loses the start of its prompt, never its answer.
    "max_length": 512,
    # The longest answer, in tokens: training cuts a longer one, and serving stops there.
    "max_new_tokens": 128,
}
BATCH_SIZES = (1, 2, 4, 8, 16)
LORA_RANKS = (4, 8, 16, 32, 64)
OPTIMIZER = "AdamW"

# Rows the evaluation generates an answer for. Generating is token by token, so a large test split would take hours;
# the report says how many rows it used.
EVAL_MAX_ROWS = 200
# Rows scored per generate() call.
GENERATE_BATCH = 8
