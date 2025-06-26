from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

@dataclass
class AlphaOneConfiguration:
    additional_stop_words: list[str] = field(metadata={"help": "Additional stop words"})
    slow_thinking_stop_words: list[str] = field(metadata={"help": "Trigger words for disabling slow thinking"})
    slow_thinking_suppression_phrases: list[str] = field(metadata={"help": "Trigger word to replace with </think>"})

QWQ_ADDITIONAL_STOP_WORDS = ["<|endoftext|>"]
QWQ_WAIT_WORDS = [
    "Wait",
    ".Wait",
    "Wait, ",
    # "Wait,"
]
QWQ_SLOW_THINKING_SUPPRESSION_WORDS = ["\nWait"]

QWEN_ADDITIONAL_STOP_WORDS = ["<|im_end|>"]
QWEN_WAIT_WORDS = ["Wait", "Wait, ", "But", "But ", "Alternatively", "Alternatively, "]
QWEN_SLOW_THINKING_SUPPRESSION_WORDS = ["\nWait", "\nBut", "\nAlternatively"]

def get_configuration(model_type: str) -> AlphaOneConfiguration:
    if model_type == "qwq":
        return AlphaOneConfiguration(additional_stop_words=QWQ_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWQ_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_suppression_phrases=QWQ_SLOW_THINKING_SUPPRESSION_WORDS)
    elif model_type == "qwen3":
        return AlphaOneConfiguration(additional_stop_words=QWEN_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWEN_WAIT_WORDS,
                                     slow_thinking_suppression_phrases=QWEN_SLOW_THINKING_SUPPRESSION_WORDS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")