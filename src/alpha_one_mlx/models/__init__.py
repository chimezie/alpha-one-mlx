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

QWEN_ADDITIONAL_STOP_WORDS = ["<|im_end|>"]
QWEN_WAIT_WORDS = ["Wait", "But", "Alternatively"]
QWEN_SLOW_THINKING_SUPPRESSION_WORDS = ["\nWait", "\nBut", "\nAlternatively"]

def get_configuration(model_type: str) -> AlphaOneConfiguration:
    if model_type == "qwen3":
        return AlphaOneConfiguration(additional_stop_words=QWEN_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWEN_WAIT_WORDS,
                                     slow_thinking_suppression_phrases=QWEN_SLOW_THINKING_SUPPRESSION_WORDS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")