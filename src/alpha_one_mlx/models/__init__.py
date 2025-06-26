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
    slow_thinking_transition_phrases: list[str] = field(metadata={"help": "Trigger word to replace with </think>"})

QWQ_ADDITIONAL_STOP_WORDS = ["<|endoftext|>"]
QWQ_WAIT_WORDS = [
    "Wait",
    ".Wait",
    "Wait, ",
    # "Wait,"
]
QWQ_CORE_WAIT_TRIGGER_WORDS = ["\nWait", "Wait"]

QWEN_ADDITIONAL_STOP_WORDS = ["<|im_end|>"]
QWEN_WAIT_WORDS = ["Wait", "Wait, ", "But", "But ", "Alternatively", "Alternatively, "]
QWEN_CORE_WAIT_TRIGGER_WORDS = ["\nWait", "\nBut", "\nAlternatively",
                                "Wait", "But", "Alternatively"]

def get_configuration(model_type: str) -> AlphaOneConfiguration:
    if model_type == "qwq":
        return AlphaOneConfiguration(additional_stop_words=QWQ_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWQ_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_transition_phrases=QWQ_CORE_WAIT_TRIGGER_WORDS)
    elif model_type == "qwen3":
        return AlphaOneConfiguration(additional_stop_words=QWEN_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWEN_WAIT_WORDS,
                                     slow_thinking_transition_phrases=QWEN_CORE_WAIT_TRIGGER_WORDS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")