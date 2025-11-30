import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Tuple,
    TypedDict,
    Union
)

from mlx_lm.tokenizer_utils import TokenizerWrapper

from transformers import PreTrainedTokenizer

#An LLM message which maps roles to 'assistant' or 'user' and 'content' to the value of the message
MessageInfo = TypedDict('MessageInfo', {'role': str, 'content': str})

class AbstractThinkingTemplateParser(ABC):
    @abstractmethod
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
                 thinking_template: Optional[re.Pattern] = None):
        self.tokenizer = tokenizer
        self.thinking_template = thinking_template

    def num_tokens(self, content: str) -> List[int]:
        """
        Encodes a given string into a list of integer tokens using the tokenizer. This method does
        not add special tokens during the encoding process and returns the encoded integer list.

        :param tokenizer: The tokenizer to use for encoding.
        :param content: The input string to encode.
        :type content: str
        :return: A list of integers representing the encoded tokens of the input string.
        :rtype: List[int]
        """
        return self.tokenizer.encode(content, add_special_tokens=False)

    @abstractmethod
    def break_llm_response_parts(self, llm_response: str) -> Optional[Tuple[str, str]]:
        """
        Extract thoughts and response from format LLM response.

        Args:
            llm_response: Raw response string from LLM in Qwen3 format

        Returns:
            Tuple of (thoughts, response) if format matches, else None
        """
        return None

QWEN3_THINKING_TEMPLATE = re.compile(r'\s*<think>(?P<thoughts>.+)</think>(?P<response>.+)$', re.DOTALL)

class Qwen3ThinkingTemplateParser(AbstractThinkingTemplateParser):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
                 thinking_template: Optional[re.Pattern] = None):
        # Use the provided template or default to QWEN3 template
        template = thinking_template if thinking_template is not None else QWEN3_THINKING_TEMPLATE
        super().__init__(tokenizer, template)

    def break_llm_response_parts(self, llm_response: str) -> Optional[Tuple[str, str]]:
        match = self.thinking_template.match(llm_response)
        if match:
            thoughts = match.group('thoughts').strip()
            response = match.group('response').strip()
            return thoughts, response
        else:
            return None

QWEN_ADDITIONAL_STOP_WORDS = ["<|im_end|>"]
QWEN_WAIT_WORDS = ["Wait", "But", "Alternatively"]
QWEN_SLOW_THINKING_SUPPRESSION_WORDS = ["\nWait", "\nBut", "\nAlternatively"]

@dataclass
class AlphaOneConfiguration:
    additional_stop_words: list[str] = field(metadata={"help": "Additional stop words"})
    slow_thinking_stop_words: list[str] = field(metadata={"help": "Trigger words for disabling slow thinking"})
    slow_thinking_suppression_phrases: list[str] = field(metadata={"help": "Trigger word to replace with </think>"})

def get_configuration(model_type: str) -> AlphaOneConfiguration:
    if model_type == "qwen3":
        return AlphaOneConfiguration(additional_stop_words=QWEN_ADDITIONAL_STOP_WORDS,
                                     slow_thinking_stop_words=QWEN_WAIT_WORDS,
                                     slow_thinking_suppression_phrases=QWEN_SLOW_THINKING_SUPPRESSION_WORDS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_thinking_template_parser(model_type: str,
                                 tokenizer: Union[PreTrainedTokenizer,
                                 TokenizerWrapper]) -> AbstractThinkingTemplateParser:
    if model_type == "qwen3":
        return Qwen3ThinkingTemplateParser(tokenizer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def average_thinking_tokens(parser: AbstractThinkingTemplateParser,
                            messages: List[MessageInfo]) -> float:
    """
    Calculates the average number of thinking tokens for a given sequence of messages
    using the specified thinking template parser.

    This function computes the average via the template parser
    to determine token usage for each message and then aggregates the results.

    :param parser: An implementation of a thinking template parser that processes
        messages separates the thinking part from the rest of the response
    :type parser: AbstractThinkingTemplateParser
    :param messages: A list of MessageInfo objects for which the token calculations
        will be computed.
    :type messages: List[MessageInfo]
    :return: The average number of thinking tokens per message in the provided
        sequence. If no messages are provided, the result is 0.
    :rtype: float
    """
    thinking_token_count = []

    for message in messages:
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            parsed_result = parser.break_llm_response_parts(content)
            if parsed_result is not None:
                thoughts, _ = parsed_result
                thinking_token_count.append(len(parser.num_tokens(thoughts)))
    if not thinking_token_count:
        return 0.0
    return sum(thinking_token_count) / len(thinking_token_count)
