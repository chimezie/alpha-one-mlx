import functools
import random
import mlx.core as mx
import click
import warnings
import mlx.nn as nn
from tqdm import tqdm

from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from transformers import PreTrainedTokenizer

from .models import get_configuration, AlphaOneConfiguration
from itertools import chain
from typing import (
    List,
    Optional,
    Union,
    Any
)

NEWLINE_CHARACTERS = ["\n\n", ",\n\n", ".\n\n", "]\n\n", ")\n\n", "],\n\n", "].\n\n", ").\n\n", ".)\n\n",
                      " \n\n", "\n \n", "  \n\n", "   \n\n"]

class NewlineWait:
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
                 wait_words: List[str],
                 max_token_per_call: int = 0,
                 threshold: int = 0,
                 track_progress: bool = False):
        """
        A logits processor that modulates slow thinking by producing "wait" tokens

        """
        #Save new line character, wait, and think tokens
        self.newline_ids = list(chain.from_iterable(tokenizer.batch_encode_plus(NEWLINE_CHARACTERS,
                                                                                add_special_tokens=False).input_ids))
        self.wait_words = wait_words
        self.wait_ids = []
        self.wait_word_lookup = {}
        for word in wait_words:
            try:
                wait_token = tokenize_single_token(word, tokenizer, verbose=track_progress)
                self.wait_ids.append(wait_token)
                self.wait_word_lookup[wait_token] = word
            except AssertionError:
                warnings.warn(f'Bypassing multiple token EOS phrase: "{word}" ...')
                continue
        if not self.wait_words:
            raise RuntimeError("No valid wait words provided")
        self.end_think_id = tokenize_single_token("</think>", tokenizer)
        self.max_token_per_call = max_token_per_call
        self.threshold = threshold
        self.number_of_thinking_phase_tokens = self.max_token_per_call - self.threshold
        self.track_progress = track_progress
        if track_progress:
            self.pbar = tqdm(total=self.number_of_thinking_phase_tokens)
        self._num_tokens_generated = 0
        self.non_wait_token_mask = None
        self.token_ids = None

    def __call__(self, token_ids: mx.array, logits: mx.array) -> mx.array:
        """
        Logits processor that updates logits so a reasoning "Wait" is produced (indicating slow thinking)

        :param token_ids: Tokens generated so far
        :param logits: Logits
        :return: updated or same logits
        """
        num_tokens_generated = len(token_ids)
        if self.track_progress:
            self.pbar.update(num_tokens_generated - self._num_tokens_generated)
            self._num_tokens_generated = num_tokens_generated

        #Generate at least 2 tokens before slow thinking modulation
        if num_tokens_generated < 2:
            return logits

        #Number of tokens left before reaching the maximum allowed
        num_remaining_tokens = self.max_token_per_call - num_tokens_generated
        self.token_ids = token_ids

        if num_remaining_tokens >= self.threshold and token_ids[-1] in self.newline_ids:
            #The number of tokens left to generate is greater than the threshold and a new line token was just produced

            #Proportion of progress towards the end of the thinking phase
            #"slow thinking first, then fast thinking is a better slow thinking scheduling strategy."
            p_wait = ((-1 / self.number_of_thinking_phase_tokens) * num_tokens_generated) + 1

            #Randomly initiate slow thinking with increasing probability along the progress towards the end of the
            # thinking phase
            rand_variable = random.random()
            if rand_variable < p_wait:
                if self.non_wait_token_mask is None:
                    #Create -infinity mask for non-wait tokens only once
                    vocab_size = logits.shape[-1]
                    self.non_wait_token_mask = mx.array([i not in self.wait_ids for i in range(vocab_size)])
                logits = mx.where(self.non_wait_token_mask, mx.array(-mx.inf, dtype=logits.dtype), logits)
                if self.track_progress:
                    wait_token_info = ", ".join([f"{w_id} ({self.wait_word_lookup[w_id]})" for w_id in self.wait_ids])
                    self.pbar.write(f"Boosting slow thinking probability: {wait_token_info} ...")
        return logits

def alpha_one(model: nn.Module,
              tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
              query: str,
              configuration: AlphaOneConfiguration,
              max_tokens_per_call: int = 8192,
              threshold: int = 0,
              temperature: float = 0.6,
              top_p: float = 0.95,
              min_p:float = 0,
              top_k:int = 20,
              seed: int = None,
              apply_chat_template: bool = True,
              verbose: bool = False,
              generation_crawl: bool = False,
              stop_on_empty_post_alpha_generation: bool = True,
              baseline: bool = False,
              eos: Optional[List[str]] = None,
              wait_words: Optional[List[str]] = None,
              prompt_cache: Optional[Any] = None,
              draft_model: nn.Module = None):
    if eos is None:
        eos = []
    if wait_words is None:
        wait_words = []
    if threshold > max_tokens_per_call:
        raise ValueError(f"Threshold is too high (by {threshold - max_tokens_per_call:,}) (increase max tokens or "
                         f"threshold by proportional amount)")
    if seed is not None:
        mx.random.seed(seed)
        random.seed(seed)

    slow_thinking_words = wait_words if wait_words else configuration.slow_thinking_stop_words

    logits_processor = NewlineWait(tokenizer,
                                   slow_thinking_words,
                                   max_token_per_call=max_tokens_per_call,
                                   threshold=threshold,
                                   track_progress=verbose)
    sampler = make_sampler(
        temp=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k
    )
    for stop_word in list(eos):
        try:
            tokenizer.add_eos_token(tokenize_single_token(stop_word, tokenizer))
        except AssertionError:
            warnings.warn(f'Bypassing multiple token EOS phrase: "{stop_word}" ...')
            continue

    if apply_chat_template:
        query = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True
        )

    initial_prompt = prepare_prompt(query, tokenizer)

    output_response = ''

    generated_text = ''
    generated_tokens = []
    thinking_phase_length = 0
    if verbose:
        print("Pre alpha-moment modulation: ")
    for idx, response in enumerate(stream_generate(
            model,
            tokenizer,
            initial_prompt,
            max_tokens=max_tokens_per_call - threshold,
            sampler=sampler,
            logits_processors=[logits_processor] if not baseline else [],
            prompt_cache=prompt_cache,
            draft_model=draft_model)):
        generated_text += response.text
        generated_tokens.append(response.token)
        avg_thinking_token_length = int(max_tokens_per_call - threshold) / 1.4
        if response.token == logits_processor.end_think_id:
            if baseline:
                thinking_phase_length = idx + 1
            else:
                raise Exception(f"Exited thinking phase prematurely (after {idx+1:,} tokens).  Update Average thinking "
                                f"phase token length accordingly.  "
                                f"Threshold is {threshold:,}, max tokens is {max_tokens_per_call:,}, "
                                f"average thinking token length is {avg_thinking_token_length:,}, and the scaled "
                                f"thinking token length is {1.4 * avg_thinking_token_length:,} ")

    if verbose:
        logits_processor.pbar.close()
        print(response.finish_reason)
    if baseline:
        if verbose:
            print(f"prompts: {query}\n=====================")
            print(f"Thinking phase length: {thinking_phase_length:,} tokens")
        return generated_text

    output_response += generated_text
    modulated_query = query + generated_text
    remaining_tokens_ = max_tokens_per_call - len(generated_tokens)

    #Post alpha-moment modulation (disabling further slow thinking)
    #"After the α moment, we guide [alpha one] to transition into fast reasoning by disabling further slow thinking."
    for stop_word in slow_thinking_words:
        try:
            tokenizer.add_eos_token(tokenize_single_token(stop_word, tokenizer))
        except AssertionError:
            warnings.warn(f'Bypassing multiple token EOS phrase: "{stop_word}" ...')
            continue

    active_traj = True
    if verbose:
        print("Starting post-alpha moment modulation")

    thinking_ended = False
    while True:
        prompt = prepare_prompt(modulated_query, tokenizer)
        generated_text = ''
        generated_tokens = []
        for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=remaining_tokens_,
                sampler=sampler,
                prompt_cache=prompt_cache,
                draft_model=draft_model
        ):
            generated_text += response.text
            generated_tokens.append(response.token)
            if response.token == logits_processor.end_think_id:
                thinking_ended = True
            if generation_crawl:
                print(response.text, flush=True, end="")
        if generation_crawl:
            print()

        if not thinking_ended:
            for phrase in configuration.slow_thinking_suppression_phrases:
                if phrase in generated_text:
                    warnings.warn(f"Slow thinking phrase ({phrase}) found in post-alpha moment modulation!")

            #" [..] any generated slow reasoning transition token “wait” is replaced with “</think>” to explicitly
            # mark the end of the thinking phase, reinforcing a shift to fast thinking before entering the answering phase
            generated_text = functools.reduce(lambda text, phrase: text.replace(f"{phrase}", "</think>"),
                                              configuration.slow_thinking_suppression_phrases, generated_text)
        if active_traj:
            output_response += generated_text
            modulated_query += generated_text

            remaining_tokens_ = max(1, remaining_tokens_ - len(generated_tokens))
            if (response.finish_reason == "stop" or
                remaining_tokens_ == 1 or
                (stop_on_empty_post_alpha_generation and generated_text == "")):
                if response.token in tokenizer.eos_token_ids:
                    if verbose:
                        warnings.warn("Generated text ends with stop word")
                elif remaining_tokens_ == 1:
                    if verbose:
                        warnings.warn("Only 1 token left")
                else:
                    if verbose:
                        warnings.warn("Generated text is empty")
                active_traj = False

        if not active_traj:
            warnings.warn("Exiting alpha one modulation")
            break
    return output_response if not generation_crawl else None


def prepare_prompt(query: str, tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper]):
    add_special_tokens = tokenizer.bos_token is None or not query.startswith(tokenizer.bos_token)
    initial_prompt = tokenizer.encode(query, add_special_tokens=add_special_tokens)
    return initial_prompt

def tokenize_single_token(word: str, tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper], verbose: bool = False):
    tokens = tokenizer.encode(word, add_special_tokens=False)
    assert len(tokens) == 1, f"'{word}' -> {tokens}"
    if verbose:
        print(f"'{word}' -> {tokens}")
    return tokens[0]

@click.command()
@click.option('--baseline/--no-baseline', default=False)
@click.option('--verbose/--no-verbose', default=False)
@click.option('--generation-crawl/--no-generation-crawl', default=False)
@click.option('--thinking-token-length', type=int, default=2650,
              help='Average thinking phase token length (defaults to 2650)')
@click.option('--max-tokens', type=int, default=8192, help='Maximum tokens to generate (defaults to 8192)')
@click.option('--alpha', type=float, default=1.4, help='Universal modulating parameter for scaling the '
                                                       'thinking phase (defaults to 1.4 per paper)')
@click.option('--temp', type=float, default=1.0, help='The temperature (defaults to 1)')
@click.option('--query', help='The user question')
@click.option('--model', help='The model to use', default="mlx-community/QwQ-32b-4bit-DWQ")
@click.option('--eos', help='Additional EOS words (0 or more) to add to tokenizer', multiple=True)
@click.option('--wait-words', help='Words (0 or more) to for slow-thinking modulation', multiple=True)
@click.option('--draft-model', help='Draft model (for speculative decoding)')
def main(baseline,
         verbose,
         generation_crawl,
         thinking_token_length,
         max_tokens,
         alpha,
         temp,
         query,
         model,
         eos,
         wait_words,
         draft_model):
    from mlx_lm.utils import load
    model, tokenizer = load(model)
    configuration = get_configuration(model.model_type)
    threshold = int(max_tokens - alpha * thinking_token_length)

    if draft_model is not None:
        draft_model, draft_tokenizer = load(draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    print(alpha_one(model,
                    tokenizer,
                    query,
                    configuration=configuration,
                    max_tokens_per_call=max_tokens,
                    threshold=threshold,
                    temperature=temp,
                    top_p=0.95,
                    min_p=0,
                    top_k=20,
                    seed=42,
                    apply_chat_template=True,
                    verbose=verbose,
                    generation_crawl=generation_crawl,
                    baseline=baseline,
                    eos=eos,
                    wait_words=wait_words,
                    draft_model=draft_model))
if __name__ == '__main__':
    main()
