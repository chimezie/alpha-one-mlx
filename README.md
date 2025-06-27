# Alpha One MLX
An implementation via [mlx-lm](https://github.com/ml-explore/mlx-lm) of [AlphaOne (&alpha;)](https://alphaone-project.github.io/), a universal framework for modulating 
reasoning progress in large reasoning models (LRMs) at test time, reconstituted as a framework for modulating LRM reasoning for general inference.

# Installation
Install via pip from Git

# Usage
```commandline
% alpha_one_mlx_reasoner --help
Usage: alpha_one_mlx_reasoner [OPTIONS]

Options:
  --baseline / --no-baseline
  --verbose / --no-verbose
  --generation-crawl / --no-generation-crawl
  --thinking-token-length INTEGER
                                  Average thinking phase token length
                                  (defaults to 2,650)
  --max-tokens INTEGER            Maximum tokens to generate (defaults to
                                  8,192)
  --alpha FLOAT                   Universal modulating parameter for scaling
                                  the thinking phase (defaults to 1.4 per
                                  paper)
  --temp FLOAT                    The temperature (defaults to 1)
  --query TEXT                    The user question
  --model TEXT                    The model to use
  --eos TEXT                      Additional EOS words (0 or more) to add to
                                  tokenizer
  --wait-words TEXT               Words (0 or more) to for slow-thinking
                                  modulation
  --help                          Show this message and exit.
```

# Model types
The following model types are supported

## Qwen3
For these models, the step of activating slow thinking during the pre-alpha moment modulation is done by keeping the 
model's probability of producing the tokens for any of the following words and suppressing all others:
- _"Wait"_
- _"But"_
- _"Alternatively"_

Any occurrence of these tokens during post-alpha moment modulation will stop the generation process.  This list of words 
can be overridden by specifying your own using the `--wait-words` option.

In addition, any occurrence of these words (following a newline) during post-alpha moment modulation will cause the 
process to force a transition to *fast thinking* by producing the _"&lt;/think>"_ token.