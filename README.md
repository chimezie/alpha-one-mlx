# Alpha One MLX
An implementation of [AlphaOne (&alpha;)](https://alphaone-project.github.io/), a universal framework for modulating 
reasoning progress in large reasoning models (LRMs) at test time, reconstituted as a framework for modulating 
LRM reasoning for general inference and implemented in [mlx-lm](https://github.com/ml-explore/mlx-lm).

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
                                  (defaults to 2650)
  --max-tokens INTEGER            Maximium tokens to generate (defaults to
                                  8192)
  --temp FLOAT                    The temperature (defaults to 1)
  --query TEXT                    The user question
  --model TEXT                    The model to use
  --help                          Show this message and exit.
```