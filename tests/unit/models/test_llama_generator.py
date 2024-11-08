from llama_benchmarks.models import llama
from llama_benchmarks.models.llama import LlamaGenerator


def test_config():
    #
    # Whens
    #

    # I created a Llama 3.2 3B config
    config = llama.config("Llama3.2-3B")

    #
    # Thens
    #

    # d_model should be 3072
    assert config.d_model == 3072

    # vocab_size should be 128256
    assert config.vocab_size == 128256


def test_llama_generator():
    #
    # Givens
    #

    # Prompt
    prompt = "alpha beta gamma"

    # I created a Llama 3.2 3B generator
    generator = LlamaGenerator(llama.config("Llama3.2-3B"))

    #
    # Whens
    #

    # I start token generation
    it = generator(prompt)

    # I collect next token
    prompt += next(it)

    #
    # Thens
    #

    # prompt should be "alpha beta gamma delta"
    assert prompt == "alpha beta gamma delta"
