import llama_benchmarks as llb
from llama_benchmarks.models import llama
from llama_benchmarks.models.llama import LlamaModel



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


def test_llama_model():
    #
    # Givens
    #

    # Prompt
    prompt = "alpha beta gamma"

    # I created a Llama 3.2 3B LlamaModel
    model = LlamaModel(llama.config("Llama3.2-3B"))

    #
    # Whens
    #

    # I start token generation
    it = model(prompt)

    # I collect next token
    token_id = next(it)

    #
    # Thens
    #

    # next token should be delta
    assert model.tokenizer.decode([token_id]) == " delta"

    #
    # Whens
    #

    # I collect next token
    token_id = next(it)

    #
    # Thens
    #

    # next token should be epsilon
    assert model.tokenizer.decode([token_id]) == " epsilon"
