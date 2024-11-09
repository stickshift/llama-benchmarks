from pathlib import Path
from textwrap import dedent

import llama_benchmarks as llb
from llama_benchmarks.mmlu import OPTIONS, Question


def test_load_dataset(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset
    questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # There should be 14,042 questions
    assert len(questions) == 14042

    # ids should be unique
    assert len({q.qid for q in questions}) == len(questions)

    # There should be 57 categories
    assert len({q.category for q in questions}) == 57


def test_load_dataset_idempotent(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset twice
    questions1 = llb.mmlu.load_dataset(mmlu_dataset_path)
    questions2 = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # questions1 and questions2 should be the same
    assert questions2 == questions1


def test_swap_answers():
    #
    # Givens
    #

    # question1 has answer "A"
    question1 = Question(qid=0, category="category", question="question", A="A", B="B", C="C", D="D", answer="A")

    #
    # Whens
    #

    # I swap question1 answers to option "A"
    question2 = llb.mmlu.swap_answers([question1], "A")[0]

    #
    # Thens
    #

    # question2 should be identical to question1
    assert question2.qid == question1.qid
    assert question2.category == question1.category
    assert question2.question == question1.question
    assert question2.A == question1.A
    assert question2.B == question1.B
    assert question2.C == question1.C
    assert question2.D == question1.D
    assert question2.answer == question1.answer

    #
    # Whens
    #

    # I swap question1 answers to option "B"
    question2 = llb.mmlu.swap_answers([question1], "B")[0]

    #
    # Thens
    #

    # question2 should be question1 w/ A and B swapped and updated answer
    assert question2.qid == question1.qid
    assert question2.category == question1.category
    assert question2.question == question1.question
    assert question2.A == question1.B
    assert question2.B == question1.A
    assert question2.C == question1.C
    assert question2.D == question1.D
    assert question2.answer == "B"


def test_swap_answers_mmlu(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset
    questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # Answers should be distributed across all options
    distribution = llb.mmlu.answer_distribution(questions)
    assert all(distribution[option] > 0 for option in OPTIONS)

    #
    # Whens
    #

    # I swap answers to option "A"
    questions_a = llb.mmlu.swap_answers(questions, "A")

    #
    # Thens
    #

    # Answers should all be A
    assert all(q.answer == "A" for q in questions_a)


def test_generate_prompt(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded mmlu dataset
    questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    # I selected question 11776
    question = questions[11776]

    #
    # Whens
    #

    # I generate prompt for question 11776
    prompt = llb.mmlu.generate_prompt(question)

    #
    # Thens
    #

    # Prompt should be populated
    expected = dedent(
        """
        Question: The defendant was caught in a thunderstorm while walking down the street. As the defendant was about to open an umbrella that she was carrying, a stranger to the defendant came up to her, snatched the umbrella out of the defendant's hand, and said, "You thief! That's my umbrella. " Enraged by being accosted in such a manner, the defendant grabbed for the umbrella with one hand and pushed the stranger with the other. The stranger hung on to the umbrella but fell over backward onto the sidewalk, which was wet and slippery. When the defendant saw the stranger hit the ground, she calmed down, decided the umbrella was not worth all the commotion, and walked off. As it turned out, the umbrella did in fact belong to the stranger, and the defendant had picked it up by mistake when she was left a restaurant earlier that day. A few moments later, the stranger got up in a daze and stepped into the gutter, where he was struck by a car that was passing another car on the right, in violation of a state law. The stranger died in the hospital two hours later. Which of the following is the most serious crime for which the defendant could be found guilty?
        
        A) Battery.
        B) Larceny.
        C) Involuntary manslaughter.
        D) No crime.
        
        Answer: """
    ).lstrip()
    assert prompt == expected


def test_debias_examples(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset
    examples = llb.mmlu.load_dataset(mmlu_dataset_path, segment="dev")

    # I record categories
    categories = {e.category for e in examples}

    #
    # Thens
    #

    # Answers should NOT be evenly distributed
    for category in categories:
        selected = tuple(e for e in examples if e.category == category)
        distribution = llb.mmlu.answer_distribution(selected)
        assert len(set(distribution.values())) > 1

    #
    # Whens
    #

    # I debias examples
    examples = llb.mmlu.debias_example_answers(examples)

    #
    # Thens
    #

    # Answers should be evenly distributed
    for category in categories:
        selected = tuple(e for e in examples if e.category == category)
        distribution = llb.mmlu.answer_distribution(selected)
        assert len(set(distribution.values())) == 1


def test_debias_questions(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset
    questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # Answers should NOT be evenly distributed
    distribution = llb.mmlu.answer_distribution(questions)
    assert len(set(distribution.values())) > 1

    #
    # Whens
    #

    # I debias questions
    questions = llb.mmlu.debias_question_answers(questions)

    #
    # Thens
    #

    # Answers should be evenly distributed
    distribution = llb.mmlu.answer_distribution(questions)
    assert len(set(distribution.values())) == 1
