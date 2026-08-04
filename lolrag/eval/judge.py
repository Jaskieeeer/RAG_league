"""The LLM judges: groundedness, reference entailment and refusal classification.

The entailment judge is deliberately not a fact checker. It grades an answer
against the dataset's reference answer and nothing else, because the corpus is
narrower than the real game: Blitzcrank's Rocket Grab stuns in the live game and
the corpus never says so, so a judge allowed to use its own knowledge would
reward an ungrounded baseline for the stun and penalise the grounded pipeline for
omitting it. That single constraint is what makes the prose numbers mean
anything, so it is stated three times in the prompt.
"""

from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from lolrag.config import Settings

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict groundedness judge for a retrieval-augmented question "
    "answering system. You are given a QUESTION, the CONTEXT that was retrieved, "
    "and the ANSWER the system produced. Rate on an integer scale from 1 to 5 how "
    "well the ANSWER is supported ONLY by the CONTEXT, using no outside knowledge "
    "of your own.\n\n"
    "Scoring:\n"
    "5 = every claim in the answer is directly supported by the context.\n"
    "1 = the answer is mostly fabricated or unsupported by the context.\n"
    "Intermediate scores reflect the proportion of supported claims.\n\n"
    "Rules:\n"
    "- If the context does not contain the requested information and the answer "
    "correctly declines to answer or states the information is unavailable, that "
    "is FAITHFUL and must score high.\n"
    "- Inventing specifics that are not present in the context, such as numbers, "
    "names, or dates, is unfaithful and must score low.\n"
    "- Judge only groundedness against the context, not whether the answer is "
    "correct in the real world."
)

_JUDGE_HUMAN_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Rate the faithfulness of the ANSWER to the CONTEXT."
)

_ENTAILMENT_SYSTEM_PROMPT = (
    "You compare a candidate ANSWER against a REFERENCE answer for a question "
    "about League of Legends.\n\n"
    "Judge one thing only: does the ANSWER assert what the REFERENCE asserts?\n\n"
    "You are NOT a fact checker. Never ask whether a statement is true in the "
    "real game. Do not use your own knowledge of League of Legends at any point. "
    "The REFERENCE is the only authority. It was written from a fixed corpus that "
    "is deliberately narrower than the real game, so claims that are true in the "
    "game but absent from the REFERENCE are out of scope for this judgement.\n\n"
    "Rules:\n"
    "- Only the substantive claims the REFERENCE makes are graded. An ANSWER that "
    "omits something the REFERENCE never states is complete, not incomplete. For "
    "example, if the REFERENCE says an ability pulls an enemy and deals damage "
    "and says nothing about a stun, then an ANSWER that does not mention a stun "
    "is missing nothing; do not mark it down.\n"
    "- Extra claims the REFERENCE does not make are neither rewarded nor "
    "penalised, unless such a claim contradicts the REFERENCE.\n"
    "- Wording, ordering, length and level of detail are irrelevant. Only the "
    "asserted content counts.\n"
    "- Text that declines to answer, or says the information is unavailable, "
    "asserts nothing.\n"
    "- Where the REFERENCE states that the corpus does not record something, an "
    "ANSWER that confidently supplies it contradicts the REFERENCE.\n\n"
    "Verdicts:\n"
    "- entails: the ANSWER asserts every substantive claim of the REFERENCE and "
    "contradicts none of them.\n"
    "- partial: the ANSWER asserts some but not all of them and contradicts none.\n"
    "- misses: the ANSWER asserts almost none of them, or contradicts one."
)

_ENTAILMENT_HUMAN_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "REFERENCE:\n{reference}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Does the ANSWER assert what the REFERENCE asserts?"
)

_REFUSAL_SYSTEM_PROMPT = (
    "You classify one answer from a question answering system on a single axis: "
    "did it decline, or did it assert?\n\n"
    "Set declined to true when the ANSWER says it does not know, says the "
    "information is not available in its sources, refuses to answer, or corrects "
    "a false premise in the QUESTION instead of playing along with it.\n"
    "Set declined to false when the ANSWER supplies a substantive answer to the "
    "QUESTION, however hedged.\n\n"
    "An answer that declines and then answers anyway has asserted; declined is "
    "false. Do not judge whether anything in the ANSWER is correct."
)

_REFUSAL_HUMAN_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nDid the ANSWER decline?"

ENTAILMENT_SCORES: dict[str, float] = {"entails": 1.0, "partial": 0.5, "misses": 0.0}

JUDGE_MAX_ATTEMPTS = 4


class FaithfulnessVerdict(BaseModel):
    """A groundedness judgement of an answer against its retrieved context.

    Args:
        score: Integer faithfulness rating from 1 to 5, where 5 means every claim
            is directly supported by the context and 1 means mostly fabricated.
        reasoning: Short justification for the assigned score.
    """

    score: int = Field(
        ge=1,
        le=5,
        description="Faithfulness rating from 1 to 5, 5 meaning fully grounded in context.",
    )
    reasoning: str = Field(description="Short justification for the assigned score.")


class EntailmentVerdict(BaseModel):
    """A judgement of whether an answer asserts what a reference answer asserts.

    Args:
        verdict: One of entails, partial, misses.
        reasoning: Short justification naming the reference claims the answer
            did or did not assert.
    """

    verdict: Literal["entails", "partial", "misses"] = Field(
        description="Whether the answer asserts every, some or almost none of the reference claims."
    )
    reasoning: str = Field(description="Short justification for the verdict.")


class RefusalVerdict(BaseModel):
    """A classification of whether an answer declined or asserted.

    Args:
        declined: True if the answer declined, said it did not know, or
            corrected the question's false premise instead of answering it.
        reasoning: Short justification for the classification.
    """

    declined: bool = Field(description="Whether the answer declined instead of asserting.")
    reasoning: str = Field(description="Short justification for the classification.")


# ---------- judge construction ----------


@lru_cache
def get_judge(model_name: str, api_key: str | None, schema: type[BaseModel]) -> Runnable:
    """Return a process-wide cached judge bound to one structured output schema.

    The judge retries rather than falling back to another model. A full run is
    hundreds of judged answers, and one transient 503 partway through would
    otherwise throw the whole paid run away; a second model would have salvaged
    it at the price of scoring part of the dataset with a different instrument,
    which is not a trade worth making for a comparison.

    Args:
        model_name: Gemini chat model identifier used as the judge.
        api_key: Gemini API key, or None to defer to the GOOGLE_API_KEY
            environment variable.
        schema: pydantic model the judge must return.

    Returns:
        Runnable returning an instance of schema, retrying up to
        JUDGE_MAX_ATTEMPTS times with exponential backoff, cached per unique
        argument combination.
    """
    judge = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
    return judge.with_structured_output(schema).with_retry(stop_after_attempt=JUDGE_MAX_ATTEMPTS)


def _judge_key(settings: Settings) -> str | None:
    """Read the Gemini API key out of settings for the judge.

    Args:
        settings: Application settings providing google_api_key.

    Returns:
        The key as plain text, or None when none is configured.
    """
    return settings.google_api_key.get_secret_value() if settings.google_api_key else None


# ---------- judgements ----------


def judge_faithfulness(
    question: str, context: str, answer: str, settings: Settings
) -> FaithfulnessVerdict:
    """Judge how well an answer is grounded in its retrieved context.

    Args:
        question: The question the answer responds to.
        context: The retrieved context the answer must be grounded in.
        answer: The generated answer to evaluate.
        settings: Application settings providing eval_judge_model_name and
            google_api_key.

    Returns:
        FaithfulnessVerdict scoring the answer's groundedness against the context.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _JUDGE_SYSTEM_PROMPT), ("human", _JUDGE_HUMAN_TEMPLATE)]
    )
    judge = get_judge(settings.eval_judge_model_name, _judge_key(settings), FaithfulnessVerdict)
    messages = prompt.format_messages(question=question, context=context, answer=answer)
    return judge.invoke(messages)


def judge_entailment(
    question: str, reference: str, answer: str, settings: Settings
) -> EntailmentVerdict:
    """Judge whether an answer asserts what the reference answer asserts.

    Args:
        question: The question both answers respond to.
        reference: The dataset's reference answer, the only authority.
        answer: The generated answer to evaluate.
        settings: Application settings providing eval_judge_model_name and
            google_api_key.

    Returns:
        EntailmentVerdict of entails, partial or misses. Truth in the real game
        is never asked about, so an answer is not rewarded for real-world facts
        the reference omits nor penalised for omitting them.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _ENTAILMENT_SYSTEM_PROMPT), ("human", _ENTAILMENT_HUMAN_TEMPLATE)]
    )
    judge = get_judge(settings.eval_judge_model_name, _judge_key(settings), EntailmentVerdict)
    messages = prompt.format_messages(question=question, reference=reference, answer=answer)
    return judge.invoke(messages)


def judge_refusal(question: str, answer: str, settings: Settings) -> RefusalVerdict:
    """Classify whether an answer declined or asserted a claim.

    Args:
        question: The question the answer responds to.
        answer: The generated answer to classify.
        settings: Application settings providing eval_judge_model_name and
            google_api_key.

    Returns:
        RefusalVerdict whose declined flag is the only thing the refusal block
        scores.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _REFUSAL_SYSTEM_PROMPT), ("human", _REFUSAL_HUMAN_TEMPLATE)]
    )
    judge = get_judge(settings.eval_judge_model_name, _judge_key(settings), RefusalVerdict)
    messages = prompt.format_messages(question=question, answer=answer)
    return judge.invoke(messages)
