from functools import lru_cache

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


@lru_cache
def get_judge(model_name: str, api_key: str | None) -> Runnable:
    """Return a process-wide cached faithfulness judge with structured output.

    Args:
        model_name: Gemini chat model identifier used as the judge.
        api_key: Gemini API key, or None to defer to the GOOGLE_API_KEY
            environment variable.

    Returns:
        Runnable that returns a FaithfulnessVerdict, cached per unique argument
        combination.
    """
    judge = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
    return judge.with_structured_output(FaithfulnessVerdict)


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
        [
            ("system", _JUDGE_SYSTEM_PROMPT),
            ("human", _JUDGE_HUMAN_TEMPLATE),
        ]
    )
    api_key = settings.google_api_key.get_secret_value() if settings.google_api_key else None
    judge = get_judge(settings.eval_judge_model_name, api_key)
    messages = prompt.format_messages(question=question, context=context, answer=answer)
    verdict = judge.invoke(messages)
    return verdict
