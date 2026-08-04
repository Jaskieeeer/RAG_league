"""The no-retrieval control system: the same model answering from pretraining alone.

Everything except retrieval is held constant - same Gemini model, same fallback,
same temperature, same assistant persona - so a difference between the two
systems is attributable to the retrieved context and to nothing else.

The grounding clause is the one thing that cannot be held constant. Keeping
"answer using only the context below" with no context would not be a control, it
would be an instruction to refuse, and it would measure obedience rather than
knowledge. The clause is therefore replaced by its closest no-context
equivalent: answer from your own knowledge, and say so if you do not know.
"""

from langchain_core.prompts import ChatPromptTemplate

from lolrag.config import Settings
from lolrag.pipeline import get_llm

_BASELINE_SYSTEM_PROMPT = (
    "You are a League of Legends knowledge assistant. Answer the question from "
    "your own knowledge. No reference material is provided. If you do not know "
    "the answer, say so explicitly."
)


def build_baseline_prompt() -> ChatPromptTemplate:
    """Build the chat prompt the no-retrieval baseline answers with.

    Returns:
        ChatPromptTemplate with the assistant persona in the system message and
        the literal {question} in the human message, carrying no {context}
        placeholder of any kind.
    """
    return ChatPromptTemplate.from_messages(
        [("system", _BASELINE_SYSTEM_PROMPT), ("human", "{question}")]
    )


def generate_without_context(question: str, settings: Settings) -> str:
    """Answer a question with no retrieved context, using the pipeline's model.

    Args:
        question: User question to answer.
        settings: Application settings providing llm_model_name,
            llm_fallback_model_name, llm_temperature and google_api_key.

    Returns:
        Generated answer text, grounded in nothing but the model's pretraining.
    """
    messages = build_baseline_prompt().format_messages(question=question)
    llm = get_llm(
        settings.llm_model_name,
        settings.llm_fallback_model_name,
        settings.llm_temperature,
        settings.google_api_key.get_secret_value() if settings.google_api_key else None,
    )
    return str(llm.invoke(messages).text)
