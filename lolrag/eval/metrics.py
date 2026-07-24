def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> bool:
    """Report whether any relevant id appears within the first k retrieved ids.

    Args:
        retrieved_ids: Retrieved ids in rank order, best first.
        relevant_ids: Ids considered relevant for the query.
        k: Cutoff rank; only the first k retrieved ids are considered.

    Returns:
        True if at least one of the first k retrieved ids is in relevant_ids,
        otherwise False.
    """
    return any(retrieved_id in relevant_ids for retrieved_id in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute the reciprocal rank of the first relevant retrieved id.

    Args:
        retrieved_ids: Retrieved ids in rank order, best first.
        relevant_ids: Ids considered relevant for the query.

    Returns:
        1 divided by the 1-indexed rank of the first relevant id, or 0.0 if no
        retrieved id is relevant.
    """
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / index
    return 0.0
