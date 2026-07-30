SPELL_SLOTS = ("Q", "W", "E", "R")
PASSIVE_SLOT = "P"

UNIVERSE_SLUG_OVERRIDES = {"Renata": "renataglasc"}


def universe_slug(ddragon_id: str) -> str:
    """Return the Riot Universe slug for a Data Dragon champion id.

    Args:
        ddragon_id: Data Dragon champion id string, e.g. "Aatrox" or
            "MonkeyKing".

    Returns:
        The Universe slug, which is the lowercased id for every champion
        except the ones listed in UNIVERSE_SLUG_OVERRIDES.
    """
    return UNIVERSE_SLUG_OVERRIDES.get(ddragon_id, ddragon_id.lower())
