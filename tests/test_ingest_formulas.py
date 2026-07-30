from typing import Any

from lolrag.ingest.formulas import (
    BreakpointStep,
    BreakpointsTerm,
    ByLevelTerm,
    ConstantTerm,
    Formula,
    ProductTerm,
    StatTerm,
    SumTerm,
    Unevaluable,
    evaluate_calculation,
)

MAX_RANK = 5
Q_BASE_DAMAGE = [-5.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
Q_AD_RATIO = [0.525, 0.6, 0.675, 0.75, 0.825, 0.9, 0.975]
SLICED_BASE_DAMAGE = (10.0, 25.0, 40.0, 55.0, 70.0)
SLICED_AD_RATIO = (0.6, 0.675, 0.75, 0.825, 0.9)

DATA_VALUES: dict[str, list[float] | None] = {
    "QBaseDamage": Q_BASE_DAMAGE,
    "QTotalADRatio": Q_AD_RATIO,
    "Unused": None,
}


# ---------- node builders ----------


def part(node_type: str, **fields: Any) -> dict[str, Any]:
    """Build one calculation part carrying exactly the fields given.

    Args:
        node_type: Value of the part's __type.
        fields: Source fields to publish. Anything left out is a field the bin
            omits because it sits at its default.

    Returns:
        A calculation part object.
    """
    return {**fields, "__type": node_type}


def calculation(*parts: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Build a GameCalculation summing the given parts.

    Args:
        parts: The parts of mFormulaParts, in source order.
        fields: Further fields such as mMultiplier or mDisplayAsPercent.

    Returns:
        A GameCalculation object.
    """
    return {"mFormulaParts": list(parts), **fields, "__type": "GameCalculation"}


def modified(reference: str, multiplier: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Build a GameCalculationModified scaling another calculation.

    Args:
        reference: Name of the calculation this one defers to.
        multiplier: The part supplying the factor it is scaled by.
        fields: Further fields the wrapper publishes.

    Returns:
        A GameCalculationModified object.
    """
    return {
        "mModifiedGameCalculation": reference,
        "mMultiplier": multiplier,
        **fields,
        "__type": "GameCalculationModified",
    }


def evaluate(
    calculations: dict[str, Any],
    name: str = "Damage",
    max_rank: int | None = MAX_RANK,
    data_values: dict[str, list[float] | None] | None = None,
    effect_values: dict[int, list[float] | None] | None = None,
) -> Formula | Unevaluable:
    """Evaluate one calculation against the shared Aatrox-shaped data values.

    Args:
        calculations: The spell's mSpellCalculations object.
        name: Calculation to evaluate.
        max_rank: Number of learnable ranks, or None for a passive.
        data_values: Data values to resolve against, defaulting to DATA_VALUES.
        effect_values: Effect amounts by one-based index, if any.

    Returns:
        The evaluation result.
    """
    return evaluate_calculation(
        name,
        calculations,
        DATA_VALUES if data_values is None else data_values,
        max_rank,
        effect_values=effect_values,
    )


def term(result: Formula | Unevaluable) -> Any:
    """Return the term tree of a result that must have evaluated.

    Args:
        result: Result returned by evaluate_calculation.

    Returns:
        The root term.

    Raises:
        AssertionError: If the result is unevaluable, so a broken evaluation
            fails on the test that asked for the term rather than later.
    """
    assert isinstance(result, Formula), result
    return result.term


# ---------- leaf parts ----------


def test_number_part_repeats_one_value_across_every_rank() -> None:
    """A rank-invariant number still arrives as one entry per rank."""
    result = evaluate({"Damage": calculation(part("NumberCalculationPart", mNumber=25.0))})

    assert term(result) == SumTerm(terms=(ConstantTerm(values=(25.0,) * 5),))


def test_number_part_reads_an_absent_number_as_zero() -> None:
    """These bins omit any field at its default, so no mNumber means zero."""
    result = evaluate({"Damage": calculation(part("NumberCalculationPart"))})

    assert term(result) == SumTerm(terms=(ConstantTerm(values=(0.0,) * 5),))


def test_named_data_value_part_slices_the_padded_source_array() -> None:
    """A data value arrives cut down to the ranks the ability actually has."""
    result = evaluate(
        {"Damage": calculation(part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"))}
    )

    assert term(result) == SumTerm(terms=(ConstantTerm(values=SLICED_BASE_DAMAGE),))


def test_named_data_value_part_reads_a_null_array_as_zero() -> None:
    """A value the bin publishes with a null array sits at its default everywhere."""
    result = evaluate(
        {"Damage": calculation(part("NamedDataValueCalculationPart", mDataValue="Unused"))}
    )

    assert term(result) == SumTerm(terms=(ConstantTerm(values=(0.0,) * 5),))


def test_named_data_value_part_refuses_a_name_the_spell_does_not_publish() -> None:
    """A stale reference is not a defaulted field, so it never reads as zero."""
    result = evaluate(
        {"Damage": calculation(part("NamedDataValueCalculationPart", mDataValue="Missing"))}
    )

    assert result == Unevaluable(
        node_type="NamedDataValueCalculationPart", reason="no data value Missing"
    )


def test_a_passive_takes_the_rank_one_entry_of_every_array() -> None:
    """A passive publishes no max rank, so each constant carries one entry."""
    result = evaluate(
        {"Damage": calculation(part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"))},
        max_rank=None,
    )

    assert term(result) == SumTerm(terms=(ConstantTerm(values=(10.0,)),))


def test_effect_value_part_resolves_against_the_supplied_effect_amounts() -> None:
    """An effect index reads the owning spell's mEffectAmount array at that index."""
    result = evaluate(
        {"Damage": calculation(part("EffectValueCalculationPart", mEffectIndex=1))},
        effect_values={1: Q_BASE_DAMAGE},
    )

    assert term(result) == SumTerm(terms=(ConstantTerm(values=SLICED_BASE_DAMAGE),))


def test_effect_value_part_is_unevaluable_when_the_caller_supplied_nothing() -> None:
    """Without effect amounts the calculation is refused rather than guessed."""
    result = evaluate({"Damage": calculation(part("EffectValueCalculationPart", mEffectIndex=1))})

    assert result == Unevaluable(node_type="EffectValueCalculationPart", reason="no effect value 1")


# ---------- stat scaling ----------


def test_stat_by_coefficient_reads_the_absent_enums_as_total_ability_power() -> None:
    """An omitted mStat is Ability Power and an omitted mStatFormula is the total stat."""
    result = evaluate(
        {"Damage": calculation(part("StatByCoefficientCalculationPart", mCoefficient=0.8))}
    )

    assert term(result) == SumTerm(
        terms=(
            StatTerm(coefficient=ConstantTerm(values=(0.8,) * 5), stat="ap", stat_formula="total"),
        )
    )


def test_stat_by_named_data_value_reads_mstat_two_and_formula_two_as_bonus_attack_damage() -> None:
    """The corpus proves mStatFormula two is the bonus stat and mStat two is attack damage."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "StatByNamedDataValueCalculationPart",
                    mDataValue="QTotalADRatio",
                    mStat=2,
                    mStatFormula=2,
                )
            )
        }
    )

    assert term(result) == SumTerm(
        terms=(
            StatTerm(
                coefficient=ConstantTerm(values=SLICED_AD_RATIO), stat="ad", stat_formula="bonus"
            ),
        )
    )


def test_an_undecoded_stat_enum_leaves_the_stat_null_without_blocking() -> None:
    """An mStat with no proven meaning yields None rather than a guess or a refusal."""
    result = evaluate(
        {"Damage": calculation(part("StatByCoefficientCalculationPart", mCoefficient=1.0, mStat=9))}
    )

    assert term(result) == SumTerm(
        terms=(
            StatTerm(coefficient=ConstantTerm(values=(1.0,) * 5), stat=None, stat_formula="total"),
        )
    )


def test_an_undecoded_stat_formula_enum_leaves_the_formula_null() -> None:
    """Only zero and two are proven, so mStatFormula one stays undecided."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "StatByCoefficientCalculationPart", mCoefficient=0.01, mStat=12, mStatFormula=1
                )
            )
        }
    )

    assert term(result) == SumTerm(
        terms=(
            StatTerm(
                coefficient=ConstantTerm(values=(0.01,) * 5), stat="health", stat_formula=None
            ),
        )
    )


def test_stat_by_sub_part_wraps_the_term_its_subpart_reduces_to() -> None:
    """A stat declared on a wrapper scales whatever the subpart computes."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "StatBySubPartCalculationPart",
                    mStat=12,
                    mSubpart=part("NamedDataValueCalculationPart", mDataValue="QTotalADRatio"),
                )
            )
        }
    )

    assert term(result) == SumTerm(
        terms=(
            StatTerm(
                coefficient=ConstantTerm(values=SLICED_AD_RATIO),
                stat="health",
                stat_formula="total",
            ),
        )
    )


# ---------- by-level parts ----------


def test_interpolation_part_carries_its_two_endpoints() -> None:
    """A level interpolation reduces to the level-one and maximum-level values."""
    result = evaluate(
        {
            "Damage": calculation(
                part("ByCharLevelInterpolationCalculationPart", mStartValue=0.04, mEndValue=0.10)
            )
        }
    )

    assert term(result) == SumTerm(terms=(ByLevelTerm(start=0.04, end=0.10),))


def test_breakpoints_part_carries_every_step_in_source_order() -> None:
    """A breakpoint publishing nothing but a level stops the growth there."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "ByCharLevelBreakpointsCalculationPart",
                    mLevel1Value=0.55,
                    mInitialBonusPerLevel=0.01,
                    mBreakpoints=[
                        part("Breakpoint", mLevel=6, mAdditionalBonusAtThisLevel=0.05),
                        part("Breakpoint", mLevel=11, mBonusPerLevelAtAndAfter=0.02),
                        part("Breakpoint", mLevel=19),
                    ],
                )
            )
        }
    )

    assert term(result) == SumTerm(
        terms=(
            BreakpointsTerm(
                level1_value=0.55,
                initial_bonus_per_level=0.01,
                steps=(
                    BreakpointStep(level=6, additional_bonus=0.05, bonus_per_level_after=0.0),
                    BreakpointStep(level=11, additional_bonus=0.0, bonus_per_level_after=0.02),
                    BreakpointStep(level=19, additional_bonus=0.0, bonus_per_level_after=0.0),
                ),
            ),
        )
    )


def test_breakpoints_part_reads_an_absent_level_one_value_as_zero() -> None:
    """A breakpoints part with no published fields is zero everywhere."""
    result = evaluate({"Damage": calculation(part("ByCharLevelBreakpointsCalculationPart"))})

    assert term(result) == SumTerm(
        terms=(BreakpointsTerm(level1_value=0.0, initial_bonus_per_level=0.0, steps=()),)
    )


# ---------- nesting ----------


def test_sum_and_product_parts_nest_without_flattening() -> None:
    """A product of two sums keeps both operands, which a flat shape would lose."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "ProductOfSubPartsCalculationPart",
                    mPart1=part(
                        "SumOfSubPartsCalculationPart",
                        mSubparts=[
                            part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"),
                            part("NumberCalculationPart", mNumber=5.0),
                        ],
                    ),
                    mPart2=part("NumberCalculationPart", mNumber=1.75),
                )
            )
        }
    )

    assert term(result) == SumTerm(
        terms=(
            ProductTerm(
                terms=(
                    SumTerm(
                        terms=(
                            ConstantTerm(values=SLICED_BASE_DAMAGE),
                            ConstantTerm(values=(5.0,) * 5),
                        )
                    ),
                    ConstantTerm(values=(1.75,) * 5),
                )
            ),
        )
    )


def test_a_calculation_multiplier_becomes_a_product_around_the_whole_sum() -> None:
    """mMultiplier is a formula part scaling the sum, not a plain number beside it."""
    result = evaluate(
        {
            "Damage": calculation(
                part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"),
                mMultiplier=part("NumberCalculationPart", mNumber=0.5),
            )
        }
    )

    assert term(result) == ProductTerm(
        terms=(
            SumTerm(terms=(ConstantTerm(values=SLICED_BASE_DAMAGE),)),
            ConstantTerm(values=(0.5,) * 5),
        )
    )


def test_a_modified_calculation_follows_its_reference_to_a_sibling() -> None:
    """The Aatrox edge case: the sweet-spot damage is the base damage times 1.75."""
    calculations = {
        "QDamage": calculation(part("NamedDataValueCalculationPart", mDataValue="QBaseDamage")),
        "QEdgeDamage": modified(
            "QDamage",
            part(
                "SumOfSubPartsCalculationPart",
                mSubparts=[
                    part("NumberCalculationPart", mNumber=1.0),
                    part("NumberCalculationPart", mNumber=0.75),
                ],
            ),
        ),
    }

    result = evaluate(calculations, name="QEdgeDamage")

    assert term(result) == ProductTerm(
        terms=(
            SumTerm(terms=(ConstantTerm(values=SLICED_BASE_DAMAGE),)),
            SumTerm(terms=(ConstantTerm(values=(1.0,) * 5), ConstantTerm(values=(0.75,) * 5))),
        )
    )


def test_a_reference_to_a_missing_calculation_is_unevaluable() -> None:
    """A reference naming nothing the spell publishes blocks the calculation."""
    calculations = {"QEdgeDamage": modified("Absent", part("NumberCalculationPart", mNumber=2.0))}

    result = evaluate(calculations, name="QEdgeDamage")

    assert result == Unevaluable(node_type="MissingGameCalculation", reason="no calculation Absent")


def test_a_reference_cycle_is_refused_instead_of_recursing_forever() -> None:
    """Two calculations that defer to each other terminate with a named block."""
    calculations = {
        "First": modified("Second", part("NumberCalculationPart", mNumber=2.0)),
        "Second": modified("First", part("NumberCalculationPart", mNumber=3.0)),
    }

    result = evaluate(calculations, name="First")

    assert result == Unevaluable(
        node_type="GameCalculationModified", reason="reference cycle at First"
    )


# ---------- refusals ----------


def test_an_unhandled_part_type_names_itself_instead_of_raising() -> None:
    """The tail of rare part types produces a signal, never an exception."""
    result = evaluate({"Damage": calculation(part("CooldownMultiplierCalculationPart"))})

    assert result == Unevaluable(
        node_type="CooldownMultiplierCalculationPart", reason="unsupported part type"
    )


def test_an_unhandled_part_deep_in_a_tree_blocks_the_whole_calculation() -> None:
    """No partial tree leaks out, because a formula missing a term renders a wrong number."""
    result = evaluate(
        {
            "Damage": calculation(
                part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"),
                part(
                    "SumOfSubPartsCalculationPart",
                    mSubparts=[
                        part("NumberCalculationPart", mNumber=1.0),
                        part(
                            "ProductOfSubPartsCalculationPart",
                            mPart1=part("NumberCalculationPart", mNumber=2.0),
                            mPart2=part(
                                "BuffCounterByCoefficientCalculationPart",
                                mBuffName="Stacks",
                                mCoefficient=1.0,
                            ),
                        ),
                    ],
                ),
            )
        }
    )

    assert result == Unevaluable(
        node_type="BuffCounterByCoefficientCalculationPart", reason="unsupported part type"
    )


def test_an_unhandled_wrapper_type_names_itself() -> None:
    """A conditional calculation is a wrapper this module does not decode."""
    calculations = {
        "Damage": {
            "mConditionalGameCalculation": "Other",
            "mConditionalCalculationRequirements": [],
            "__type": "GameCalculationConditional",
        }
    }

    result = evaluate(calculations)

    assert result == Unevaluable(
        node_type="GameCalculationConditional", reason="unsupported calculation type"
    )


def test_a_field_this_module_has_never_decoded_blocks_its_node() -> None:
    """An unknown field can change the number, so the node carrying it is refused."""
    result = evaluate(
        {
            "Damage": calculation(
                part(
                    "ByCharLevelInterpolationCalculationPart",
                    mStartValue=0.2,
                    mEndValue=1.0,
                    mScaleByStatProgressionMultiplier=True,
                )
            )
        }
    )

    assert result == Unevaluable(
        node_type="ByCharLevelInterpolationCalculationPart",
        reason="unrecognised field mScaleByStatProgressionMultiplier",
    )


def test_a_name_the_spell_does_not_publish_is_unevaluable() -> None:
    """Asking for a calculation that does not exist is answered, not raised."""
    result = evaluate({"Damage": calculation()}, name="Absent")

    assert result == Unevaluable(node_type="MissingGameCalculation", reason="no calculation Absent")


# ---------- calculation attributes ----------


def test_the_display_flags_come_from_the_calculation_the_caller_named() -> None:
    """Percentage and precision are read off the named calculation, not inherited."""
    calculations = {
        "Base": calculation(
            part("NamedDataValueCalculationPart", mDataValue="QBaseDamage"),
            mDisplayAsPercent=True,
            mPrecision=2,
        ),
        "Scaled": modified("Base", part("NumberCalculationPart", mNumber=2.0)),
    }

    base = evaluate(calculations, name="Base")
    scaled = evaluate(calculations, name="Scaled")

    assert isinstance(base, Formula)
    assert isinstance(scaled, Formula)
    assert (base.display_as_percent, base.precision) == (True, 2)
    assert (scaled.display_as_percent, scaled.precision) == (False, 0)
