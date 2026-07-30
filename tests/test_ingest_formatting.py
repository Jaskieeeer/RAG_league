from lolrag.ingest.formatting import format_term
from lolrag.ingest.formulas import (
    BreakpointStep,
    BreakpointsTerm,
    ByLevelTerm,
    ConstantTerm,
    ProductTerm,
    StatTerm,
    SumTerm,
)

Q_BASE_DAMAGE = ConstantTerm(values=(10.0, 25.0, 40.0, 55.0, 70.0))
Q_AD_SCALING = StatTerm(
    coefficient=ConstantTerm(values=(0.6, 0.675, 0.75, 0.825, 0.9)),
    stat="ad",
    stat_formula="total",
)


# ---------- numbers ----------


def test_a_rank_invariant_array_collapses_to_a_single_number() -> None:
    """Repeating one number five times reads as noise, so it is stated once."""
    assert format_term(ConstantTerm(values=(5.0,) * 5)) == "5"


def test_a_varying_array_joins_every_rank_with_a_slash() -> None:
    """A number that changes with rank keeps every rank, in the game's own idiom."""
    assert format_term(Q_BASE_DAMAGE) == "10/25/40/55/70"


def test_four_significant_figures_strip_the_source_float32_noise() -> None:
    """A stored 0.800000011920929 means 0.8 and must not be shown as it is stored."""
    assert format_term(ConstantTerm(values=(0.800000011920929,))) == "0.8"


def test_four_significant_figures_keep_a_repeating_fraction() -> None:
    """A stored 0.3333300054073334 is a third, not an integer."""
    assert format_term(ConstantTerm(values=(0.3333300054073334,)), scale=100.0) == "33.33"


def test_an_explicit_decimal_count_is_honoured_exactly() -> None:
    """A source asking for one decimal gets 1.0, not the shorter 1."""
    assert format_term(ConstantTerm(values=(1.0,)), decimals=1) == "1.0"


def test_an_explicit_zero_decimal_count_rounds_to_an_integer() -> None:
    """The K'Sante case: 33.333 percent asked for at zero decimals is 33."""
    assert format_term(ConstantTerm(values=(0.3333300054073334,)), scale=100.0, decimals=0) == "33"


def test_a_scale_multiplies_every_number_in_the_tree() -> None:
    """A tooltip token's own arithmetic reaches the renderer as a scale."""
    assert format_term(Q_BASE_DAMAGE, scale=0.5) == "5/12.5/20/27.5/35"


def test_a_value_that_rounds_to_nothing_negative_is_not_shown_as_minus_zero() -> None:
    """A displayed "-0" is a rendering artefact, never a fact about the game."""
    assert format_term(ConstantTerm(values=(-0.004,)), decimals=1) == "0.0"


# ---------- percentages ----------


def test_a_percentage_scales_the_base_by_a_hundred_and_signs_it() -> None:
    """A calculation flagged as a percentage states a fraction, not a magnitude."""
    assert format_term(ConstantTerm(values=(0.4, 0.55)), percent=True) == "40/55%"


def test_a_percentage_scaling_renders_per_hundred_points_of_the_stat() -> None:
    """Camille's W adds percentage points per point of a stat, not a share of it."""
    term = SumTerm(
        terms=(
            ConstantTerm(values=(0.055, 0.06, 0.065, 0.07, 0.075)),
            StatTerm(
                coefficient=ConstantTerm(values=(0.00025,) * 5), stat="ad", stat_formula="bonus"
            ),
        )
    )

    assert format_term(term, percent=True) == "5.5/6/6.5/7/7.5% (+2.5% per 100 bonus AD)"


def test_a_percentage_scaling_with_an_undecoded_stat_is_refused() -> None:
    """A rate per hundred points of an unnamed stat states nothing usable."""
    term = SumTerm(
        terms=(
            ConstantTerm(values=(0.1,)),
            StatTerm(coefficient=ConstantTerm(values=(1.5,)), stat=None, stat_formula=None),
        )
    )

    assert format_term(term, percent=True) is None


# ---------- stat scaling ----------


def test_stat_scaling_renders_in_the_games_own_idiom() -> None:
    """The base damage first, then the ratio as a percentage of the named stat."""
    term = SumTerm(terms=(Q_BASE_DAMAGE, Q_AD_SCALING))

    assert format_term(term) == "10/25/40/55/70 (+60/67.5/75/82.5/90% total AD)"


def test_an_undecoded_stat_is_named_as_unnamed_rather_than_guessed() -> None:
    """Pyke's R scales with an enum this corpus cannot decode; it says so."""
    term = SumTerm(
        terms=(
            ConstantTerm(values=(250.0,)),
            StatTerm(coefficient=ConstantTerm(values=(1.5,)), stat=None, stat_formula=None),
        )
    )

    assert format_term(term) == "250 (+150% of an unnamed stat)"


def test_an_undecoded_stat_formula_leaves_the_word_out_and_keeps_the_stat() -> None:
    """Whether it is the total or the bonus stat is unproven, so it is unsaid."""
    term = StatTerm(coefficient=ConstantTerm(values=(0.6,)), stat="ap", stat_formula=None)

    assert format_term(term) == "60% AP"


def test_a_formula_that_is_only_stat_scaling_drops_its_zero_base() -> None:
    """Writing "0 (+60% total AD)" adds a number the source never states."""
    term = SumTerm(terms=(ConstantTerm(values=(0.0,)), Q_AD_SCALING))

    assert format_term(term) == "60/67.5/75/82.5/90% total AD"


def test_every_stat_code_gets_a_readable_name() -> None:
    """The corpus stores stat codes; a reader needs the words."""
    rendered = [
        format_term(StatTerm(coefficient=ConstantTerm(values=(1.0,)), stat=stat, stat_formula=None))
        for stat in ("ad", "ap", "armor", "magic_resist", "attack_speed", "crit", "health")
    ]

    assert rendered == [
        "100% AD",
        "100% AP",
        "100% armor",
        "100% magic resist",
        "100% attack speed",
        "100% critical strike chance",
        "100% health",
    ]


# ---------- level scaling ----------


def test_a_by_level_term_renders_a_range_across_the_champions_levels() -> None:
    """A level range must not be mistaken for a per-rank array, so it says so."""
    assert format_term(ByLevelTerm(start=15.0, end=25.0)) == "15 to 25 (based on level)"


def test_a_by_level_term_with_equal_endpoints_is_a_plain_number() -> None:
    """Nothing scales, so nothing claims to."""
    assert format_term(ByLevelTerm(start=8.0, end=8.0)) == "8"


def test_a_breakpoint_carrying_only_a_level_stops_the_growth_there() -> None:
    """Camille's Q converts 40 percent at level one, growing 4 points a level, and
    its only step is a bare level 17: the value caps at exactly 100 percent."""
    term = BreakpointsTerm(
        level1_value=0.4,
        initial_bonus_per_level=0.04,
        steps=(BreakpointStep(level=17, additional_bonus=0.0, bonus_per_level_after=0.0),),
    )

    assert format_term(term, percent=True) == "40 to 100% (based on level)"


def test_a_breakpoint_replaces_the_rate_in_force_before_it() -> None:
    """Pyke's R executes below 250 health at level one and 550 at level 18."""
    term = BreakpointsTerm(
        level1_value=250.0,
        initial_bonus_per_level=0.0,
        steps=(
            BreakpointStep(level=7, additional_bonus=0.0, bonus_per_level_after=40.0),
            BreakpointStep(level=10, additional_bonus=0.0, bonus_per_level_after=30.0),
            BreakpointStep(level=12, additional_bonus=0.0, bonus_per_level_after=20.0),
            BreakpointStep(level=17, additional_bonus=0.0, bonus_per_level_after=10.0),
        ),
    )

    assert format_term(term) == "250 to 550 (based on level)"


def test_a_breakpoint_one_off_bonus_lands_on_its_own_level() -> None:
    """Akshan's passive cooldown steps 16, 12, 8, 4 at levels 1, 6, 11 and 16."""
    term = BreakpointsTerm(
        level1_value=16.0,
        initial_bonus_per_level=0.0,
        steps=tuple(
            BreakpointStep(level=level, additional_bonus=-4.0, bonus_per_level_after=0.0)
            for level in (6, 11, 16)
        ),
    )

    assert format_term(term) == "16 to 4 (based on level)"


# ---------- nesting ----------


def test_a_constant_factor_distributes_into_the_base_and_into_the_scaling() -> None:
    """Aatrox's edge damage is his Q times 1.75, ratio included."""
    term = ProductTerm(
        terms=(
            SumTerm(terms=(Q_BASE_DAMAGE, Q_AD_SCALING)),
            SumTerm(terms=(ConstantTerm(values=(1.0,) * 5), ConstantTerm(values=(0.75,) * 5))),
        )
    )

    assert format_term(term) == (
        "17.5/43.75/70/96.25/122.5 (+105/118.1/131.2/144.4/157.5% total AD)"
    )


def test_nested_sums_add_up_into_one_number() -> None:
    """A sum of sums is one quantity, however deeply the source nests it."""
    term = SumTerm(
        terms=(
            SumTerm(terms=(ConstantTerm(values=(10.0,)), ConstantTerm(values=(5.0,)))),
            ConstantTerm(values=(2.5,)),
        )
    )

    assert format_term(term) == "17.5"


def test_a_constant_added_to_a_level_range_shifts_both_of_its_ends() -> None:
    """Azir's soldier damage is a per-rank base plus a level-scaled bonus."""
    term = SumTerm(terms=(ConstantTerm(values=(50.0, 70.0)), ByLevelTerm(start=0.0, end=72.0)))

    assert format_term(term) == "50 to 122/70 to 142 (based on level)"


def test_a_stat_multiplied_by_another_stat_is_refused() -> None:
    """Critical strike chance times attack damage is a curve, not a line."""
    term = ProductTerm(
        terms=(
            StatTerm(coefficient=ConstantTerm(values=(1.0,)), stat="crit", stat_formula="total"),
            StatTerm(coefficient=ConstantTerm(values=(1.0,)), stat="ad", stat_formula="total"),
        )
    )

    assert format_term(term) is None
