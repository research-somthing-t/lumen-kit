from __future__ import annotations

from .pipeline import Draft, StagePlan, Step

_DRAFTS = (
    Draft(start="plain", model="gpt-6.2-sol", effort="high"),
    Draft(start="proportions", model="gpt-6.2-sol", effort="high"),
    Draft(start="plain", model="gpt-6.2-sol", effort="high"),
    Draft(start="proportions", model="gpt-6.2-sol", effort="high"),
    Draft(start="plain", model="gpt-6.2-sol", effort="high"),
    Draft(start="proportions", model="gpt-6.2-sol", effort="high"),
)

_STEPS = (
    Step(rewrites=(("gpt-6.2-sol", "high"),), each_lineage=True, judge_pick=True, focus=("materials", "shape"), sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, focus=("materials", "shape"), feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", patch=True, feed_diffs=True, feed_critique=True, sillint=True),
    Step(rewrites=(("gpt-6.2-sol", "xhigh"),), fallback=("gpt-6.2-sol", "high"), each_lineage=True, judge_pick=True, nudge="details", patch=True, feed_diffs=True, feed_critique=True, sillint=True),
)

ACTIVE_PLAN = StagePlan(
    drafts=_DRAFTS,
    steps=_STEPS,
    draft_keep=1,
    draft_each_start=True,
    fill=0.999,
    glow=0.0,
    measured=False,
    outline_fit=False,
    outline_fit_last=True,
    replan=True,
    merge_pick=True,
    sillint=True,
    tourn_mode="",
)
