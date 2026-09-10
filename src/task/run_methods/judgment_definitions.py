"""Reusable GRADE Evidence-to-Decision judgment guidance.

Methodological basis:
- Alonso-Coello et al. BMJ 2016;353:i2016 (EtD introduction).
- Alonso-Coello et al. BMJ 2016;353:i2089 (clinical recommendations).
- GRADE Working Group, GRADE Book: EtD frameworks, certainty of evidence,
  and decision thresholds.

These are qualitative decision rules, not universal numerical cut-offs. GRADE
expects thresholds for trivial/small/moderate/large effects to be specified for
the outcome and context whenever possible. The rules never expose a record's
reference judgment or evidence.
"""

from __future__ import annotations


CRITERION_JUDGMENT_DEFINITIONS: dict[str, dict[str, str]] = {
    "Problem": {
        "No": "The consequences are not serious or important in this context, and the problem is neither urgent nor an established priority.",
        "Probably no": "The problem is probably not a priority after considering seriousness, burden, urgency, and established priorities, but relevant uncertainty remains.",
        "Probably yes": "The problem is probably a priority because of its seriousness, burden, urgency, or recognised priority status, but the case is not unequivocal.",
        "Yes": "The problem is clearly a priority because its consequences are serious or important, it is urgent, or it is an established priority in the stated context.",
    },
    "Desirable Effects": {
        "Trivial": "Across critical outcomes, the absolute benefit is absent or below the context-specific threshold for an important benefit.",
        "Small": "The absolute benefit crosses the small-benefit threshold but not the moderate threshold, after weighting outcomes by importance and baseline risk.",
        "Moderate": "The absolute benefit crosses the moderate-benefit threshold but not the large threshold, after weighting outcomes by importance and baseline risk.",
        "Large": "The absolute benefit crosses the context-specific large-benefit threshold for one or more critical outcomes and is not offset within this criterion by other desirable outcomes. Do not infer 'large' from statistical significance or a relative effect alone.",
    },
    "Undesirable Effects": {
        "Large": "The absolute harm or burden crosses the context-specific large-harm threshold for one or more critical outcomes. Do not infer magnitude from statistical significance or a relative effect alone.",
        "Moderate": "The absolute harm or burden crosses the moderate-harm threshold but not the large threshold, after weighting outcomes by importance and baseline risk.",
        "Small": "The absolute harm or burden crosses the small-harm threshold but not the moderate threshold.",
        "Trivial": "Across critical outcomes, harm or burden is absent or below the context-specific threshold for an important harm.",
    },
    "Certainty of evidence": {
        "Very low": "There is very little confidence that the true effect lies in the target magnitude range or on the stated side of a decision threshold; it may differ substantially.",
        "Low": "There is limited confidence that the true effect lies in the target range or on the stated side of a decision threshold; it is likely to fall in a different range.",
        "Moderate": "There is moderate confidence that the true effect lies in the target range or on the stated side of a decision threshold; it may possibly fall in a different range.",
        "High": "There is high confidence that the true effect lies in the target range or on the stated side of a decision threshold.",
        "No included studies": "No eligible studies provide evidence for the relevant outcomes.",
    },
    "Values": {
        "Important uncertainty or variability": "Evidence or compelling considerations show important uncertainty about, or substantial variability in, the relative importance people assign to the main outcomes—enough to affect the decision.",
        "Possibly important uncertainty or variability": "Decision-relevant uncertainty or variability in outcome importance is plausible, but not established.",
        "Probably no important uncertainty or variability": "Decision-relevant uncertainty or variability in outcome importance is unlikely, although some remains.",
        "No important uncertainty or variability": "The relative importance assigned to the main outcomes is sufficiently certain and consistent that variability is unlikely to alter the decision.",
        "No known undesirable outcomes": "No relevant undesirable outcomes requiring value trade-offs are known.",
    },
    "Balance of effects": {
        "Favors the comparison": "Considering the magnitude of all desirable and undesirable health effects, outcome importance, and certainty, the net health effect clearly favors the comparison.",
        "Probably favors the comparison": "The net health effect probably favors the comparison, but uncertainty about effects, their magnitude, or outcome importance prevents a definite judgment.",
        "Does not favor either the intervention or the comparison": "The net health effects are similar, trivial, or too closely balanced to favor either option.",
        "Probably favors the intervention": "The net health effect probably favors the intervention, but uncertainty about effects, their magnitude, or outcome importance prevents a definite judgment.",
        "Favors the intervention": "Considering the magnitude of all desirable and undesirable health effects, outcome importance, and certainty, the net health effect clearly favors the intervention.",
    },
    "Resources required": {
        "Large costs": "Incremental resource use crosses the context-specific threshold for a large cost, considering the stated perspective and opportunity costs.",
        "Moderate costs": "Incremental resource use crosses the moderate-cost threshold but not the large-cost threshold, considering the stated perspective.",
        "Negligible costs and savings": "Resource differences are absent or too small to matter.",
        "Moderate savings": "Incremental savings cross the moderate-savings threshold but not the large-savings threshold, considering the stated perspective.",
        "Large savings": "Incremental savings cross the context-specific threshold for large savings, considering the stated perspective and opportunity costs.",
    },
    "Certainty of evidence of required resources": {
        "Very low": "There is very little confidence in the resource-use estimate.",
        "Low": "Confidence in the resource-use estimate is limited.",
        "Moderate": "Confidence is moderate; the resource estimate is probably close but may differ substantially.",
        "High": "There is high confidence that the resource estimate is close to the true value.",
        "No included studies": "No included study directly provides usable evidence about resource requirements; do not convert assumptions or indirect narrative into an evidence-certainty rating.",
    },
    "Cost effectiveness": {
        "Favors the comparison": "Comparative costs and health effects, judged against the relevant willingness-to-pay or opportunity-cost threshold, clearly favor the comparison.",
        "Probably favors the comparison": "Cost effectiveness probably favors the comparison, but uncertainty in costs, effects, or the relevant threshold remains.",
        "Does not favor either the intervention or the comparison": "Neither option has a meaningful cost-effectiveness advantage.",
        "Probably favors the intervention": "Cost effectiveness probably favors the intervention, but uncertainty in costs, effects, or the relevant threshold remains.",
        "Favors the intervention": "Comparative costs and health effects, judged against the relevant willingness-to-pay or opportunity-cost threshold, clearly favor the intervention.",
        "No included studies": "No eligible economic evidence supports a cost-effectiveness judgment.",
    },
    "Equity": {
        "Reduced": "The intervention clearly worsens unfair health differences, for example by disproportionately disadvantaging groups with poorer baseline health or access.",
        "Probably reduced": "Equity is likely to worsen, but uncertainty remains.",
        "Probably no impact": "A meaningful equity effect is unlikely.",
        "Probably increased": "Equity is likely to improve, but uncertainty remains.",
        "Increased": "The intervention clearly reduces unfair health differences, for example by disproportionately benefiting disadvantaged groups or improving access.",
    },
    "Acceptability": {
        "No": "Key stakeholders clearly would not accept the intervention, considering distribution of benefits, harms and costs; autonomy; ethical concerns; and reactions to the intervention.",
        "Probably no": "The intervention is probably unacceptable to key stakeholders on those considerations, but uncertainty remains.",
        "Probably yes": "The intervention is probably acceptable to key stakeholders, although reservations or uncertainty remain.",
        "Yes": "Key stakeholders clearly would accept the intervention after considering benefits, harms, costs, autonomy, ethics, and the intervention itself.",
    },
    "Feasibility": {
        "No": "Implementation is not feasible in the stated setting because major practical barriers cannot reasonably be overcome.",
        "Probably no": "Implementation is probably not feasible given existing infrastructure, workforce, training, coordination, legal, or administrative barriers, though limited implementation may be possible.",
        "Probably yes": "Implementation is probably feasible in the stated setting; remaining infrastructure, workforce, training, coordination, legal, or administrative barriers appear manageable.",
        "Yes": "Implementation is clearly feasible with the available infrastructure, workforce, training, coordination, legal arrangements, and administrative capacity.",
    },
}


GENERIC_DEFINITIONS = {
    "Varies": "The appropriate judgment genuinely differs across important populations, settings, outcomes, or implementation conditions.",
    "Don't know": "The available information is insufficient to select any substantive option; do not use this merely because some uncertainty exists.",
}


CRITERION_DECISION_RULES: dict[str, str] = {
    "Desirable Effects": "Judge magnitude, not certainty. Prefer absolute effects for each critical outcome; apply explicit outcome- and context-specific thresholds when supplied. Relative effects and statistical significance alone do not determine magnitude.",
    "Undesirable Effects": "Judge magnitude, not certainty. Include harms and treatment burden; prefer absolute effects for each critical outcome and apply context-specific thresholds when supplied.",
    "Certainty of evidence": "Use the overall certainty for critical outcomes that drive the decision, based on GRADE domains and whether the true effect could cross a decision threshold. Do not lower the magnitude judgment merely because certainty is low.",
    "Values": "Judge uncertainty or variability in the relative importance of outcomes, not variability in treatment effects or baseline risks.",
    "Balance of effects": "Integrate desirable and undesirable health effects, their importance, and certainty. Do not include resource use unless the framework explicitly combines it here.",
    "Resources required": "Use incremental resource consequences relative to the comparator, the stated perspective, time horizon, and opportunity costs. Do not treat acquisition price alone as total resource use.",
    "Certainty of evidence of required resources": "Rate certainty of the resource-use evidence, not whether resource use is high or low. If no included study reports usable resource evidence, choose 'No included studies'.",
    "Cost effectiveness": "Use comparative costs and effects relative to a context-appropriate threshold; cost saving alone is not automatically cost effective if health outcomes worsen.",
    "Equity": "Judge effects on unfair health differences, especially for disadvantaged groups; a uniform average benefit does not by itself imply increased equity.",
    "Acceptability": "Assess acceptance by all key stakeholder groups and distinguish acceptability from practical implementability.",
    "Feasibility": "Assess whether implementation can work in the stated setting; distinguish feasibility from stakeholder acceptance and affordability.",
}


def format_judgment_definitions(criterion: str, options) -> str:
    """Return definitions only for the options offered in the current record."""
    definitions = CRITERION_JUDGMENT_DEFINITIONS.get(criterion, {})
    lines = []
    decision_rule = CRITERION_DECISION_RULES.get(criterion)
    if decision_rule:
        lines.append(f"Decision rule: {decision_rule}")
    for option in options:
        text = definitions.get(option) or GENERIC_DEFINITIONS.get(option)
        if text:
            lines.append(f"- {option}: {text}")
    return "\n".join(lines)
