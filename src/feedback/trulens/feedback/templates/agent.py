"""
Agentic evaluation templates: logical consistency, execution
efficiency, plan adherence, plan quality, tool selection,
tool calling, tool quality.
"""

from inspect import cleandoc
from typing import ClassVar

from trulens.feedback.templates.base import LIKERT_0_3_PROMPT
from trulens.feedback.templates.base import CriteriaOutputSpaceMixin
from trulens.feedback.templates.base import OutputSpace
from trulens.feedback.templates.base import Semantics
from trulens.feedback.templates.base import WithPrompt


class LogicalConsistency(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the logical consistency of the agentic
    system's plan and execution.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    Score the logical consistency of the trace, including both the plan and execution.

    {max_score}: Every action, claim, and transition in the trace is explicitly justified using information available in the prior context. Each statement is directly supported by and traceable to previous data, instructions, or content—no part of the response is fabricated or inferred from unstated assumptions. If an error from an earlier step is identified and corrected, the error is explicitly acknowledged before the correction is made, maintaining logical transparency. Each system instruction is followed. The reasoning remains coherent and free of contradictions or logical leaps.

    Middle scores: There are occasional lapses in logic, minor unsupported assertions, or isolated explanatory gaps. Errors may be corrected, but corrections are occasionally introduced without clear acknowledgement of prior mistakes, creating minor inconsistencies or reducing transparency. Some statements may not be fully traceable to prior context, or some assumptions are made without explicit support from available evidence. Factual consistency may suffer from minor errors or embellishments, but the overall reasoning remains intact. Most previously assigned tasks and instructions remain intact.

    {min_score}: There is frequent or severe breakdown in the logical flow; many statements are either unsupported by, or cannot be grounded in, the prior context. Corrections for earlier errors are often made without any explicit acknowledgement, resulting in contradictions or confusing transitions. Key actions or facts are invented, fabricated, or otherwise not observable in the given information. Major contradictions, invalid assumptions, or arbitrary transitions undermine the overall reasoning and conclusion. Most previously assigned tasks are not fulfilled, and internal system instructions are largely disregarded.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous and analytical LOGICAL CONSISTENCY evaluator: provide a score for the logical consistency given an agentic system's trace.

        You must assign a single numerical score from {output_space_prompt}.

        Evaluation criteria:
        {criteria}
        {additional_instructions}

        Be critical in your evaluation. For each step in the trace with an issue (eg. contradictions, unsupported statements, or previous instructions not followed), identify that step and explain the problem specifically. Flag any implicit assumptions.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}

        LOGICAL CONSISTENCY SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class ExecutionEfficiency(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the efficiency of the agentic system's
    execution.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    Score the efficiency of the execution.

    {max_score}: All relevant actions are executed exactly once, in a streamlined and optimized sequence. There is no unnecessary busywork, repetition, backtracking, or wasted computation/resources. Each step genuinely contributes to progressing towards the goal without extraneous operations. Error handling is appropriately lean and resolves quickly, without requiring multiple attempts due to easily correctable input errors (e.g., incorrect tool arguments). Verification steps provide unique feedback, serve as sanity checks, or use a demonstrably different approach from the initial approach to ensure correctness, without duplicating prior effort.

    Middle scores: Some instances of workflow inefficiency such as redundant actions, non-ideal ordering of steps that cause rework, excessive error handling, missed opportunities for consolidation, or unnecessary resource use. There might be occasional minor input errors or misconfigurations that lead to a slightly increased number of attempts but are eventually corrected without major disruption. The inefficiencies may have noticeable but not devastating impact on the overall process.

    {min_score}: Workflow is highly inefficient: dominated by loops, duplicated efforts, poorly ordered sequence, or significant wasted computation that break progress. Multiple repeated tool calls required to recover from preventable mistakes in invocation or argument generation. Verification steps are highly redundant and do not provide any value. The workflow's operational flow is severely hampered by unnecessary or counterproductive actions.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous and analytical EXECUTION EFFICIENCY evaluator: provide a score for how efficiently the agent executes its steps. Your assessment should strictly focus on the sequencing, resource utilization, and avoidance of redundant or wasteful actions within the execution itself, regardless of whether the plan was ultimately successful or fully adhered to.

        You must assign a single numerical score from {output_space_prompt}.

        Evaluation criteria:
        {criteria}
        {additional_instructions}

        Evaluation steps to give feedback on key steps in the execution are allowed. Otherwise, be critical in your evaluation. For each step in the execution trace with an issue (e.g., redundancies, unnecessary retries, inefficient sequencing, missed optimization opportunities, or preventable errors), identify that step and explain the problem specifically.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}

        EXECUTION EFFICIENCY SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class PlanAdherence(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the adherence of the agentic system's
    execution to the agentic system's plan.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    Score the adherence of the execution to the plan.

    {max_score}: Each step in the plan was executed and completed correctly and in entirety. No steps were skipped, reordered, or modified without explicit reasoning. Any deviations from the plan were explicitly justified and directly attributable to unforeseen, external factors. If replanning was necessary, the revised plan was followed exactly.

    Middle scores: Most steps in the plan were faithfully executed and completed as intended. Minor deviations from the plan or partial step completions have plausible explanations or can be easily inferred from context. If replanning was necessary, the revised plan was generally followed.

    {min_score}: Multiple planned steps were omitted, performed out of order, or replaced with unplanned actions. No meaningful attempt was made to explain, justify, or document plan changes or new actions. The plan was largely ignored or disregarded in execution, or steps were not completed as intended. If replanning was necessary, the revised plan was not followed.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous and analytical PLAN ADHERENCE evaluator: you are given the entire trace which contains both the plan and the execution. First, identify the plan and any subsequent replans within the trace. Then, evaluate how closely the execution follows the plan or replans.
        You must assign a single numerical score from {output_space_prompt}.

        Plan Extraction Procedure:
        1. Scan for the sections labeled with a PLAN keyword. The first section labeled with a PLAN keyword is the initial plan, and any subsequent section labeled with a PLAN keyword is a replan.
        2. If no explicitly labeled PLAN section exists, infer the plan from any 'Thinking' or planning sections [or to-do checklist].
        3. If no plan can be found through the above steps, output: "I cannot find a plan."
        Do NOT infer or fill gaps using execution steps.

        You MUST structure your entire response using the following markdown template:
        -----
        **Plan Identification**
        [Paste initial plan or state: 'I cannot find a plan.']

        **Plan Adherence Analysis**
        [Analyze how the agent followed the initial plan. Note each deviation leading up to the first replan (if any).]

        For each replan (if exists):
        **Replan Identification:**
        [Paste the replan.]

        **Replan Adherence Analysis:**
        [Analyze how the agent followed the new replan. Note each deviation leading up to the next replan (if any).]
        -----

        Evaluation criteria:
        {criteria}
        {additional_instructions}
        Adherence is judged step-by-step; if a plan mandates tool usage or sub-tasks, their omission or incomplete execution always counts as a failure of adherence, regardless of the effect on final output completeness or quality. Be critical in your evaluation and focus on identifying any deviations from the plan or any steps that were not completed as intended. For each identified deviation from the plan, cite the associated execution steps (or lack thereof) and explain the problem specifically.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}

        PLAN ADHERENCE SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class PlanQuality(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the quality of the agentic system's plan to
    address the user's query.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    Score the quality of the plan.

    {max_score}: The plan is well-structured, optimal, and directly addresses the user's query by breaking it down into clear, actionable, and logical steps. Every step is justified, necessary, and includes sufficient detail to ensure feasibility and efficiency without being overly verbose. Each step in the plan could be feasibly executed by the tools provided. If replanning occurs, the revised plan is presented with an explicit rationale. The replan is a direct and effective response to the observed triggers (e.g., errors, new information) and learns from prior attempts by not repeating problematic steps.

    Middle scores: The plan generally addresses the query and appears feasible. Minor issues may be present: some steps lack explicit justification, a few steps may be unnecessary or unclear, or non-critical actions may be missing. The step order or rationale might be partially implied rather than fully articulated. Most steps in the plan could be feasibly executed by the tools provided. If replanning occurs, the rationale is vague or weakly connected to the trigger. The replan partially addresses the trigger but may be inefficient or repeats minor errors from the previous plan.

    {min_score}: The plan fails to directly address the user's query or cannot feasibly accomplish the goal. Critical steps in the plan are missing, irrelevant, unsupported, or based on fabricated reasoning. Replanning (if any) is arbitrary, unexplained, or disconnected from observable evidence in prior context. The overall plan lacks adequate justification and transparency, with major gaps or unjustified assertions. Many steps in the plan cannot be feasibly executed by the tools provided. If replanning occurs, it is arbitrary, unexplained, or disconnected from any trigger. The replan fails to address the issue and repeats the same critical mistakes as the previous attempt.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous and analytical PLAN QUALITY evaluator. You are responsible for evaluating the intrinsic quality of the initial written plan, judging it against the context and tools available at the moment of its creation. CRITICAL: It is an immediate failure of your task to reference whether the agent followed the plan or mention any part of the execution, including agent actions, tool outputs, or the final answer.

        Plan Extraction Procedure:
        1. Scan for the sections labeled with a PLAN keyword. The first section labeled with a PLAN keyword is the initial plan, and any subsequent section labeled with a PLAN keyword is a replan.
        2. If no explicitly labeled PLAN section exists, infer the plan from any 'Thinking' or planning sections [or to-do checklist].
        3. If no plan can be found through the above steps, output: "I cannot find a plan."
        Do NOT infer or fill gaps using execution steps.

        Evaluating the Initial Plan:
        1. The Available Tools: Does the plan correctly select from the list of provided tools? Does it ignore a more appropriate or efficient tool that was available? Does it try to use a tool that doesn't exist?
        2. Tool Definitions: Does the plan propose using a tool correctly, according to its description and required arguments?
        3. Pre-existing Knowledge: Does the plan include redundant steps to find information that was already present in the initial prompt or conversation history? Does the plan include relevant information from fact-finding or exploration prior to planning?
        4. An optimal plan isn't just logical in theory; it's the most intelligent strategy given the specific resources the planner had.
        When evaluating the initial plan, ignore all execution steps, tool outputs, and agent actions, even if available and visible in the trace. Your quality evaluation for this initial plan MUST be based solely on its intrinsic quality. You are judging the strategy, not the outcome. Never use agent choices, answers, or deviations from the plan to deduce flaws, gaps, or weaknesses in the plan itself.

        Replanning (if found):
        1. Look at the tool outputs, error messages, or observations in the trace that precede the replan to understand why replanning was necessary.
        2. Identify the trigger and explain why the original plan was insufficient. Is the reason for replanning justified?
        3. Judge the new plan. Are they a logical, necessary, and efficient correction to the specific problem identified in the trigger? You are not judging the original failure itself, but the quality of the agent's reaction to that failure.

        List only inherent plan flaws (e.g., step uses nonexistent tool, redundant action, ignores key context).
        You MUST structure your entire response using the following markdown template:
        -----
        **Initial Plan Identification**
        [Paste initial plan or state: 'I cannot find a plan.']

        For each replan (if exists):
        **Replan Identification**
        [Paste each replan. For each replan, state the written rationale/explanation.]

        **Plan Quality Analysis**
        [Analysis solely on plan/replan text and rationale.]

        **Verdict on Plan Flaws**
        [List only actual flaws in the plans themselves.]
        -----
        You must assign a single numerical score from {output_space_prompt} based SOLELY on the intrinsic quality of the plan and replans. Do NOT score on the execution quality.

        Evaluation criteria:
        {criteria}
        {additional_instructions}

        Be critical in your evaluation. For each step in the plan that is not necessary, unclear, or unsupported, identify that step and explain the problem specifically.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}

        PLAN QUALITY SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class ToolSelection(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the agent's *choice of tools* for its
    tasks/subtasks given tool descriptions.
    Mapped to PLAN (lower-level complement to Plan Quality).
    Excludes execution efficiency and adherence; focuses on
    suitability of selection.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    criteria_template: ClassVar[str] = """
    Score the appropriateness of tool SELECTION decisions relative to stated goals and available tools.
    {max_score}: Consistently selects the most suitable tools for each subtask, honors mandated tools, avoids tools when internal reasoning suffices, and reflects awareness of tool capabilities/limits.
    Middle scores: Generally appropriate selections with occasional missed opportunities (better tool existed), unnecessary tool choices for internal tasks, or weak justification.
    {min_score}: Frequently selects ill-suited/irrelevant tools, ignores mandated tools, or bypasses obviously superior tools; relies on non-tools where a tool is necessary.
    Consider: match-to-goal, comparative suitability, instruction compliance, and awareness of constraints. Do NOT judge call syntax, output interpretation, efficiency, or adherence.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous TOOL SELECTION evaluator. Judge whether the agent chose the right tools for its tasks given the tool descriptions.
        You must assign a single numerical score from {output_space_prompt}.
        Evaluation criteria:
        {criteria}
        {additional_instructions}
        Important scope boundaries:
        - Do NOT penalize call syntax/semantics or output interpretation (Tool Calling).
        - Do NOT penalize workflow efficiency (Execution Efficiency) or plan deviations (Plan Adherence).
        - Focus strictly on selection quality per subtask.
        Be critical. For each selection issue, cite the relevant spans and explain specifically.
        You must structure your response exactly as specified in the provided tool_selection_prompt.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}
        TOOL SELECTION SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class ToolCalling(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the agent's *tool invocation quality* that is
    within the agent's control: argument validity/completeness,
    semantic appropriateness, preconditions/postconditions,
    and output interpretation.
    Mapped to ACT (specialized complement to Plan Adherence).
    Excludes selection and efficiency.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    criteria_template: ClassVar[str] = """
    Score the quality of TOOL CALLS within the agent's control.
    {max_score}: Inputs are syntactically valid and semantically appropriate; required params and preconditions are satisfied; outputs are interpreted faithfully and integrated correctly; tool-returned errors are acknowledged and handled reasonably.
    Middle scores: Minor issues with argument completeness, semantic underspecification, limited reformulation, or shallow/partial output use; some missed acknowledgements of errors.
    {min_score}: Invalid/missing arguments, repeated schema violations, semantically off-target queries without correction; outputs ignored/misread/fabricated; tool errors unacknowledged.
    Consider only what is under the agent's control. Do NOT judge tool choice (Tool Selection), workflow efficiency, or external system reliability (Tool Quality).
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous TOOL CALLING evaluator. Judge how well the agent formed tool inputs and interpreted outputs, given tool definitions.
        You must assign a single numerical score from {output_space_prompt}.
        Evaluation criteria:
        {criteria}
        {additional_instructions}
        Important scope boundaries:
        - In-scope: argument/schema correctness, semantic fit of query, preconditions/postconditions, grounded interpretation of outputs, explicit handling of tool-returned errors.
        - Out-of-scope: tool selection (Tool Selection), workflow efficiency (Execution Efficiency), external service/tool reliability (Tool Quality).
        Be critical. For each calling issue, cite the relevant spans and explain specifically.
        You must structure your response exactly as specified in the provided tool_calling_prompt.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}
        TOOL CALLING SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class ToolQuality(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    Evaluates the *tool/system side* quality and reliability
    observed in the trace (external errors, availability,
    stability, domain-specific output quality like search
    relevance).
    Independent of agent behavior; complements GPA by
    isolating tool-side failures.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name

    criteria_template: ClassVar[str] = """
    Score the QUALITY/RELIABILITY of tools as observed, independent of agent choices.
    {max_score}: Tools respond reliably with relevant/complete outputs; no or rare external errors (5xx/4xx/429), no unexplained timeouts; domain quality (e.g., search relevance) is consistently strong.
    Middle scores: Occasional external errors or weak outputs; intermittent relevance/latency issues; overall usable but flaky.
    {min_score}: Frequent external errors, timeouts, rate limits, auth failures, or persistently poor domain output quality.
    Consider only external/tool-side quality given the inputs. If inputs are clearly invalid, note it but do not penalize Tool Quality (penalize under Tool Calling).
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a meticulous TOOL QUALITY evaluator. Judge external/tool-side reliability and output quality observed in the trace.
        You must assign a single numerical score from {output_space_prompt}.
        Evaluation criteria:
        {criteria}
        {additional_instructions}
        Important scope boundaries:
        - In-scope: service errors (5xx), rate limiting (429), auth (401/403), resource not found (404), timeouts, flakiness, determinism, and domain-specific output quality (e.g., search relevance).
        - Out-of-scope: agent's selection, argument formation, or workflow efficiency.
        Be critical. For each tool quality issue, cite the relevant spans and explain specifically.
        You must structure your response exactly as specified in the provided tool_quality_prompt.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """{trace}
        TOOL QUALITY SCORE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )
