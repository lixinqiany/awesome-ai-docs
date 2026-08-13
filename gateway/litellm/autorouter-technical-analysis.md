# LiteLLM Autorouter: Technical Analysis

> An implementation-oriented analysis of LiteLLM's complexity-aware autorouting pipeline.

## Document Scope

This document explains how LiteLLM's autorouter combines complexity classification, deterministic overrides, and adaptive model selection. It focuses on the routing decisions made before a completion is sent to a model and the feedback loop that updates adaptive routing after a response.

The terminology and behavior described here follow the current LiteLLM implementation. The four canonical complexity tiers are `SIMPLE`, `MEDIUM`, `COMPLEX`, and `REASONING`.

## Contents

- [1. Routing Architecture](#1-routing-architecture)
- [2. Complexity-Tier Configuration](#2-complexity-tier-configuration)
- [3. Heuristic Classification](#3-heuristic-classification)
- [4. LLM-Based Classification](#4-llm-based-classification)
- [5. Keyword and Semantic Matching](#5-keyword-and-semantic-matching)
- [6. Escalation Keyword Matching](#6-escalation-keyword-matching)
- [7. Adaptive Routing](#7-adaptive-routing)
- [8. Bandit Initialization and Persistence](#8-bandit-initialization-and-persistence)
- [9. Post-Call Feedback and Updates](#9-post-call-feedback-and-updates)
- [10. End-to-End Decision Flow](#10-end-to-end-decision-flow)
- [11. Design Trade-offs](#11-design-trade-offs)

## 1. Routing Architecture

LiteLLM's complexity router separates the routing problem into two concerns:

1. **Determine the effective complexity tier.** A request can be classified by a local heuristic scorer or by a configured LLM classifier. Keyword and semantic rules may override that classification, and an escalation phrase may promote it afterward.
2. **Choose a model from the resulting candidate set.** The router can select a configured model directly, choose randomly from a tier pool, or use adaptive bandit selection when adaptive routing is enabled.

Complexity-tier configuration defines the model pools used by the Complexity Router, also known as Auto-Router v2. Each incoming request is assigned to one of four ordered tiers, and each tier can contain one model or a pool of candidate models.

The default path is local and deterministic. Optional LLM classification, embedding-based matching, and adaptive selection add capability at the cost of additional latency, operational state, or model calls.

## 2. Complexity-Tier Configuration

The tier map is the foundation of the router. A simple deployment can assign one model to each tier:

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: gpt-4o-mini
          MEDIUM: gpt-4o
          COMPLEX: claude-sonnet-4
          REASONING: o1-preview
```

### 2.1 Canonical Tiers

The tiers are ordered from lowest to highest expected difficulty:

| Tier | Intended workload |
| --- | --- |
| `SIMPLE` | Greetings, short factual lookups, and requests with a known answer |
| `MEDIUM` | Everyday requests that need explanation, light reasoning, or minor technical work |
| `COMPLEX` | Non-trivial code, architecture, multi-part technical work, or specialized domain depth |
| `REASONING` | Open-ended analysis, proofs, explicit step-by-step reasoning, and difficult trade-offs |

### 2.2 Tier Pools

A tier may point to a single model or to a list of models. With adaptive routing disabled, a list is treated as a pool from which LiteLLM selects a model. With adaptive routing enabled, the pool becomes the tier's home set for adaptive selection.

```yaml
complexity_router_config:
  tiers:
    SIMPLE:
      - gpt-4o-mini
      - claude-haiku
    MEDIUM: gpt-4o
    COMPLEX:
      - claude-sonnet-4
      - gpt-4o
    REASONING: o1-preview
```

### 2.3 Display Labels

Deployments can assign operator-facing labels without changing the canonical configuration keys:

```yaml
complexity_router_config:
  tier_labels:
    SIMPLE: Cheap
    MEDIUM: Standard
    COMPLEX: Premium
    REASONING: Deep
```

Labels are used in dashboards, spend logs, and the LLM classifier rubric. The canonical values remain stable in configuration and routing records, so historical records remain comparable after a label change. The heuristic scorer does not use labels when calculating a score.

## 3. Heuristic Classification

### 3.1 Local Scoring

The default classifier is a deterministic, rule-based heuristic. It makes no external LLM calls, works offline, and is designed for very low latency. Each request is evaluated across seven dimensions, and the normalized weighted sum is mapped to one of the four tiers.

The default token estimate uses an average of four characters per token. This is a coarse signal rather than a provider-specific tokenizer calculation.

### 3.2 Scoring Dimensions

| Dimension | Scoring behavior | Weight |
| --- | --- | ---: |
| Token count | Below 15 tokens: `-1`; 15 to 400 tokens: `0`; above 400 tokens: `1` | `0.10` |
| Code presence | No matches: `0`; one match: `0.5`; at least two matches: `1` | `0.30` |
| Reasoning markers | No matches: `0`; one match: `0.7`; at least two matches: `1` | `0.25` |
| Technical terms | Fewer than two matches: `0`; two or three matches: `0.5`; at least four matches: `1` | `0.25` |
| Simple indicators | No matches: `0`; at least one match: `-1` | `0.05` |
| Multi-step patterns | No match: `0`; at least one match: `0.5` | `0.03` |
| Question complexity | At most three question marks: `0`; more than three: `0.5` | `0.02` |

The main signal sources are content features rather than raw length. Code and reasoning signals carry the largest weights, while simple indicators, explicit multi-step structure, and question count act as smaller corrections.

### 3.3 Tier Boundaries

The weighted score is mapped using three configurable boundaries:

| Tier | Default score range | Boundary below the tier |
| --- | --- | --- |
| `SIMPLE` | `< 0.15` | None |
| `MEDIUM` | `0.15` to `< 0.35` | `simple_medium` |
| `COMPLEX` | `0.35` to `< 0.60` | `medium_complex` |
| `REASONING` | `>= 0.60` | `complex_reasoning` |

```yaml
complexity_router_config:
  tier_boundaries:
    simple_medium: 0.15
    medium_complex: 0.35
    complex_reasoning: 0.60
```

The boundary keys retain their canonical names even when a deployment renames the visible tiers.

### 3.4 Reasoning Override

The heuristic path contains an explicit reasoning override. If the user message contains at least two reasoning markers, the request is assigned to `REASONING` regardless of the weighted score. This prevents a short prompt such as an explicit request to reason through a difficult problem from being treated as simple solely because it has few tokens.

Reasoning markers in the caller's system prompt do not trigger this override. This keeps a general instruction such as “think carefully before answering” from forcing every request into the highest tier.

### 3.5 Language Sensitivity

The built-in keyword lists are predominantly English. The local classifier can therefore lose accuracy on non-English requests unless operators customize the keyword lists and validate the result against representative traffic. Token count remains a coarse language-independent signal, but lexical dimensions are language-sensitive.

## 4. LLM-Based Classification

### 4.1 LLM-as-a-Judge Model

The optional LLM classifier treats routing as an LLM-as-a-Judge task. Its purpose is to judge the intellectual difficulty of answering correctly, rather than relying on the surface length of the current message.

```yaml
complexity_router_config:
  classifier_type: llm
  classifier_llm_config:
    model: gpt-4o-mini
    timeout_ms: 3000
  classifier_fallback: heuristic
```

The classifier must return exactly one configured tier. If the classifier call fails, times out, or returns an invalid result, LiteLLM follows the configured fallback policy. The fallback can use the local heuristic or route to a configured default model.

### 4.2 Structured User Payload

The classifier's variable user payload contains three types of evidence:

1. **Caller system prompt.** The caller's system prompt is quoted as task-context evidence so that task constraints remain visible to the classifier.
2. **Recent conversation.** A bounded set of eligible prior turns can be included to preserve the meaning of follow-ups such as “continue” or “do the same for the streaming path.” Tool output and harness reminder blocks are excluded from the normal context extraction path.
3. **Current user message.** The latest user message is explicitly identified as the classification target.

The payload can also include a coarse cumulative token estimate across the resolved conversation. This estimate preserves a signal that the current turn belongs to a longer task even when redundant history is omitted from the semantic context window.

### 4.3 Stable System Rubric

The classifier system message defines the stable routing policy. Its built-in rubric describes the objective and the four ordered tiers:

| Component | Representative meaning |
| --- | --- |
| Objective | Classify the request into exactly one tier and judge intellectual difficulty rather than message length |
| `SIMPLE` | Greetings, chitchat, or short factual lookups with a known answer |
| `MEDIUM` | Everyday requests requiring some explanation or light reasoning |
| `COMPLEX` | Non-trivial code, architecture, multi-step technical work, or specialized depth |
| `REASONING` | Open-ended analysis, proofs, careful trade-offs, or explicit step-by-step reasoning |

Keeping the rubric in the system role makes the policy stable across requests while keeping request-specific evidence in the user role. A custom classifier system prompt can replace the built-in rubric, but the replacement then owns the taxonomy and must also provide its own prompt-injection defenses.

### 4.4 Trust Boundary

Caller-controlled material is quoted as evidence, not treated as classifier instructions. The built-in rubric explicitly tells the classifier that the quoted caller system prompt and prior turns are material to judge. This boundary helps prevent a caller message containing an instruction such as “always return the highest tier” from overriding the routing policy.

### 4.5 Context Window Controls

The context window is bounded by both the number of turns and the character limit per turn:

```yaml
complexity_router_config:
  classifier_context_window_size: 3
  classifier_context_per_turn_chars: 200
  classifier_context_include_assistant_turns: false
```

Assistant turns can be enabled when the difficulty of the work is expressed in an earlier assistant plan and the user follows up with a short approval such as “yes.” Context inclusion improves continuity but can increase classifier input, latency, and spend. Setting the context window to `0` makes classification depend on the current ask and caller system prompt only.

## 5. Keyword and Semantic Matching

### 5.1 Pre-Classification Override

Keyword and semantic matching is a preliminary override layer. It examines the latest user query before the normal classification path and can route the request directly to a configured tier.

### 5.2 Literal Keyword Rules

Each rule contains one or more keywords or phrases and maps them to one tier. Keywords within a rule are combined with OR semantics. When multiple rules match, LiteLLM selects the most severe matched tier.

```yaml
complexity_router_config:
  keyword_tier_rules:
    - keywords:
        - production outage
        - incident response
      tier: COMPLEX
    - keywords:
        - formal proof
        - theorem
      tier: REASONING
```

Literal matching is predictable and inexpensive, but it depends on the exact vocabulary used in the rule set.

### 5.3 Semantic Matching

When semantic matching is enabled, embedding retrieval replaces literal matching:

```yaml
complexity_router_config:
  semantic_keyword_matching: true
  embedding_model: text-embedding-3-small
  match_threshold: 0.5
  keyword_tier_rules:
    - keywords:
        - production outage
        - incident response
      tier: COMPLEX
```

Rules targeting the same tier are merged into one semantic route, while each configured phrase remains a separate utterance with its own embedding. For a tier, the route score is the maximum cosine similarity between the query and any utterance assigned to that tier. The highest-scoring tier is selected only when its score meets the configured threshold, which defaults to `0.5`.

If embedding generation or retrieval fails, the stage follows the configured fallback path and continues to classification. It does not silently convert a semantic rule into a literal rule.

### 5.4 Ordering with Other Stages

The effective order is:

1. Resolve the latest real user message and remove configured harness reminder blocks.
2. Apply literal or semantic keyword-tier rules.
3. If no override matches, run the configured heuristic or LLM classifier.
4. Apply escalation keyword matching to the resulting tier.
5. Select a model from the final candidate set.

This ordering lets operators express high-confidence routing policies without forcing every request through a more expensive classifier.

## 6. Escalation Keyword Matching

### 6.1 Post-Decision Promotion

Escalation keyword matching is different from keyword-tier matching. It does not choose a target tier. Instead, after a classifier or keyword rule produces a base tier, an escalation phrase promotes that result by one step. A request already in `REASONING` remains in `REASONING` because no higher tier exists.

The default escalation phrase is `LITELLM ESCALATE`. Operators can provide their own phrases or disable the feature with an empty list.

```yaml
complexity_router_config:
  escalation_keywords:
    - NEED A STRONGER MODEL
    - LITELLM ESCALATE
```

### 6.2 Matching Semantics

Escalation matching is case-sensitive and uses substring detection on the most recent user query. It allows a caller to request a stronger tier without selecting a specific provider or model. This is useful as a controlled retry or self-service escalation mechanism when the initial answer is not sufficient.

## 7. Adaptive Routing

### 7.1 Request-Type-Conditioned Selection

Adaptive routing is a model-selection layer that runs after the effective complexity tier has been determined. It is not a replacement classifier. The adaptive router first classifies the request into one of seven built-in request types:

| Request type | Typical examples |
| --- | --- |
| `CODE_GENERATION` | Writing new functions, modules, or applications |
| `CODE_UNDERSTANDING` | Explaining, reviewing, or debugging existing code |
| `TECHNICAL_DESIGN` | Architecture, interfaces, infrastructure, and system design |
| `ANALYTICAL_REASONING` | Comparisons, deductions, proofs, and deep analysis |
| `WRITING` | Drafting, rewriting, editing, and style transformation |
| `FACTUAL_LOOKUP` | Short questions with a known factual answer |
| `GENERAL` | Requests that do not fit another category |

Rather than maintaining one global ranking of models, LiteLLM maintains a separate bandit cell for each `(request_type, model)` pair. A model can therefore learn to perform well for code generation without being assumed to be equally strong for writing or factual lookup.

### 7.2 Thompson Sampling

For each eligible model, the router samples a quality estimate from its Beta posterior. It then combines that estimate with a normalized cost score and selects the model with the highest combined score.

The generic selection score is:

```text
selection_score = quality_weight * sampled_quality
                  + cost_weight * normalized_cost
```

The default adaptive-router weights are quality `0.7` and cost `0.3` in the standalone adaptive router. Complexity-router adaptive mode has its own configurable weights and may apply a tier-distance penalty.

Thompson sampling is stochastic by design. It balances exploitation of models with strong observed posteriors and exploration of models that still have uncertainty.

### 7.3 Cost Normalization

Model costs are normalized with a max-min transformation into the interval `[0, 1]`. The cheapest eligible model receives the highest cost score, while the most expensive model receives the lowest score. When all eligible models have the same cost, the cost score defaults to `0.5`.

### 7.4 Complexity Floors and Tier Distance

Adaptive selection can operate in two modes:

- **Classified-tier eligibility.** Thompson sampling considers only the models in the classified tier's pool.
- **All-pool eligibility.** Models across pools can compete, but a model farther from the classified tier receives a tier-distance penalty.

The second mode acts as a soft floor. It can discover a cheaper model above or below the classified tier when the quality and cost trade-off justifies it, while still discouraging large departures from the classifier's decision.

## 8. Bandit Initialization and Persistence

### 8.1 Beta Posterior Cells

Each `(request_type, model)` cell is represented as a Beta distribution:

```text
Beta(alpha, beta)

posterior_mean = alpha / (alpha + beta)
```

Here, `alpha` is pseudo-success evidence and `beta` is pseudo-failure evidence. The posterior mean is the current expected quality for that model and request type.

### 8.2 Cold-Start Priors

When an adaptive router is created, every cell receives a non-zero cold-start prior. The prior mean is derived from the model's declared quality tier, with an additional strength bonus when the model declares a matching request type. The prior is capped to avoid excessive confidence, and its total mass is initialized to `10`, allowing roughly ten real observations to move the estimate materially.

Example model metadata:

```yaml
model_info:
  adaptive_router_preferences:
    quality_tier: 3
    strengths:
      - code_generation
      - analytical_reasoning
```

### 8.3 Database State

Persisted bandit state is loaded from the database and overrides the corresponding cold-start cells. Missing cells continue to use their cold-start priors. Runtime updates are aggregated in memory and flushed asynchronously, so persistence is eventually consistent rather than part of the request's critical path.

The state update queue applies atomic increments for bandit deltas, which avoids read-modify-write races when multiple proxy workers flush updates concurrently. Session snapshots use last-write-wins upserts.

## 9. Post-Call Feedback and Updates

### 9.1 Implicit Feedback

Adaptive routing learns from lightweight signals after a call. The current user turn can provide feedback about the previous response, while response and tool signals are attributed to the model serving the current turn.

Positive follow-ups such as “thanks” or “that worked” contribute positive evidence. Misalignment, stagnation, disengagement, and failures contribute negative evidence. Repeated tool loops contribute weaker negative evidence, while exhaustion signals are tracked separately from quality.

### 9.2 Signal-to-Posterior Mapping

The current v0 mapping is:

| Signal | Posterior update |
| --- | --- |
| Satisfaction | `alpha += 1` |
| Misalignment | `beta += 1` |
| Stagnation | `beta += 1` |
| Disengagement | `beta += 1` |
| Failure | `beta += 1` |
| Repeated tool loop | `beta += 0.5` |
| Exhaustion | No quality update |

As `beta` grows, the posterior mean and usually the Thompson samples for that cell decrease. The model is not permanently excluded, however, because sampling remains stochastic and can still explore a degraded or uncertain cell.

### 9.3 Session Attribution

Session context records which model and request type produced the previous response. This allows feedback from a short follow-up to be attributed to the correct earlier model instead of the model selected for the current turn. For a `GENERAL` follow-up such as “okay” or “thanks,” the previous request type can be retained to avoid attributing feedback to an unrelated category.

The in-memory session state is bounded and expires after a configured owner-cache TTL. Sensitive message text and tool-history fields are not included in the persisted session snapshot.

### 9.4 Operational Limits

The v0 adaptive implementation has several intentional limits:

- The bandit sample mass has a hard cap of `200`; updates beyond the cap are dropped rather than rescaled.
- Signal detection uses bounded local state and lightweight regex and tool-call detectors rather than an external judge.
- Quality and cost are scored, but latency is not currently part of the selection score.
- The request taxonomy is fixed to seven built-in types.

These constraints keep the hot path small and predictable, but operators should validate the signal mapping and priors against real traffic before treating the posterior as a calibrated quality metric.

## 10. End-to-End Decision Flow

The complete decision path can be summarized as follows:

```text
Incoming request
       |
       v
Extract the latest real user message
       |
       v
Strip configured harness reminder blocks
       |
       v
Literal or semantic keyword-tier override?
       | yes                         | no
       v                            v
Use matched tier             Heuristic or LLM classification
       |                            |
       +-------------+--------------+
                     v
              Apply escalation phrase
                     |
                     v
             Resolve eligible models
                     |
          Adaptive selection enabled?
                | yes          | no
                v              v
       Thompson sample and   Pick from final
       apply cost/tier score  tier pool
                |              |
                +------+-------+
                       v
                 Send completion
                       |
                       v
          Record bounded post-call signals
                       |
                       v
          Update bandit cells and flush state
```

The routing decision record preserves provenance such as the selected tier, score, boundaries, matching signals, escalation phrase, classifier model, and classifier cost when those facts apply. This makes routing behavior inspectable in spend and operational logs.

## 11. Design Trade-offs

### 11.1 Determinism versus Semantic Coverage

The heuristic classifier is inexpensive, deterministic, and easy to debug, but its lexical coverage is limited. The LLM classifier handles context and nuanced difficulty better, but introduces model-call latency, spend, failure modes, and prompt-injection concerns.

### 11.2 Explicit Policy versus Learned Behavior

Keyword-tier rules and escalation phrases give operators direct control over high-confidence cases. Adaptive routing learns from observed outcomes and can discover request-type-specific strengths, but its decisions are stochastic and depend on the quality of implicit feedback.

### 11.3 Cost versus Context

Including prior turns and assistant messages helps classify short follow-ups correctly, but it increases classifier input and may expose more conversation context to the classifier deployment. Bounded windows and per-turn truncation provide a tunable compromise.

### 11.4 Exploration versus Stability

Thompson sampling preserves exploration, which prevents an early mistake from permanently excluding a model. The same property means identical requests can be routed to different models over time. Operators that need stable provider affinity can enable session affinity or restrict adaptive eligibility to the classified tier.

## Summary

LiteLLM's autorouter is best understood as a layered control system rather than one classifier. Local complexity scoring supplies a fast baseline, LLM classification adds contextual judgment, keyword and semantic rules express explicit policy, escalation provides a controlled promotion path, and adaptive routing learns request-type-specific model quality over time. The resulting architecture balances cost, quality, latency, and operator control while keeping each layer independently configurable.
