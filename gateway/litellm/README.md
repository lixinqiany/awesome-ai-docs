# LiteLLM

This directory collects technical analyses and implementation notes about [LiteLLM](https://github.com/BerriAI/litellm), an open-source LLM gateway and Python SDK.

## Documents

- [LiteLLM Autorouter: Technical Analysis](autorouter-technical-analysis.md)
- [LiteLLM Load Balancing: Technical Analysis](load-balancing-technical-analysis.md)

## Coverage

The autorouter analysis covers the architecture overview, complexity tiers, heuristic and LLM-based classification, keyword and semantic overrides, escalation keywords, adaptive routing, Thompson sampling, and posterior updates.

The load-balancing analysis covers model groups, deployments, routing groups, candidate filtering, deployment-level routing strategies, Redis-backed runtime state, cooldowns, retries, same-group failover, cross-group fallbacks, strategy precedence, budget filters, and the boundary between ordinary load balancing and model-level pre-routing.
