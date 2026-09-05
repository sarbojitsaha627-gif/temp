---
name: Finance Controller Agent
overview: Razorpay Track 4 Finance Controller with split routing, dual-mode HITL, M:N allocation uniqueness, leak-attributing safety checks, and resolved 5-point structural architecture lock-in. Spec names stay; all architecture decisions locked in prior to Step 1.
todos:
  - id: wait-step-1
    content: Wait for numbered Step 1; implement only that slice against this lock-in
    status: pending
  - id: split-bypass-routing
    content: "Graph routing: exact → assign; fuzzy_resolved → verify; only unresolved/ambiguous clusters → llm"
    status: pending
  - id: retry-error-delta-loop
    content: "Verification retry loop: check_retry_limit loops up to 2 times to llm with signed last_error_delta (paise) before escalating to hitl"
    status: pending
  - id: graph-invocation-scope
    content: "Per-cluster graph invocation (Option B): plain Python batch orchestrator + per-cluster LangGraph subgraphs with isolated thread checkpoints"
    status: pending
  - id: explicit-categorize-exception
    content: "CategorizeException node: strict 8+3 taxonomy enforcement with 1:1 synthetic data mapping table and T+2 window validation"
    status: pending
  - id: cluster-candidates-node
    content: "ClusterCandidates node: disjoint partitioning + max 8 cap ensures entity exclusivity and exactly 1 LLM call per cluster"
    status: pending
  - id: concurrency-collision-guard
    content: "Concurrency guard: Disjoint cluster partitioning + try/except IntegrityError handler on UNIQUE(entity_type, entity_id)"
    status: pending
  - id: two-pass-threeway-match
    content: "Two-pass staged 3-way match: Hop 1 (Orders ↔ Settlements) & Hop 2 (Settlements ↔ BankTxns) with unified match_allocations and two-hop conservation"
    status: pending
  - id: dual-mode-hitl
    content: EVAL_MODE bypasses blocking interrupt(); auto-tag ESCALATED_UNRESOLVED and continue to report
    status: pending
  - id: allocation-junction
    content: match_groups + match_allocations unique on (entity_type, entity_id); allocated_paise for M:N
    status: pending
  - id: leak-attribution
    content: Safety failure creates UNACCOUNTED_LEDGER_LEAK with residual delta, then always emits report
    status: pending
  - id: no-llm-arithmetic
    content: Keep LLM on IDs/categories only; amounts and balance checks stay in deterministic code
    status: pending
  - id: gst-rounding-tolerance
    content: "IndependentVerifier allows ±1 paisa per order item for GST rounding accumulation; attributed as ROUNDING_ACCUMULATION, not a leak"
    status: pending
  - id: utr-normalization
    content: "IngestBatch runs canonical UTR regex extractor before ExactIdJoin; strips CBS prefixes/suffixes"
    status: pending
  - id: duplicate-batch-guard
    content: "Idempotent batch ingestion via SHA-256 source file checksum + reconciliation_run_id; reject re-uploads"
    status: pending
  - id: llm-api-failure-handling
    content: "LlmReActLoop retries API 429/500 with exponential backoff (max 3); on exhaustion escalate to ESCALATED_UNRESOLVED"
    status: pending
---

# AI Finance Controller — Plan Lock-In & Architecture Decisions

Empty workspace; we wait for **Step 1** and treat each numbered step as cumulative. All table, state, node, and tool **names stay as given**. 

This document locks in the **five core architecture decisions** alongside the required system safeguards (split routing, dual-mode HITL, $M:N$ allocation junction, and leak attribution) to prevent token burn, pipeline blocking, false constraint rejections, and silent unbalance.

---

## Complete End-to-End Architecture Flowchart

```mermaid
flowchart TD
  subgraph Orchestrator_Pre [Batch Orchestrator: Deterministic Ingestion & Fast Path]
    ingest[IngestBatch\nParse, Validate, Normalize UTRs & Timestamps, Dedup Check]
    hop1_exact[Hop 1: ExactIdJoin\nOrders ↔ Settlements on payment_id/order_id]
    hop2_exact[Hop 2: ExactIdJoin\nSettlements ↔ BankTxns on UTR]
    fuzzy[FuzzyScore\nToken/Levenshtein/Amount Similarity]
    cluster[ClusterCandidates\nDisjoint Partitioning & Max 8 Cap]
  end

  subgraph LangGraph_Cluster [Per-Cluster LangGraph StateGraph: thread_id = batch:cluster_id]
    llm[LlmReActLoop\n1 Prompt per Cluster, max 5 iterations]
    fallback_check{CheckModelFallback\nModels remaining in chain?}
    verify[IndependentVerifier\nDeterministic Integer Paise Arithmetic]
    retry_check{CheckRetryLimit\nretry_count < 2?}
    hitl[HumanReviewInterrupt\nDual-Mode: Block vs ESCALATED_UNRESOLVED]
    categorize[CategorizeException\nStrict 8+3 Canonical Taxonomy]
  end

  subgraph Orchestrator_Post [Batch Orchestrator: Persistence & Global Safety]
    assign[PersistMatchOrException\nAtomic Commit to match_groups & match_allocations]
    safety[BatchSafetyChecks\nTwo-Hop & Global Conservation Equation]
    leak[UnaccountedLedgerLeak\nAttribute Delta to Residual Exception]
    report[ReportMatchRateThroughput\nMetrics & Audit Summary]
  end

  ingest --> hop1_exact
  hop1_exact -->|confident_exact (balanced)| hop2_exact
  hop1_exact -->|unmatched / partial| fuzzy
  hop1_exact -->|id_match_amount_mismatch| cluster

  hop2_exact -->|confident_exact (balanced)| assign
  hop2_exact -->|unmatched| fuzzy
  hop2_exact -->|id_match_amount_mismatch| cluster

  fuzzy -->|fuzzy_resolved (score_gap >= 0.15 AND no collision)| verify
  fuzzy -->|ambiguous_candidates| cluster
  fuzzy -->|no_candidates_found| categorize

  cluster --> llm
  llm -->|Success| verify
  llm -->|Hard Failure\n(429, timeout, parse fail)| fallback_check

  fallback_check -->|yes: Wipe messages &\nnext model| llm
  fallback_check -->|no: Chain exhausted| categorize

  verify -->|pass| assign
  verify -->|fail (discrepancy delta)| retry_check

  retry_check -->|yes: retry_count < 2\n(pass last_error_delta in paise)| llm
  retry_check -->|no: retry_count >= 2| hitl

  hitl -->|resolved_by_human| assign
  hitl -->|unresolved / eval_mode| categorize

  categorize --> assign

  assign --> safety
  safety -->|balanced (zero residual)| report
  safety -->|conservation_fail (leak detected)| leak
  leak --> report
```

---

## Five Structural Architecture Decisions

### 1. Restore the Retry-with-Error-Delta Loop

```mermaid
flowchart LR
  verify[IndependentVerifier] -->|pass| assign[PersistMatchOrException]
  verify -->|fail| check{CheckRetryLimit\nretry_count < 2?}
  check -->|yes: retry_count < 2\n(feed last_error_delta)| llm[LlmReActLoop]
  check -->|no: retry_count >= 2| hitl[HumanReviewInterrupt]
```

- **Pipeline Placement:** A dedicated conditional evaluation node `CheckRetryLimit` is inserted between `IndependentVerifier` and `HumanReviewInterrupt`.
- **Mechanics & State Variables:**
  - `retry_count: int` (initialized to `0`, capped at `2`).
  - `last_error_delta: Optional[int]` (signed integer paise discrepancy: $\text{expected} - \text{calculated}$).
  - `verification_feedback: Optional[str]` (exact diagnostic string generated by `IndependentVerifier`, e.g. `"Gross 100000 paise != Net 97000 + Fee 2000 + Tax 360. Discrepancy delta = +640 paise. Identify missing fee or rate adjustment."`).
  - When `IndependentVerifier` fails:
    1. If `retry_count < 2`: Increments `retry_count += 1`, injects `last_error_delta` and `verification_feedback` into the ReAct prompt context, and loops back to `LlmReActLoop`.
    2. If `retry_count >= 2`: Retries are exhausted; routes to `HumanReviewInterrupt` (or auto-escalates in `EVAL_MODE`).
- **Decision Note:**
  > **Decision:** Implemented an explicit 2-attempt self-correction loop (`CheckRetryLimit`).
  > **Rationale:** LLMs frequently omit small secondary deductions (e.g. 18% GST on platform fees or partner discounts) on the first pass. Supplying the exact signed `last_error_delta` in paise enables the LLM to self-correct with high accuracy in 1 additional turn without incurring human intervention costs, while a hard cap of 2 prevents infinite token burn.

---

### 2. Define Graph Invocation Scope & Orchestration Boundary

- **Selected Model:** **Option (b) — Plain Python Batch Orchestrator + Per-Cluster LangGraph StateGraph**.
- **Execution Architecture & Boundaries:**
  1. **Batch Orchestrator Tier (Plain Python & Vectorized DB Queries):** 
     - `IngestBatch`, `ExactIdJoin` (Hop 1 & Hop 2), `FuzzyScore`, and `ClusterCandidates` execute as high-throughput, deterministic batch functions in plain Python/SQL before any LangGraph invocation begins.
     - Confident exact and unambiguous fuzzy matches write directly to `match_groups` and `match_allocations` in sub-milliseconds without LangGraph state overhead.
  2. **Cluster Graph Invocation Tier (LangGraph `StateGraph`):** 
     - For each ambiguous candidate cluster output by `ClusterCandidates`, an independent LangGraph invocation is spawned with an isolated checkpoint thread:
       $$\text{thread\_id} = f"{batch\_id}:cluster\_\{cluster\_id\}"$$
     - The LangGraph graph exclusively wraps the cluster-level reasoning loop:
       $$\text{LlmReActLoop} \rightarrow \text{IndependentVerifier} \rightarrow \text{CheckRetryLimit} \rightarrow \text{HumanReviewInterrupt} \rightarrow \text{CategorizeException}$$
  3. **Non-Blocking HITL Guarantee:** 
     - If a cluster triggers `HumanReviewInterrupt` under `EVAL_MODE=false`, LangGraph's `interrupt()` suspends *only that specific cluster's thread checkpoint* in Postgres. All other independent cluster graph threads and deterministic fast-paths continue executing concurrently to completion.
  4. **Concurrency & `match_allocations` Unique Constraint Collision Defense:**
     - *Primary Guard (Exclusivity by Construction):* `ClusterCandidates` enforces disjoint-set graph partitioning (connected components / Union-Find) so an entity belongs to at most one `CandidateCluster`.
     - *Secondary Guard (Write-Time Concurrency Handler):* If an unexpected race condition occurs on `UNIQUE(entity_type, entity_id)` during parallel execution:
       - `PersistMatchOrException` catches Postgres `UniqueViolation` / SQLAlchemy `IntegrityError` in an atomic transaction.
       - Cleanly rolls back the transaction without crashing the worker thread.
       - Queries `match_allocations` to inspect the winning `match_group_id`.
       - Re-evaluates remaining unallocated candidates for the losing cluster; if no candidates remain, the losing cluster's orphaned primary entity is safely routed to `CategorizeException(ESCALATED_UNRESOLVED)` with `reasoning_trace` recording `"Entity <id> claimed by concurrent cluster <winner_id>; re-routed to exception staging"`.
  5. **Batch Aggregation & Safety Pass:** 
     - The orchestrator collects completed allocations, logs any pending HITL thread IDs (or `ESCALATED_UNRESOLVED` records in `EVAL_MODE=true`), and executes `BatchSafetyChecks` and `ReportMatchRateThroughput`.
- **Decision Note:**
  > **Decision:** Selected Option (b) (Plain Python Batch Orchestrator + Per-Cluster LangGraph StateGraph).
  > **Rationale:** LangGraph's native `interrupt()` halts graph execution for the target thread. Running deterministic fast-path joins and batch clustering inside a single monolithic StateGraph introduces serialization overhead, state bloat, and halts entire batches on single-item interrupts. Option (b) ensures deterministic fast paths run in milliseconds, isolates LLM checkpointing strictly to ambiguous clusters, guarantees non-blocking HITL, and produces clean per-record trace trees in LangSmith.

---

### 3. Assign Exception Categorization to an Explicit Node & Taxonomy Mapping

- **Pipeline Placement:** `CategorizeException` is an explicit, dedicated node positioned immediately before `PersistMatchOrException` for all unmatched, verification-exhausted, or unresolvable records.
- **Node Responsibilities & Taxonomy Rules Engine (Canonical 8 + 3 Taxonomy):**
  - Evaluates deterministic rule predicates against the canonical 8 core domain categories + 3 operational categories:
    1. **`TIMING_SETTLEMENT_FLOAT` (with $T+2$ Window Rule):** 
       - Calculates business-day elapsed duration between order timestamp and settlement value date:
       - If $\Delta t \le 2$ business days (48 hours) with status pending $\rightarrow$ sub-type `PENDING_FLOAT` (in-flight operational float, not an anomaly).
       - If $\Delta t > 2$ business days without settlement $\rightarrow$ sub-type `ANOMALOUS_FLOAT_OVERDUE`.
    2. **`GATEWAY_FEE_MISMATCH`:** Fee schedule variance from contract (>2% MDR + 18% GST).
    3. **`UNRECONCILED_BANK_FEE`:** Bank charges/debits lacking gateway settlement advice.
    4. **`SPLIT_PAYOUT_PARTIAL_DROP`:** Settlement payout covers only part of multi-order bundle or dropped refund.
    5. **`CHARGEBACK_DEBIT_UNMATCHED`:** Dispute debit without corresponding order record.
    6. **`CURRENCY_CONVERSION_VARIANCE`:** FX exchange rate variance on cross-border transactions.
    7. **`SUSPICIOUS_ROUND_NUMBER_DRAIN`:** Repeated round-sum drains or duplicate payout anomalies.
    8. **`MISSING_SETTLEMENT_RECORD`:** Order captured in merchant ledger with zero settlement advice or bank credit (genuinely absent, not late).
    9. **`UNMAPPED_BANK_DEPOSIT`:** Bank statement credit received with zero corresponding gateway settlement advice or order.
    10. **`ESCALATED_UNRESOLVED` (Additive):** Human review bypassed in `EVAL_MODE=true` or unresolvable unknown.
    11. **`UNACCOUNTED_LEDGER_LEAK` (Additive):** Global balance residual created during `BatchSafetyChecks`.
- **Strict Validation & No Fallbacks:**
  - Every row staged in `exception_staging` must strictly match an allowed Enum value in the taxonomy.
  - `null`, empty, or arbitrary freeform strings are rejected by Pydantic validation and mapped to `ESCALATED_UNRESOLVED` with explicit diagnostic metadata.

- **Taxonomy-to-Synthetic-Data Mapping Table:**
  Every synthetic scenario generated by the Step 2 data generator maps 1:1 to exactly one canonical resolution path or exception category:

| Synthetic Data Generator Case | Resolution Pipeline Path | Final Target Category / Ledger Destination | Exact Invariant & Accounting Logic |
| :--- | :--- | :--- | :--- |
| **`clean_match`** | `ExactIdJoin` (Hop 1 & 2) | `match_groups` (`tier='exact'`, `verified=True`) | Exact ID match on `order_id` & `utr`; $\text{gross} = \text{net} + \text{fee} + \text{tax}$ ($\Delta = 0$). No exception staged. |
| **`fee_deduction_only`** | `ExactIdJoin` $\rightarrow$ `LlmReActLoop` | `match_groups` (`tier='exact'`/`'llm'`) OR `GATEWAY_FEE_MISMATCH` | If fee aligns with contracted schedule (2% + 18% GST), verified match. If fee breaches schedule, staged as `GATEWAY_FEE_MISMATCH`. |
| **`timing_lag`** | `CategorizeException` | `TIMING_SETTLEMENT_FLOAT` (`PENDING_FLOAT` vs `ANOMALOUS_FLOAT_OVERDUE`) | Evaluated via Indian banking calendar: $\le 2$ business days $\rightarrow$ `PENDING_FLOAT` (normal float); $> 2$ business days $\rightarrow$ `ANOMALOUS_FLOAT_OVERDUE`. |
| **`partial_refund`** | `ExactIdJoin` / `LlmReActLoop` | `match_groups` (`tier='exact'`/`'llm'`) OR `SPLIT_PAYOUT_PARTIAL_DROP` | Signed `allocated_paise` accounts for refund deduction in allocation ledger. If refund was omitted/dropped by gateway, staged as `SPLIT_PAYOUT_PARTIAL_DROP`. |
| **`reference_mismatch`** | `FuzzyScore` $\rightarrow$ `ClusterCandidates` $\rightarrow$ `LlmReActLoop` | `match_groups` (`tier='fuzzy'`/`'llm'`) OR `ESCALATED_UNRESOLVED` | Mismatched reference/typo resolved by RapidFuzz (score gap $\ge 0.15$) or ReAct narration tools. If unresolvable, staged as `ESCALATED_UNRESOLVED` in `EVAL_MODE`. |
| **`duplicate`** | `ExactIdJoin` / `CategorizeException` | `SUSPICIOUS_ROUND_NUMBER_DRAIN` | Duplicate payout attempts or repeated round-sum drains flagged as suspicious drain anomaly. |
| **`missing_settlement`** | `CategorizeException` | `MISSING_SETTLEMENT_RECORD` | Order captured in merchant ledger but zero settlement advice or bank credit ever received. Staged with gross order paise in `residual_paise`. |
| **`unmapped_bank_deposit`** | `CategorizeException` | `UNMAPPED_BANK_DEPOSIT` | Unidentified credit landed on bank statement with zero corresponding gateway settlement advice or order. Staged with credit paise in `residual_paise`. |

- **Decision Note:**
  > **Decision:** Created an explicit `CategorizeException` node with strict 8+3 taxonomy rules, complete 1:1 synthetic data mapping, and $T+2$ float checking.
  > **Rationale:** Exception classification is a core financial compliance function. Explicit rule-based categorization guarantees deterministic audit trails, correctly distinguishes normal $T+2$ pending float from overdue anomalies, ensures complete coverage of all synthetic generation edge cases, and prevents unclassified exception leaks.

---

### 4. Restore `ClusterCandidates` & Disjoint Partitioning Guarantee

- **Pipeline Placement:** `ClusterCandidates` is restored as an explicit, first-class node positioned between `FuzzyScore` and `LlmReActLoop`.
- **Mechanism & Disjoint Partitioning:**
  - `FuzzyScore` calculates pairwise similarity scores (Token Sort Ratio, Levenshtein, amount delta, timestamp proximity).
  - `ClusterCandidates` filters candidate pairs falling in the ambiguous band ($0.50 \le \text{score} < 0.85$ or score gap $< 0.15$ or same-amount/same-day collisions) and groups them using:
    - A temporal window ($\pm N$ days, default $\pm 2$ business days).
    - Amount proximity & common entity references (e.g. shared customer/merchant/UTR prefix).
  - **Exclusivity by Construction (Disjoint Partitioning):** Employs connected-component / disjoint-set graph partitioning (Union-Find). Every entity involved in ambiguous pairings belongs to exactly one `CandidateCluster`. No entity appears in more than one concurrent cluster.
  - Groups candidates into discrete `CandidateCluster` objects:
    ```python
    class CandidateCluster(BaseModel):
        cluster_id: str
        primary_entity_type: Literal["order", "settlement", "bank_transaction"]
        primary_entity_id: str
        candidate_matches: List[CandidateMatch]
        window_start: datetime
        window_end: datetime
        aggregate_delta_paise: int
        has_amount_collision: bool
    ```
- **Token & Latency Guarantee:**
  - Ensures **exactly 1 LLM call per candidate cluster** (e.g. 1 settlement with 3 competing order candidates presented in a single prompt) rather than 1 LLM call per candidate pair ($O(K)$ cluster calls instead of $O(N \times M)$ pairwise calls).
  - **Hard Partition Cap:** Caps candidate clusters at a **maximum of 8 candidates**. If high-volume collisions (e.g. flash sales) produce $>8$ candidates, `ClusterCandidates` automatically sub-partitions them into separate micro-clusters using timestamp buckets (e.g. 6-hour intervals) or customer/merchant reference prefixes, preventing context blowup and LLM attention degradation.
- **Decision Note:**
  > **Decision:** Restored `ClusterCandidates` as an explicit node between `FuzzyScore` and `LlmReActLoop` with a hard cap of 8 candidates per cluster and disjoint entity partitioning.
  > **Rationale:** Without an explicit clustering node, fuzzy matching either floods the LLM with combinatorial pairwise calls or leaves grouping to an opaque sub-routine. Explicit windowed clustering with disjoint partitioning guarantees zero entity collisions across concurrent workers, protects token budget, slashes latency ($<5$s), and provides the LLM with complete comparative context across competing candidates without attention degradation.

---

### 5. Define the Three-Way Match Structure

- **Selected Structure:** **Two-Pass Staged Reconciliation with Unified Match Allocation Ledger**.
- **The Two Reconciliation Hops:**
  1. **Hop 1 (Merchant Ledger to Gateway Advice):** `orders` + `refunds` + `disputes` $\leftrightarrow$ `settlements`
     - **Conservation Equation:** 
       $$\sum \text{orders.gross\_paise} - \sum \text{refunds.amount\_paise} - \sum \text{disputes.amount\_paise} = \text{settlement.net\_paise} + \text{settlement.fee\_paise} + \text{settlement.tax\_paise}$$
     - **Candidate Keys:** `payment_id`, `order_id`, `refund_id`, settlement line items, amount sum, date window.
  2. **Hop 2 (Gateway Payout to Bank Statement):** `settlements` $\leftrightarrow$ `bank_transactions`
     - **Conservation Equation:**
       $$\sum \text{settlements.net\_paise} = \text{bank\_transaction.credit\_paise} - \text{bank\_charges\_paise}$$
     - **Candidate Keys:** `utr`, settlement reference ID, bank narration, value date ($\pm 1$ business day).
- **Unified Allocation Schema (`match_groups` + `match_allocations`):**
  - `match_groups` holds the reconciliation metadata: `id`, `tier` (`exact | fuzzy | llm | human`), `confidence_score`, `verified`, `residual_paise`, `reasoning_trace`, `cited_evidence` (JSONB), `hitl_status`, `processing_ms`, `created_at`, and `original_tier_hint`.
  - `match_allocations` records individual entity participations:
    - `match_group_id: UUID (FK)`
    - `entity_type: Literal["order", "refund", "dispute_debit", "settlement", "bank_transaction"]`
    - `entity_id: str`
    - `allocated_paise: BigInteger` (**signed**, no `CHECK (> 0)` constraint, enabling negative cash flow / refund / clawback accounting)
    - **Constraint:** `UNIQUE (entity_type, entity_id)` guarantees zero double-spending across both hops.
- **Two-Hop & Global Conservation in `BatchSafetyChecks`:**
  `BatchSafetyChecks` verifies conservation across both hops:
  $$\sum_{\text{all orders}} \text{gross} - \sum \text{refunds} - \sum \text{disputes} = \sum_{\text{all bank txns}} \text{credit} + \sum \text{gateway\_fees} + \sum \text{taxes} + \sum \text{bank\_charges} + \sum \text{residuals}$$
  If net discrepancy $\neq 0$, the exact signed residual `delta_paise` is attributed to `UNACCOUNTED_LEDGER_LEAK`.
- **Decision Note:**
  > **Decision:** Selected Two-Pass Staged Reconciliation (Hop 1: Orders/Refunds ↔ Settlements, Hop 2: Settlements ↔ BankTxns) with unified `match_allocations` and signed integer amounts.
  > **Rationale:** In Indian banking and payment rails, bank statements never contain individual merchant `order_id`s; the gateway settlement UTR is the sole bridge. Attempting a single-pass 3-way join forces unnatural heuristic leaps across missing keys. Staged two-pass reconciliation reflects the actual financial flow, natively supports negative refund clawbacks, enables independent exact/fuzzy joins per hop, and seamlessly aggregates into global ledger conservation.

---

## Core System Invariants & Safeguards (Preserved)

### Split and Bypass Routing
- **Exact confident matches** $\rightarrow$ `assign` (bypass fuzzy, cluster, llm, verify).
- **Fuzzy resolved** (score gap $\ge 0.15$ AND no same-amount same-day collision) $\rightarrow$ `verify` $\rightarrow$ `assign`.
- **Unresolved / Ambiguous clusters** $\rightarrow$ `cluster_candidates` $\rightarrow$ `llm` $\rightarrow$ `verify`.

### Exact ID Match Amount Failure & Demotion
If an exact ID join matches on identifier (e.g. `order_id` or `utr`) but fails mathematical amount conservation ($\sum \text{gross} \neq \sum \text{net} + \text{fees} + \text{tax}$):
1. **Immediate Demotion:** Disqualified from fast-path `assign` and demoted to `ClusterCandidates` $\rightarrow$ `LlmReActLoop`.
2. **Provenance Preservation:** Tagged with `original_tier_hint = "exact_id_conservation_failed"` and initial diagnostic in `reasoning_trace`.
3. **Canonical Taxonomy Mapping:** If unresolvable, mapped via `CategorizeException` to `GATEWAY_FEE_MISMATCH` or `SPLIT_PAYOUT_PARTIAL_DROP` (never ad-hoc strings).
4. **Zero Special-Casing in `IndependentVerifier`:** All demoted proposals must strictly pass integer paise balance verification before committing.

### Conservative Definition of `fuzzy_resolved`
A fuzzy match only qualifies as `fuzzy_resolved` (bypassing the LLM) if **BOTH**:
1. Top-scoring candidate exceeds second-best score by at least **0.15** (score gap $\ge 0.15$), **AND**
2. **No other candidate shares the same `amount_paise`** within the same settlement/value_date window.
If either condition fails, route to `ClusterCandidates` $\rightarrow$ `LlmReActLoop`.

### LlmReActLoop Execution & Iteration Bounds
- **Bounded Iterations:** `LlmReActLoop` enforces a strict **`max_iterations = 5`** limit per cluster.
- **Auto-Escalation on Exhaustion:** If the agent reaches 5 iterations without a structured hypothesis or tool conclusion, it immediately aborts the loop and transitions to `CategorizeException` with `ESCALATED_UNRESOLVED` (recording iteration exhaustion in `reasoning_trace`), preventing infinite tool cycling and worker thread timeouts.
- **LLM API Failure Handling:** On transient API errors (HTTP 429 rate-limit, 500 server error, timeout, or malformed JSON response), the loop retries with **exponential backoff** (delays: 1s → 2s → 4s, **max 3 API retries** per tool call, distinct from the semantic `retry_count` / `CheckRetryLimit` loop). If all 3 API retries fail, the cluster is immediately routed to `CategorizeException` with `ESCALATED_UNRESOLVED` and `reasoning_trace` records `"LLM API unavailable after 3 retries: <HTTP status / error>"`. This prevents silent hangs when the LLM provider is degraded.

### Stricter Verification for Fallback Amount Collisions
- **Risk Context:** In ambiguous clusters where $\ge 2$ candidates share the identical `amount_paise` within the same date window, a weaker fallback model is at higher risk of hallucinating the wrong candidate despite passing the mathematical conservation check.
- **The Gate:** Inside `IndependentVerifier`, if `has_amount_collision=True` AND `model_used` is NOT the primary model (the first model in the chain), the agent requires `confidence_score >= 0.9`. 
- **Override Action:** If the score is $<0.9$, the verification is overridden to `fail` and routed to `HumanReviewInterrupt`, with `reasoning_trace` appending: `"Verification passed but routed to human review: amount-collision cluster resolved by fallback model below confidence threshold."`
- **Specificity:** This gate does NOT apply to non-collision clusters (`has_amount_collision=False`) or clusters resolved by the primary model.

### Dual-Mode HITL
- Config flag: `EVAL_MODE: bool` (env + run config; default `false` for Streamlit UI).
- **`EVAL_MODE=false`:** LangGraph `interrupt()` + Postgres checkpointer; human resumes; does not re-match on resume.
- **`EVAL_MODE=true`:** Bypasses blocking interrupt. Tags record `ESCALATED_UNRESOLVED` (honest exception) and continues so batch coverage check and report always complete.

### $M:N$ Allocation Junction (Replacing `reconciliation_results`)
`match_groups` + `match_allocations` **REPLACE** `reconciliation_results` from the original spec entirely.
- `match_groups`: Parent reconciliation record with tier, confidence, verification status, and residual.
- `match_allocations`: `(match_group_id, entity_type, entity_id, allocated_paise)` with `UNIQUE(entity_type, entity_id)` and signed `BigInteger` `allocated_paise`.
- `exception_staging`: Remains separate and dedicated for non-allocated exceptions.

### Leak Attribution, Then Always Report
If global batch conservation fails ($\text{gross} \neq \text{net} + \text{fees} + \text{tax} + \text{bank\_charges} + \text{residuals}$):
1. Catch signed `delta_paise`.
2. Insert exception `UNACCOUNTED_LEDGER_LEAK` with that delta and `reconciliation_run_id`.
3. Emit `ReportMatchRateThroughput` anyway with true match rate, throughput, and categorized exceptions.

### Correctness Rules I Will Not Compromise
- **Paise only** (`Integer` / `BigInteger`, signed where negative flows exist). Zero floats in Python, SQL, or LLM-facing amount fields.
- **No double-spend:** `UNIQUE (entity_type, entity_id)` on `match_allocations`.
- **LLM output is proposals only:** IDs + exception category + reason. Deterministic code verifies all arithmetic.
- **Bounded LLM Execution:** Max 8 candidates per cluster; max 5 ReAct iterations per invocation; max 3 API retries with exponential backoff per tool call.
- **Honest metrics:** Match rate and throughput calculated from DB state and full batch wall clock.
- **Nothing lost:** Every source row is accounted for in `match_allocations`, `exception_staging`, or `ESCALATED_UNRESOLVED`.
- **HITL:** Interactive review in UI; never blocks eval (`EVAL_MODE`).
- **LLM cost control:** Only windowed ambiguous clusters enter `LlmReActLoop`.

---

## Operational Robustness Amendments

### 1. GST Rounding Tolerance in `IndependentVerifier`
Per-item GST rounding ($\text{round}(0.18 \times \text{fee}_i)$) accumulates drift when many orders are bundled into a single settlement. The sum of individually rounded taxes often differs from the tax on the aggregate fee by $\pm 1$ to $\pm 5$ paise across a 50+ order batch.

- **Rule:** `IndependentVerifier` allows a rounding tolerance of **$\pm 1$ paisa per order item** in the conservation check:
  ```python
  max_rounding_tolerance_paise = len(order_allocations_in_group) * 1
  if abs(delta_paise) <= max_rounding_tolerance_paise:
      # Attribute to ROUNDING_ACCUMULATION, mark verified=True
  ```
- **Classification:** If the residual falls within this tolerance band and the fee/tax structure is otherwise correct, the match group is verified as `pass` with `residual_paise` recording the exact drift and `reasoning_trace` noting `"GST rounding accumulation: <N> paise across <M> orders"`.
- **Not a Leak:** This residual is **not** routed to `UNACCOUNTED_LEDGER_LEAK`. It is an expected artifact of per-item rounding arithmetic. `BatchSafetyChecks` accounts for it in the global equation as an attributed rounding residual.

### 2. Canonical UTR Normalization in `IngestBatch`
Core banking systems (HDFC, ICICI, SBI, Axis, etc.) mangle UTR strings in statement exports by adding bank-specific prefixes/suffixes, truncating to 16–20 chars, or stripping leading zeros.

- **Rule:** `IngestBatch` runs a **Canonical UTR Normalizer** on all bank transaction narrations and settlement UTR fields before `ExactIdJoin`:
  ```python
  import re
  _UTR_PATTERN = re.compile(r"([A-Z]{4}[A-Z0-9]{12,18}|[0-9]{12,22})", re.IGNORECASE)

  def extract_canonical_utr(raw_text: str) -> str | None:
      """Extract the core UTR identifier from CBS-mangled narration strings."""
      if not raw_text:
          return None
      match = _UTR_PATTERN.search(raw_text.strip())
      return match.group(1).upper() if match else raw_text.strip().upper()
  ```
- **Stored As:** The extracted canonical UTR is stored in a normalized column (e.g. `canonical_utr`) alongside the original `raw_narration`. `ExactIdJoin` (Hop 2) matches on `canonical_utr`, not the raw string.
- **Fallback:** If the regex extracts nothing (e.g. purely descriptive narration with no UTR), the record proceeds to `FuzzyScore` with the raw narration as input.
- **Timestamp Normalization:** All timestamps are normalized to **UTC** at ingestion. The $T+2$ float check in `CategorizeException` uses an **Indian business-day calendar** (skipping Sundays, 2nd/4th Saturdays, and RBI-declared holidays).

### 3. Duplicate Batch Guard (Idempotent Re-Run Protection)
If the same settlement CSV is accidentally uploaded twice, or a partial infrastructure failure triggers an automatic retry, the system must not double-allocate entities.

- **Rule:** `IngestBatch` computes a **SHA-256 checksum** of the source file content and stores it alongside the `reconciliation_run_id`:
  ```python
  source_checksum = hashlib.sha256(file_bytes).hexdigest()
  ```
- **Deduplication Check:** Before proceeding, query `reconciliation_runs` for any prior run with the same `source_checksum`:
  - If a **completed** prior run exists with the same checksum → **reject** the upload with a clear error: `"Duplicate batch detected (matches run <run_id> completed at <timestamp>). Skipping."`
  - If a **failed/partial** prior run exists with the same checksum → allow re-run but **skip entities already persisted** in `match_allocations` from the prior attempt (resume semantics).
  - If no prior run exists → proceed normally.
- **Partial Failure Recovery:** The batch orchestrator tracks which cluster graph invocations completed vs. failed. On resume, it re-spawns only incomplete clusters, preserving all previously committed `match_groups` and `match_allocations`.

### 4. LLM API Failure Handling & Fallback-Model Chain
LLM provider outages (HTTP 429 rate-limit, 500 server error, network timeout, malformed JSON) must not crash the batch or silently drop records.

- **Configured Model Chain:** The pipeline defines a fallback chain (e.g., `groq/llama-3.3-70b-versatile, groq/openai-gpt-oss-120b, gpt-4o`). The primary model is attempted first.
- **Pre-execution Gate:** At the start of `LlmReActLoop`, the node checks if the current model is present in the `exhausted_models` array (a shared JSONB list on the orchestrator's `ReconciliationRun` database record). If flagged as exhausted, it skips straight to `CheckModelFallback` without wasting any API calls.
- **429 Rate Limit Parsing (Rolling vs Daily):**
  - **Rolling Window Limit (RPM/TPM):** If the 429 response indicates a transient window limit, the graph parses the `retry-after` header, sleeps for that exact duration, and retries the same model exactly once before falling back.
  - **Daily/Hard Quota Exhaustion (RPD/TPD):** If the response indicates daily/hard quota exhaustion, the model is immediately marked as exhausted globally by appending its identifier to the `exhausted_models` JSONB array on `ReconciliationRun`. The graph skips straight to `CheckModelFallback` without making the standard 3 API retries.
- **Hard Failure Handling:** For other transient errors (500, network timeouts), each tool call inside `LlmReActLoop` uses standard **exponential backoff retry** (delays: 1s → 2s → 4s, **max 3 API retries** per tool call).
- **Cluster-Boundary Fallback (`CheckModelFallback`):** If all 3 API retries fail, OR the rolling limit retry fails, OR if the model repeatedly produces un-parsable output, that model attempt is completely abandoned. The graph routes to `CheckModelFallback`.
- **Clean State Reset:** If a fallback model is available in the chain, `CheckModelFallback` increments the model index, completely wipes the `messages` list (erasing the failed model's partial conversational history), and loops back to `LlmReActLoop` to start fresh with a clean prompt built from the `CandidateCluster`. Mid-loop model swapping within a single reasoning thread is forbidden.
- **Chain Exhaustion & Escalation:** If all models in the chain fail, the cluster is immediately routed to `CategorizeException` with `ESCALATED_UNRESOLVED`, and `reasoning_trace` records `"LLM Model chain exhausted after <N> models. Last error: <HTTP status / error>"`.
- **Honest Tracking & Reporting:** A `model_used` column in `match_groups` captures exactly which model successfully resolved the cluster. The final `ReportMatchRateThroughput` step prominently surfaces the `exhausted_models` array alongside how many clusters were resolved by the fallback models, providing full transparency on batch LLM consumption.

---

## What We Will Not Do Yet
No repository scaffolding or source code changes until you provide **Step 1**. Then implement that slice against this locked-in architecture.

## After Step 1 Lands
Implement that slice, keep spec names, use `match_groups` / `match_allocations` (replacing `reconciliation_results`), `EVAL_MODE`, per-cluster graph invocations, `CheckRetryLimit` self-correction loop, explicit `CategorizeException`, explicit `ClusterCandidates` (max 8 candidates), bounded `LlmReActLoop` (max 5 iterations, max 3 API retries with backoff), signed `allocated_paise` for refund/clawback support, two-pass staged 3-way matching, GST rounding tolerance, canonical UTR normalization, and duplicate batch guard.
