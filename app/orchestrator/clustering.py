"""
FuzzyScore + ClusterCandidates — fuzzy matching and disjoint-set clustering.

FuzzyScore:
  Uses RapidFuzz (Token Sort Ratio + partial ratio) and amount/timestamp
  proximity to score unmatched entity pairs.

ClusterCandidates (plan.md Decision #4):
  • Filters ambiguous pairs (score 0.50–0.85, score gap < 0.15, or
    same-amount/same-day collisions).
  • Groups via connected-component / Union-Find for disjoint partitioning
    — every entity belongs to at most one CandidateCluster.
  • Hard cap: max 8 candidates per cluster.  Overflow → sub-partitioned
    into micro-clusters using timestamp buckets.

Design invariants:
  • Exactly 1 LLM call per cluster (O(K) not O(N×M)).
  • Fuzzy-resolved (score gap ≥ 0.15 AND no same-amount collision) → verify.
  • All amounts are integer paise.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rapidfuzz import fuzz

from app.orchestrator.matching import DemotedRecord, UnmatchedRecord

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

# Fuzzy score thresholds
FUZZY_RESOLVED_MIN_SCORE = 0.85
FUZZY_AMBIGUOUS_MIN_SCORE = 0.50
SCORE_GAP_THRESHOLD = 0.15

# Cluster size cap
MAX_CLUSTER_SIZE = 8

# Date window for clustering (±2 business days ≈ ±4 calendar days buffer)
DEFAULT_WINDOW_DAYS = 4


# ═══════════════════════════════════════════════════════════
#  RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════


@dataclass
class FuzzyPairScore:
    """A scored candidate pair from fuzzy matching."""
    source_entity_type: str
    source_entity_id: str
    source_amount_paise: int
    target_entity_type: str
    target_entity_id: str
    target_amount_paise: int
    score: float
    score_components: dict[str, float]
    source_timestamp: datetime | None = None
    target_timestamp: datetime | None = None


@dataclass
class CandidateMatch:
    """A single candidate within a cluster."""
    entity_type: str
    entity_id: str
    amount_paise: int
    score: float
    timestamp: datetime | None = None
    raw_narration: str | None = None
    canonical_utr: str | None = None


@dataclass
class CandidateCluster:
    """A cluster of ambiguous candidates for LLM adjudication.

    Matches the schema from plan.md Decision #4.
    """
    cluster_id: str
    primary_entity_type: str
    primary_entity_id: str
    candidate_matches: list[CandidateMatch]
    window_start: datetime | None = None
    window_end: datetime | None = None
    aggregate_delta_paise: int = 0
    has_amount_collision: bool = False


@dataclass
class FuzzyResolvedMatch:
    """A fuzzy match that cleared the score-gap AND no-collision gates.

    Goes directly to IndependentVerifier (bypasses LLM).
    """
    source_entity_type: str
    source_entity_id: str
    source_amount_paise: int
    target_entity_type: str
    target_entity_id: str
    target_amount_paise: int
    score: float
    reasoning_trace: str


@dataclass
class ClusteringOutput:
    """Complete output of FuzzyScore + ClusterCandidates."""
    fuzzy_resolved: list[FuzzyResolvedMatch]
    clusters: list[CandidateCluster]
    no_candidates: list[UnmatchedRecord]


# ═══════════════════════════════════════════════════════════
#  FUZZY SCORING
# ═══════════════════════════════════════════════════════════


def _compute_fuzzy_score(
    source: UnmatchedRecord | DemotedRecord,
    target: UnmatchedRecord | DemotedRecord,
) -> float:
    """Compute a composite fuzzy similarity score between two records.

    Components:
      1. Text similarity (Token Sort Ratio on entity IDs / narration)
      2. Amount proximity (normalised difference)
      3. Timestamp proximity (days apart, if available)

    Returns a float 0.0–1.0.
    """
    scores: dict[str, float] = {}

    # ── 1. Text similarity ───────────────────────────────────
    source_text = _get_text_repr(source)
    target_text = _get_text_repr(target)

    if source_text and target_text:
        token_score = fuzz.token_sort_ratio(source_text, target_text) / 100.0
        partial_score = fuzz.partial_ratio(source_text, target_text) / 100.0
        scores["text"] = max(token_score, partial_score)
    else:
        scores["text"] = 0.0

    # ── 2. Amount proximity ──────────────────────────────────
    source_amt = _get_amount(source)
    target_amt = _get_amount(target)

    if source_amt is not None and target_amt is not None:
        max_amt = max(abs(source_amt), abs(target_amt), 1)
        amt_diff = abs(source_amt - target_amt)
        scores["amount"] = max(0.0, 1.0 - (amt_diff / max_amt))
    else:
        scores["amount"] = 0.0

    # ── 3. Timestamp proximity ───────────────────────────────
    source_ts = _get_timestamp(source)
    target_ts = _get_timestamp(target)

    if source_ts and target_ts:
        delta_days = abs((source_ts - target_ts).total_seconds()) / 86400
        scores["time"] = max(0.0, 1.0 - (delta_days / 7.0))
    else:
        scores["time"] = 0.5  # neutral

    # ── Composite ─────────────────────────────────────────────
    # Weight: text 40%, amount 40%, time 20%
    composite = (
        scores["text"] * 0.4
        + scores["amount"] * 0.4
        + scores["time"] * 0.2
    )

    return round(composite, 4)


def _get_text_repr(record: UnmatchedRecord | DemotedRecord) -> str:
    """Extract a text representation for fuzzy text matching."""
    parts: list[str] = [record.entity_id]
    if isinstance(record, UnmatchedRecord):
        if record.raw_narration:
            parts.append(record.raw_narration)
        if record.canonical_utr:
            parts.append(record.canonical_utr)
    return " ".join(parts)


def _get_amount(record: UnmatchedRecord | DemotedRecord) -> int | None:
    """Extract amount_paise from a record."""
    return record.amount_paise


def _get_timestamp(record: UnmatchedRecord | DemotedRecord) -> datetime | None:
    """Extract timestamp from a record."""
    return record.timestamp


# ═══════════════════════════════════════════════════════════
#  UNION-FIND (DISJOINT SET)
# ═══════════════════════════════════════════════════════════


class _UnionFind:
    """Disjoint-set data structure for entity clustering."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ═══════════════════════════════════════════════════════════
#  CLUSTER CANDIDATES
# ═══════════════════════════════════════════════════════════


def run_fuzzy_and_cluster(
    unmatched_sources: list[UnmatchedRecord | DemotedRecord],
    unmatched_targets: list[UnmatchedRecord | DemotedRecord],
) -> ClusteringOutput:
    """Run FuzzyScore + ClusterCandidates on unmatched records.

    Parameters
    ----------
    unmatched_sources : list
        Source-side unmatched records (typically orders).
    unmatched_targets : list
        Target-side unmatched records (typically settlements, bank_txns).

    Returns
    -------
    ClusteringOutput
        fuzzy_resolved, clusters, and no_candidates.
    """
    if not unmatched_sources or not unmatched_targets:
        # Nothing to match
        no_candidates = [
            r for r in unmatched_sources if isinstance(r, UnmatchedRecord)
        ] + [
            r for r in unmatched_targets if isinstance(r, UnmatchedRecord)
        ]
        return ClusteringOutput(
            fuzzy_resolved=[],
            clusters=[],
            no_candidates=no_candidates,
        )

    # ── 1. Compute all pairwise scores ────────────────────────
    all_pairs: list[FuzzyPairScore] = []

    for source in unmatched_sources:
        for target in unmatched_targets:
            score = _compute_fuzzy_score(source, target)
            if score >= FUZZY_AMBIGUOUS_MIN_SCORE:
                all_pairs.append(FuzzyPairScore(
                    source_entity_type=source.entity_type,
                    source_entity_id=source.entity_id,
                    source_amount_paise=_get_amount(source) or 0,
                    target_entity_type=target.entity_type,
                    target_entity_id=target.entity_id,
                    target_amount_paise=_get_amount(target) or 0,
                    score=score,
                    score_components={},
                    source_timestamp=_get_timestamp(source),
                    target_timestamp=_get_timestamp(target),
                ))

    # ── 2. Identify fuzzy-resolved vs ambiguous ──────────────
    # Group by source entity to check score gaps and collisions
    pairs_by_source: dict[str, list[FuzzyPairScore]] = {}
    for pair in all_pairs:
        key = f"{pair.source_entity_type}:{pair.source_entity_id}"
        pairs_by_source.setdefault(key, []).append(pair)

    fuzzy_resolved: list[FuzzyResolvedMatch] = []
    ambiguous_pairs: list[FuzzyPairScore] = []
    resolved_source_ids: set[str] = set()
    resolved_target_ids: set[str] = set()

    for key, pairs in pairs_by_source.items():
        pairs.sort(key=lambda p: p.score, reverse=True)

        if not pairs:
            continue

        top = pairs[0]
        second_best = pairs[1].score if len(pairs) > 1 else 0.0
        score_gap = top.score - second_best

        # Check for same-amount collision within the window
        has_collision = _has_amount_collision(top, pairs)

        if (
            top.score >= FUZZY_RESOLVED_MIN_SCORE
            and score_gap >= SCORE_GAP_THRESHOLD
            and not has_collision
        ):
            # ── Fuzzy resolved → bypass LLM ──────────────────
            fuzzy_resolved.append(FuzzyResolvedMatch(
                source_entity_type=top.source_entity_type,
                source_entity_id=top.source_entity_id,
                source_amount_paise=top.source_amount_paise,
                target_entity_type=top.target_entity_type,
                target_entity_id=top.target_entity_id,
                target_amount_paise=top.target_amount_paise,
                score=top.score,
                reasoning_trace=(
                    f"FuzzyResolved: score={top.score:.3f}, "
                    f"gap={score_gap:.3f} (≥{SCORE_GAP_THRESHOLD}), "
                    f"no collision."
                ),
            ))
            resolved_source_ids.add(top.source_entity_id)
            resolved_target_ids.add(top.target_entity_id)
        else:
            # ── Ambiguous → clustering ───────────────────────
            ambiguous_pairs.extend(pairs)

    # ── 3. Union-Find clustering on ambiguous pairs ──────────
    uf = _UnionFind()
    entity_data: dict[str, dict[str, Any]] = {}

    for pair in ambiguous_pairs:
        if pair.source_entity_id in resolved_source_ids:
            continue
        if pair.target_entity_id in resolved_target_ids:
            continue

        src_key = f"{pair.source_entity_type}:{pair.source_entity_id}"
        tgt_key = f"{pair.target_entity_type}:{pair.target_entity_id}"

        uf.union(src_key, tgt_key)

        entity_data[src_key] = {
            "entity_type": pair.source_entity_type,
            "entity_id": pair.source_entity_id,
            "amount_paise": pair.source_amount_paise,
            "score": pair.score,
            "timestamp": pair.source_timestamp,
        }
        entity_data[tgt_key] = {
            "entity_type": pair.target_entity_type,
            "entity_id": pair.target_entity_id,
            "amount_paise": pair.target_amount_paise,
            "score": pair.score,
            "timestamp": pair.target_timestamp,
        }

    # Group by connected component root
    components: dict[str, list[str]] = {}
    for key in entity_data:
        root = uf.find(key)
        components.setdefault(root, []).append(key)

    # ── 4. Build CandidateCluster objects ────────────────────
    clusters: list[CandidateCluster] = []

    for root, members in components.items():
        member_data = [entity_data[m] for m in members]

        # Sub-partition if > MAX_CLUSTER_SIZE
        sub_groups = _sub_partition(member_data)

        for sub_idx, sub_members in enumerate(sub_groups):
            if not sub_members:
                continue

            # Pick primary entity (first source-type entity)
            primary = sub_members[0]
            candidates = sub_members[1:] if len(sub_members) > 1 else sub_members

            # Detect amount collision
            amounts = [m["amount_paise"] for m in sub_members]
            has_collision = len(amounts) != len(set(amounts))

            # Compute window
            timestamps = [
                m["timestamp"] for m in sub_members
                if m.get("timestamp") is not None
            ]
            window_start = min(timestamps) if timestamps else None
            window_end = max(timestamps) if timestamps else None

            # Aggregate delta
            if len(amounts) >= 2:
                aggregate_delta = max(amounts) - min(amounts)
            else:
                aggregate_delta = 0

            cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
            if sub_idx > 0:
                cluster_id += f"_sub{sub_idx}"

            clusters.append(CandidateCluster(
                cluster_id=cluster_id,
                primary_entity_type=primary["entity_type"],
                primary_entity_id=primary["entity_id"],
                candidate_matches=[
                    CandidateMatch(
                        entity_type=c["entity_type"],
                        entity_id=c["entity_id"],
                        amount_paise=c["amount_paise"],
                        score=c.get("score", 0.0),
                        timestamp=c.get("timestamp"),
                    )
                    for c in candidates
                ],
                window_start=window_start,
                window_end=window_end,
                aggregate_delta_paise=aggregate_delta,
                has_amount_collision=has_collision,
            ))

    # ── 5. Collect true no-candidates ────────────────────────
    # Records that had zero fuzzy matches above the threshold
    paired_source_ids = {p.source_entity_id for p in all_pairs}
    paired_target_ids = {p.target_entity_id for p in all_pairs}

    no_candidates: list[UnmatchedRecord] = []
    for r in unmatched_sources:
        if (
            isinstance(r, UnmatchedRecord)
            and r.entity_id not in paired_source_ids
            and r.entity_id not in resolved_source_ids
        ):
            no_candidates.append(r)
    for r in unmatched_targets:
        if (
            isinstance(r, UnmatchedRecord)
            and r.entity_id not in paired_target_ids
            and r.entity_id not in resolved_target_ids
        ):
            no_candidates.append(r)

    logger.info(
        "FuzzyScore+ClusterCandidates: %d fuzzy_resolved, %d clusters, "
        "%d no_candidates",
        len(fuzzy_resolved), len(clusters), len(no_candidates),
    )

    return ClusteringOutput(
        fuzzy_resolved=fuzzy_resolved,
        clusters=clusters,
        no_candidates=no_candidates,
    )


def _has_amount_collision(
    top: FuzzyPairScore,
    all_pairs: list[FuzzyPairScore],
) -> bool:
    """Check if any other candidate shares the same amount within the same window."""
    for pair in all_pairs:
        if pair.target_entity_id == top.target_entity_id:
            continue
        if pair.target_amount_paise == top.target_amount_paise:
            # Same amount — check if within the date window
            if top.target_timestamp and pair.target_timestamp:
                delta = abs((top.target_timestamp - pair.target_timestamp).total_seconds())
                if delta <= DEFAULT_WINDOW_DAYS * 86400:
                    return True
            else:
                # Can't verify window — conservative: treat as collision
                return True
    return False


def _sub_partition(
    members: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Sub-partition a member list if it exceeds MAX_CLUSTER_SIZE.

    Uses timestamp buckets (6-hour intervals) for sub-partitioning.
    If timestamps are unavailable, splits by index.
    """
    if len(members) <= MAX_CLUSTER_SIZE:
        return [members]

    # Try timestamp-based sub-partitioning (6-hour buckets)
    has_timestamps = any(m.get("timestamp") is not None for m in members)

    if has_timestamps:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for m in members:
            ts = m.get("timestamp")
            if ts is not None:
                # 6-hour bucket key
                bucket_key = int(ts.timestamp()) // (6 * 3600)
            else:
                bucket_key = 0
            buckets.setdefault(bucket_key, []).append(m)

        # Sort buckets and further split if any are still too large
        result: list[list[dict[str, Any]]] = []
        for _key in sorted(buckets.keys()):
            bucket = buckets[_key]
            while len(bucket) > MAX_CLUSTER_SIZE:
                result.append(bucket[:MAX_CLUSTER_SIZE])
                bucket = bucket[MAX_CLUSTER_SIZE:]
            if bucket:
                result.append(bucket)
        return result

    # Fallback: chunk by index
    return [
        members[i:i + MAX_CLUSTER_SIZE]
        for i in range(0, len(members), MAX_CLUSTER_SIZE)
    ]
