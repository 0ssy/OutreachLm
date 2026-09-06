"""Corrected compute and memory accounting for a 70B-capacity sparse engine.

WHY THIS FILE EXISTS
    An earlier review of the 70B-on-a-laptop plan concluded "102,185 years"
    and declared it impossible. That review contained an accounting error and
    the conclusion does not follow from it.

    It used  compute = 6 * P_total * T.  For a sparsely-activated model the
    correct form is  compute = 6 * P_active * T,  where P_active is the
    parameters that actually participate per token. This is standard practice
    for mixture-of-experts models and is how Switch Transformer and Mixtral
    report their training cost. It also assumed T = 1e12 tokens, which the
    plan never specified.

    With active-parameter accounting and a token budget matched to the active
    size, the compute requirement lands in days, not millennia.

WHAT SURVIVED THE MEASUREMENTS AND WHAT DID NOT
    Survived, and is load-bearing:
      * Batch-token fusion (claim 2). Streaming weights once per step and
        applying them to a large token block is what makes the I/O affordable.
        This is the single most important idea in the plan.
      * Block sparsity (claim 1), when the active set is a CONTIGUOUS EXPERT
        TILE rather than scattered scalars. Measured: expert tiles run at
        137-158 GFLOP/s, ABOVE the 130.5 GFLOP/s large-GEMM figure, because a
        small tile stays resident in cache. Routing gather costs 1-4%.

    Did not survive, and the design must not depend on it:
      * "Zero-cycle conditional branching" over inactive weights. Measured:
        masking a dense matrix and multiplying anyway runs at 0.5x dense --
        SLOWER. The speedup comes from gathering active tiles into contiguous
        memory and running a smaller dense GEMM, not from skipping.
      * Unstructured 0.1% sparsity. Measured 201x fewer FLOPs but only ~12
        GFLOP/s -- the active submatrix is too small to use the machine. Hence
        the EXPERT_MIN_PARAMS floor below.
      * Integer arithmetic being cheaper. Measured: int32 matmul is 0.02x
        fp32, i.e. 41x SLOWER, because this CPU (Zen 3) has no VNNI and there
        is no optimised integer GEMM path. Ternary's real benefit is MEMORY:
        2 bits against 32 is 16x less traffic, and this workload is I/O bound.
        Compute in fp32; store in ternary.
      * A 1.25 GB tile resident in L3. Measured L3 is 16 MB.
      * "4 tokens per 512-bit register." This CPU is AVX2: 256-bit registers,
        no AVX-512, no VNNI (torch reports capability AVX2, Family 25 Model
        80). Register-blocking across tokens is also already what the GEMM
        kernel does. The available headroom came from routing BLOCK SIZE --
        145.6 GFLOP/s at 128 tokens/expert against 173.0 at 512 -- and is
        already taken in SPARSE_GFLOPS below.
      * Gradient-magnitude expert pruning as a free speedup. Two problems.
        It starves the pruned pool: a pruned expert emits no gradient, so its
        magnitude estimate never refreshes, so it is never reselected --
        simulated at 95.8% of experts never updated, and this project already
        measured the same mechanism at row granularity in rung 3e. Adding
        random re-probing fixes coverage (34.9% never updated) but spends the
        compute the pruning was meant to save. More fundamentally, active
        parameters ARE the model: compute is 6 * P_active * T, so cutting
        70 -> 20 active experts cuts P_active from 70M to 20M. That is a
        smaller model, not a faster one.
      * Bitwise delta compression "eliminating network delays". Cross-node
        communication is already 1.9% of the two-node budget under disjoint
        expert ownership, because experts have a single owner and their
        gradients never cross the link. Removing it entirely is a 1.02x
        speedup, and the ternary merge rules resolve a conflict this design
        does not have.

WHAT THIS DOES NOT CLAIM
    A 70B-total / 70M-active model is not a 70B dense model. Published MoE
    runs use far lower sparsity ratios -- Mixtral is 47B/13B (3.6x), Switch
    Transformer explored up to roughly 100x with measurable degradation. At
    1000x this design is well outside demonstrated territory.

    ROUTING EVIDENCE, AS IT STANDS AFTER MEASUREMENT
      * On a task where routing IS the bottleneck by construction (tokens
        carry a latent concept, the target is a concept-specific map), larger
        pools keep helping to a 1024x sparsity ratio and the gap WIDENS with
        training: relative loss 0.00018 / 0.00007 / 0.00005 for 128 / 512 /
        2048 experts at 5400 steps. An earlier flat reading at 900 steps was
        an optimisation floor, not a result.
      * On REAL text (the project's own 3.22 MB corpus, character-level, no
        residual bypass so the expert block is the only path to the head),
        pool size makes NO measurable difference: 2.4724 / 2.4736 / 2.4729 for
        8 / 64 / 512 experts. Sharding costs 0.58%.

    The most likely reading is that the language probe's trunk is too weak to
    expose structure worth routing on -- it mean-pools context and has no
    attention, so tokens look alike to the router regardless of how many
    experts are available. A router cannot discover structure the
    representation does not represent. That is consistent with production MoE,
    where expert layers sit inside attention blocks.

    So the large-table design is SUPPORTED on synthetic routing tasks and
    UNVALIDATED on language. Resolving it needs a trunk with attention, which
    is a materially larger experiment than anything run so far, and it is the
    single most important open risk to this architecture.

MEASURED CONSTANTS -- all from this machine, none assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

# src/sparse_engine/bench_expert_shape.py and bench_expert_tuned.py
# 173 GFLOP/s sustained at the tuned point: a 2048x512 expert tile (1.05M
# params) fed 512 tokens, min of 9 trials. Throughput is shape-sensitive --
# 145.6 at 128 tokens, 173.0 at 512, 148.0 at 2048 -- so the routing block
# size is a real tuning parameter, not an implementation detail.
# A single-trial sweep suggested 190; that did not survive repetition.
# src/sparse_engine/bench_routed_throughput.py and profile_dispatch.py --
# throughput of the ROUTED path, forward plus backward, useful FLOPs only.
#
# This was originally 173, taken from an isolated GEMM benchmark, which made
# the cost model ~4x optimistic until validate_budget.py caught it. It was
# then set to a flat 45.4 -- but that was measured at 16-128 tokens per
# expert, and throughput depends strongly on that ratio because it sets both
# the GEMM's row count and the padding waste:
#
#     tokens/expert     GFLOP/s     padding waste
#                32        62.7               35%
#               128       113.8                -
#               512       123.4                -
#              1024       134.4                -
#              2048       136.1               16%
#
# These are a RE-MEASUREMENT. The first pass reported almost exactly half
# these values because it timed with a single warmup iteration, so PyTorch
# thread-pool spin-up landed inside the measured region. Caught by
# fast_dispatch.py reporting 114 GF/s for a shape the curve claimed was 57.
# Re-measured with 3 warmup iterations and min-of-6 on an idle machine.
#
# A single constant is therefore wrong in both directions depending on
# configuration. `routed_gflops` interpolates this measured curve, so a
# config is priced at its own operating point. The 70B design runs at
# 65,536 x 70 / 70,000 = 66 tokens/expert; the 2B design at 819.
_ROUTED_CURVE = ((32, 62.7), (128, 113.8), (512, 123.4),
                 (1024, 134.4), (2048, 136.1))
ISOLATED_GEMM_GFLOPS = 173.0   # kept only to document the discrepancy
DENSE_GFLOPS = 130.5           # large dense GEMM
ROUTING_OVERHEAD = 0.0         # now inside the measured routed figures


def routed_gflops(tokens_per_expert: float) -> float:
    """Measured routed throughput at a given tokens-per-expert ratio.

    Log-linear interpolation between measured points, clamped at both ends so
    an out-of-range configuration is priced conservatively rather than
    extrapolated into fantasy.
    """
    import math

    pts = _ROUTED_CURVE
    if tokens_per_expert <= pts[0][0]:
        return pts[0][1]
    if tokens_per_expert >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if tokens_per_expert <= x1:
            f = ((math.log(tokens_per_expert) - math.log(x0))
                 / (math.log(x1) - math.log(x0)))
            return y0 + f * (y1 - y0)
    return pts[-1][1]

# src/sparse_engine/bench_stream_bandwidth.py
DISK_READ_GBPS = 4.10
DISK_WRITE_GBPS = 0.26         # 16x slower than read -- shapes the design
RAM_BW_GBPS = 10.6

# src/sparse_engine/bench_link.py -- cross-laptop WiFi, 144.4 Mbps negotiated.
# Bounded rather than directly measured (no listener on the far machine);
# 0.0135 GB/s is the OPTIMISTIC end of 35-75% link efficiency, so using it
# makes every two-node estimate below generous rather than flattering.
LINK_GBPS = 0.0135
LINK_RTT_MS = 0.80

# src/sparse_engine/run_delta_probe.py -- ternary delta coding against the
# PACKED 2-BIT form (not fp32, which would inflate the figure 16x for free).
# Fixed-width uint16 gaps + 4-per-byte states: a flat 2.25 bytes per flip,
# against 2.88 for the varint encoder this replaced. 12.4x at a 256-step sync
# interval, 48.0x at 16. Flips accumulate sublinearly, so longer intervals
# cost less per step even though each delta is larger.
DELTA_COMPRESSION = 12.4
DELTA_SYNC_DEFAULT = 256

# Platform (per node; the second laptop is identical)
RAM_GB = 15.4
DISK_FREE_GB = 119.5
L3_MB = 16.0
CORES = 6

# Below this an expert tile stops filling the vector units: the 0.1%
# unstructured case measured 12 GFLOP/s against 150 for a proper tile.
EXPERT_MIN_PARAMS = 512 * 1024

# PER-EXPERT DATA -- A WARNING, NOT A BLOCKER, AND EXPLICITLY UNVALIDATED.
#
# An earlier version of this file hard-blocked any configuration below 20
# assignments per expert parameter, derived from run_attn_probe.py, where
# larger pools scored monotonically worse (1.9960 / 2.0735 / 2.1058 at
# 8 / 64 / 512 experts). That gate was WRONG and is withdrawn.
#
# The probe could not test what it was read as testing. Its arms saw 5.3M
# tokens against expert-parameter counts of 0.29M / 2.36M / 18.87M, i.e.
# 18.06 / 2.26 / 0.28 tokens per parameter. The 8-expert arm sat almost
# exactly at Chinchilla-optimal; the 512-expert arm was 71x starved. The
# comparison was therefore between a well-trained small model and a badly
# undertrained large one, and it would come out the same way regardless of
# whether routing scales. On a 3.2 MB corpus no large pool CAN be trained, so
# no pool comparison at that data scale is informative.
#
# The probe's own trajectory said as much and was over-read: the 512-expert
# arm was 12.3% behind at step 1400 and 5.5% behind at 2600 -- closing, not
# diverging. That is the signature of under-training, and this project has now
# corrected the same premature-reading error four times.
#
# Dense Chinchilla also does not transfer directly. Its ~20 tokens/parameter
# is derived where every parameter participates in every token; in a routed
# model an expert sees roughly k/E of the stream by construction. Published
# sparse models operate with total-parameter token ratios far below 20 and
# still beat compute-matched dense baselines, so a hard cliff at 20 would
# forbid architectures that demonstrably work.
#
# What survives: there are diminishing returns to pool size at a fixed token
# budget, the effect is real, and its magnitude is NOT established by anything
# measured here. So this reports rather than blocks.
SOFT_TOKENS_PER_EXPERT_PARAM = 1.0

# src/sparse_engine/decompose_step.py -- costs the budget originally OMITTED.
#
# The first version of this model priced only 6 * P_active * T over
# SPARSE_GFLOPS plus a bytes/bandwidth term, and validate_budget.py showed
# real steps running 9-70x slower than it predicted. Decomposing a step
# located the gap exactly:
#
#     phase                64 experts   256 experts
#     disk read                    0%            1%
#     ternary unpack              24%           27%
#     routing (topk)               1%            0%
#     expert math                 27%           17%   <- the only term priced
#     optimiser                   48%           55%
#
# So the model was accounting for 17-27% of a step. Both missing terms are
# added below with measured rates. Neither is avoidable by tuning: weights
# are stored 2-bit and computed in fp32, so every served expert is unpacked,
# and every served expert's moments are updated.
UNPACK_PARAMS_PER_SEC = 155e6   # byte-wide LUT; 94e6 with nibble shifts
OPTIM_PARAMS_PER_SEC = 72e6     # SparseExpertAdam, measured at two shapes

# src/sparse_engine/validate_budget.py -- residual, measured end to end.
#
# History, because it shows what the residual actually was:
#   9-70x   before unpack and optimiser terms existed at all
#   2.0-3.3x after adding them, with a flat SPARSE_GFLOPS = 45.4
#   1.6x    after making routed throughput shape-aware, measured at the
#            scale this design actually runs at
#
# The residual is itself scale-dependent -- 3.2x at 64 tokens/expert, 1.6x at
# 512-2048 -- because fixed per-step overhead amortises. The value below is
# measured in the regime the 2B and 70B configurations occupy (512-2048
# tokens/expert); small configurations are priced optimistically by it, which
# is stated rather than hidden.
#
# Most of what looked like unexplained overhead was the tokens-per-expert
# dependence being flattened into one constant -- the model was pricing every
# configuration at one operating point. What remains is genuine unmodelled
# cost: allocation, Python-level dispatch, topk, and the gather/scatter around
# the GEMM.
#
# Applied rather than ignored, because the alternative is reporting a number
# measurement says is optimistic. Not a fitted parameter: it is the measured
# mean over four shapes that were not used to derive any constant, and
# test_budget_validation.py re-measures it, and it has been revised upward
# once already (1.63 -> 1.92) when in-regime validation showed the model
# still 11-25% optimistic. Revising it up is the correct direction to err:
# an under-predicting model turns a "feasible" verdict into a missed
# deadline.
MODEL_SLACK = 1.92

SECONDS_PER_DAY = 86400.0


@dataclass
class Config:
    """One candidate engine configuration.

    `nodes` and `strategy` describe how work is split across the two
    laptops. The link between them measures ~0.0135 GB/s against 4.10 GB/s
    local disk, so the strategy is chosen by that ratio, not by preference.
    """

    total_params: float
    active_params_per_token: float
    tokens: float
    expert_params: float
    tokens_per_step: int
    latent_bytes: int = 1          # int8 latent master
    weight_bits: int = 2           # ternary, packed
    latent_update_every: int = 16  # expert-level gradient accumulation
    nodes: int = 1
    strategy: str = "expert_sharded"
    trunk_params: float = 2e8      # embeddings + attention + router
    trunk_sync_every: int = 16
    delta_sync_every: int = DELTA_SYNC_DEFAULT
    d_model: int = 2048
    layers: int = 8
    overlap_io: bool = True
    micro_batch: int = 65536

    # ---------------------------------------------------------------- sizes
    @property
    def n_experts(self) -> float:
        return self.total_params / self.expert_params

    @property
    def experts_per_token(self) -> float:
        return self.active_params_per_token / self.expert_params

    @property
    def weight_gb(self) -> float:
        return self.total_params * self.weight_bits / 8 / 1e9

    @property
    def latent_gb(self) -> float:
        return self.total_params * self.latent_bytes / 1e9

    @property
    def shard_fraction(self) -> float:
        """Fraction of the expert table one node stores."""
        if self.nodes > 1 and self.strategy in ("expert_sharded",
                                                "expert_parallel"):
            return 1.0 / self.nodes
        return 1.0

    @property
    def storage_per_node_gb(self) -> float:
        return (self.weight_gb + self.latent_gb) * self.shard_fraction

    @property
    def steps(self) -> float:
        """Steps are shared: with N nodes each step covers N token blocks."""
        return self.tokens / (self.tokens_per_step * self.nodes)

    # ------------------------------------------------------------- compute
    @property
    def compute_flops(self) -> float:
        """6 * P_active * T -- the corrected form."""
        return 6.0 * self.active_params_per_token * self.tokens

    @property
    def tokens_per_expert_per_step(self) -> float:
        """Assignments each served expert receives in one step.

        This is the quantity that sets routed GEMM efficiency: it is both the
        row count of each expert's block and the driver of padding waste.
        """
        served = self.experts_touched_per_step
        return (self.tokens_per_step * self.experts_per_token
                / max(served, 1.0))

    @property
    def sparse_gflops(self) -> float:
        return routed_gflops(self.tokens_per_expert_per_step)

    @property
    def compute_seconds(self) -> float:
        eff = self.sparse_gflops * 1e9 / (1.0 + ROUTING_OVERHEAD)
        return self.compute_flops / eff / self.nodes

    # ------------------------------------------------------------------ io
    @property
    def experts_touched_per_step(self) -> float:
        """A large token block routes to essentially every local expert, so
        the shard is streamed once per step. This is why batch-token fusion
        matters: the cost is per STEP, not per token."""
        local = self.n_experts * self.shard_fraction
        return min(local, self.tokens_per_step * self.experts_per_token)

    @property
    def io_seconds(self) -> float:
        local = max(self.n_experts * self.shard_fraction, 1.0)
        frac = self.experts_touched_per_step / local
        w = self.weight_gb * self.shard_fraction
        lat = self.latent_gb * self.shard_fraction
        read = (w + lat) * frac / DISK_READ_GBPS
        write = lat * frac / DISK_WRITE_GBPS / self.latent_update_every
        return (read + write) * self.steps

    # ---------------------------------------------------------------- comm
    @property
    def comm_gb_per_step(self) -> float:
        if self.nodes < 2:
            return 0.0
        if self.strategy == "expert_sharded":
            # Only the replicated trunk is synchronised, and only every
            # trunk_sync_every steps. Experts are owned outright, so their
            # gradients never cross the link.
            return (self.trunk_params * 4 / 1e9) / self.trunk_sync_every
        if self.strategy == "data_parallel":
            # Every node holds every expert, so every expert gradient must be
            # reduced across the link.
            return (self.latent_gb) / self.latent_update_every
        if self.strategy == "data_parallel_delta":
            # Same topology, but the synchronised object is a ternary DELTA
            # rather than a latent gradient. Measured flip rates give 12.1x
            # compression of the packed 2-bit form at a 256-step interval,
            # and flips accumulate sublinearly so longer intervals are
            # cheaper per step: 0.00567 GB/step against 0.02790 at 16 steps.
            return (self.weight_gb / DELTA_COMPRESSION
                    / self.delta_sync_every)
        if self.strategy == "expert_parallel":
            # All-to-all: each token's hidden vector travels to a remote
            # expert and the result returns, at every layer.
            remote = 1.0 - 1.0 / self.nodes
            return (self.tokens_per_step * self.experts_per_token * remote
                    * self.d_model * 4 * 2 * self.layers) / 1e9
        raise ValueError(self.strategy)

    @property
    def comm_seconds(self) -> float:
        return self.comm_gb_per_step / LINK_GBPS * self.steps

    @property
    def unpack_seconds(self) -> float:
        """2-bit -> fp32 expansion for every expert served, every step.

        Unavoidable given the storage format: the design stores ternary to
        halve I/O and computes in fp32 because int32 matmul measured 0.02x
        fp32 on this CPU. Caching unpacked weights would need
        4 bytes x total_params, which is 280 GB at the 70B pool.
        """
        served = self.experts_touched_per_step * self.expert_params
        return served * self.steps / UNPACK_PARAMS_PER_SEC

    @property
    def optim_seconds(self) -> float:
        """Moment updates for every expert served, amortised over
        latent_update_every steps."""
        served = self.experts_touched_per_step * self.expert_params
        return (served * self.steps / OPTIM_PARAMS_PER_SEC
                / self.latent_update_every)

    @property
    def total_seconds(self) -> float:
        """Wall-clock, modelling streaming as overlapped with compute.

        Summing every phase assumed the cores idle while the next block is
        read, which double-buffering exists to prevent. Disk reads at
        4.10 GB/s are issued by a background thread while the current step
        computes, so only the I/O and link terms hide.

        Unpack and optimiser do NOT hide: they are CPU work on the same cores
        as the expert math, so they add. Treating them as overlappable was the
        error that made this model 9-70x optimistic.
        """
        cpu = self.compute_seconds + self.unpack_seconds + self.optim_seconds
        cpu *= MODEL_SLACK
        io_side = self.io_seconds + self.comm_seconds
        if not self.overlap_io:
            return cpu + io_side
        return max(cpu, io_side)

    @property
    def overlap_saving_days(self) -> float:
        cpu = (self.compute_seconds + self.unpack_seconds
               + self.optim_seconds) * MODEL_SLACK
        naive = cpu + self.io_seconds + self.comm_seconds
        return (naive - self.total_seconds) / SECONDS_PER_DAY

    @property
    def days(self) -> float:
        return self.total_seconds / SECONDS_PER_DAY

    # ------------------------------------------------------------ feasible
    def working_set_gb(self) -> float:
        """RAM needed at any instant: activations plus the live expert tiles.

        Activations are sized by MICRO_BATCH, not by tokens_per_step. Those
        are different quantities and conflating them was a modelling error:
        tokens_per_step controls how often weights are re-read from disk,
        while micro_batch controls how many tokens are in flight at once.
        Gradient accumulation lets one streaming pass serve many micro-batches
        against the same resident experts, which is batch-token fusion taken
        to its conclusion -- the weights are fetched once and amortised over
        every token that needs them.

        Tying the two together made large streaming steps appear to need
        103 GB of RAM, when the actual requirement is set by the micro-batch.
        """
        live_experts = min(self.n_experts, 64.0)
        tiles = live_experts * self.expert_params * (
            self.weight_bits / 8 + 4.0        # ternary + fp32 unpacked
        ) / 1e9
        acts = min(self.micro_batch, self.tokens_per_step) * \
            self.d_model * 4 * 3 / 1e9
        return tiles + acts

    @property
    def assignments_per_expert(self) -> float:
        """Token-expert assignments each expert receives over training.

        A pool does not merely have to fit and route -- every expert in it has
        to be TRAINED, and each one only sees the tokens routed to it. This is
        the quantity the token budget has to satisfy per expert, and it is
        independent of whether the active-parameter budget is satisfied.
        """
        total = self.tokens * self.experts_per_token
        return total / self.n_experts

    @property
    def tokens_per_expert_param(self) -> float:
        return self.assignments_per_expert / self.expert_params

    def problems(self, max_days: float = 90.0) -> list[str]:
        out = []
        if self.expert_params < EXPERT_MIN_PARAMS:
            out.append(
                f"expert too small ({self.expert_params / 1024:.0f}K params); "
                f"below {EXPERT_MIN_PARAMS / 1024:.0f}K the tile drops to "
                f"~12 GFLOP/s"
            )
        if self.storage_per_node_gb > DISK_FREE_GB:
            out.append(
                f"storage {self.storage_per_node_gb:.1f} GB/node exceeds "
                f"{DISK_FREE_GB} GB free"
            )
        ws = self.working_set_gb()
        if ws > RAM_GB * 0.7:
            out.append(f"working set {ws:.1f} GB too close to {RAM_GB} GB RAM")
        if self.tokens < 15.0 * self.active_params_per_token:
            out.append(
                f"token budget {self.tokens:.2e} is under-trained for "
                f"{self.active_params_per_token:.2e} active params "
                f"(Chinchilla wants ~20x)"
            )
        if self.comm_seconds > 0.25 * self.compute_seconds:
            out.append(
                f"link-bound: {self.comm_seconds / SECONDS_PER_DAY:.1f} days "
                f"of communication against "
                f"{self.compute_seconds / SECONDS_PER_DAY:.1f} days of compute"
            )
        if self.days > max_days:
            out.append(f"runtime {self.days:.0f} days exceeds {max_days:.0f}")
        return out

    def warnings(self) -> list[str]:
        """Real but unquantified risks. Reported, never blocking.

        Blocking on these would encode a threshold this project has not
        measured, which is exactly the error that produced the withdrawn
        20-tokens-per-expert-parameter gate.
        """
        out = []
        r = self.tokens_per_expert_param
        if r < SOFT_TOKENS_PER_EXPERT_PARAM:
            out.append(
                f"{r:.2f} assignments per expert parameter. Diminishing "
                f"returns to pool size are real at a fixed token budget, but "
                f"the threshold is NOT established -- see the withdrawn gate "
                f"in this module. A pool of {self.viable_pool_size():,.0f} "
                f"experts, or {self.tokens_for_this_pool():.2e} tokens, would "
                f"reach {SOFT_TOKENS_PER_EXPERT_PARAM:.0f}."
            )
        return out

    def viable_pool_size(self) -> float:
        """Pool size that would reach the soft per-expert data target."""
        total = self.tokens * self.experts_per_token
        return total / (SOFT_TOKENS_PER_EXPERT_PARAM * self.expert_params)

    def tokens_for_this_pool(self) -> float:
        """Token budget that would bring this pool to the soft target."""
        need = (SOFT_TOKENS_PER_EXPERT_PARAM * self.expert_params
                * self.n_experts)
        return need / self.experts_per_token

    def feasible(self, max_days: float = 90.0) -> bool:
        return not self.problems(max_days)


def report(name: str, c: Config) -> None:
    print(f"\n=== {name} ===")
    print(f"  nodes / strategy    {c.nodes} x  {c.strategy}")
    print(f"  total params        {c.total_params:.2e}")
    print(f"  active per token    {c.active_params_per_token:.2e} "
          f"({c.active_params_per_token / c.total_params:.2%})")
    print(f"  experts             {c.n_experts:,.0f} x "
          f"{c.expert_params / 1e6:.1f}M, {c.experts_per_token:.0f}/token")
    print(f"  tokens              {c.tokens:.2e} "
          f"({c.tokens / c.active_params_per_token:.0f}x active params)")
    print(f"  storage/node        {c.storage_per_node_gb:.1f} GB "
          f"of {DISK_FREE_GB} free")
    print(f"  RAM working set     {c.working_set_gb():.2f} GB")
    print(f"  expert math         {c.compute_seconds / SECONDS_PER_DAY:.1f} d"
          f"  ({c.tokens_per_expert_per_step:,.0f} tok/expert -> "
          f"{c.sparse_gflops:.0f} GF/s)")
    print(f"  ternary unpack      {c.unpack_seconds / SECONDS_PER_DAY:.1f} d")
    print(f"  optimiser           {c.optim_seconds / SECONDS_PER_DAY:.1f} d")
    print(f"  streaming I/O       {c.io_seconds / SECONDS_PER_DAY:.1f} d")
    print(f"  cross-node comm     {c.comm_seconds / SECONDS_PER_DAY:.1f} d "
          f"({c.comm_gb_per_step * 1000:.1f} MB/step)")
    if c.overlap_io:
        print(f"  overlap saving      {c.overlap_saving_days:.1f} d "
              f"(streaming hidden behind compute)")
    print(f"  TOTAL               {c.days:.1f} days")
    probs = c.problems()
    print(f"  verdict             {'FEASIBLE' if not probs else 'BLOCKED'}")
    for p in probs:
        print(f"      - {p}")
    for w in c.warnings():
        print(f"      ! {w}")


# tokens_per_step is a STREAMING batch, amortising each weight fetch over
# more tokens; micro_batch bounds activation RAM independently via gradient
# accumulation. Raising the former from 65,536 to 4.2M takes routed
# throughput from 44 to 78 GF/s (more tokens per expert) and cuts unpack
# 62-fold, without changing RAM at all.
BASE = dict(
    total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
    tokens_per_step=4194304, micro_batch=65536,
)


def main() -> None:
    print("70B-capacity sparse engine: TWO-NODE accounting")
    print(f"link measured at {LINK_GBPS} GB/s against {DISK_READ_GBPS} GB/s "
          f"local disk -- a {DISK_READ_GBPS / LINK_GBPS:.0f}x gap that")
    print("selects the parallelism strategy rather than merely tuning it.")

    single = Config(expert_params=1e6, nodes=1, **BASE)
    report("ONE NODE (previous result, for reference)", single)

    report("TWO NODES, expert-parallel all-to-all", Config(
        expert_params=35e6, nodes=2, strategy="expert_parallel", **BASE,
    ))

    report("TWO NODES, data-parallel with gradient sync", Config(
        expert_params=1e6, nodes=2, strategy="data_parallel", **BASE,
    ))

    best = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                  **BASE)
    report("TWO NODES, disjoint experts + trunk sync", best)

    dpd = Config(expert_params=1e6, nodes=2, strategy="data_parallel_delta",
                 **BASE)
    report("TWO NODES, data-parallel with ternary delta sync", dpd)

    print(f"\n  assignments per expert   "
          f"{dpd.assignments_per_expert:.2e}")
    print(f"  per expert PARAM         "
          f"{dpd.tokens_per_expert_param:.2f} "
          f"(soft target {SOFT_TOKENS_PER_EXPERT_PARAM:.0f}, not a gate)")

    print("\n  The per-expert data ratio is reported as a WARNING above.")
    print("  It was briefly a hard gate at 20 tokens/expert-param, which")
    print("  blocked this design. That gate came from a probe whose own arms")
    print("  were 18.06 / 2.26 / 0.28 tokens per parameter -- it compared a")
    print("  Chinchilla-optimal small model against a 71x starved large one,")
    print("  so it could not measure routing at all. Gate withdrawn.")

    print(f"\n  sharded {best.days:.1f} d but each token routes among "
          f"{best.n_experts / 2:,.0f} experts;")
    print(f"  delta-sync {dpd.days:.1f} d with all {dpd.n_experts:,.0f} "
          f"reachable.")
    print("  Measured routing cost of sharding, real corpus with an")
    print("  attention trunk: 2.3174 against 2.1058, a 10.0% penalty.")


if __name__ == "__main__":
    main()
