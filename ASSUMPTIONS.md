# Assumptions

Where the spec was ambiguous, these defaults were chosen and documented here
instead of asking questions.

1. **Config style**: `config.py` uses a lightweight `AttrDict` wrapper around
   the parsed YAML (attribute + dict access), not a full Pydantic settings
   model. Pydantic is reserved for LLM-facing strategy schema validation,
   where untrusted input actually needs strict validation.

2. **Entry direction**: A strategy's `entry` condition group is interpreted
   as opening a **long** position. Short entries are not separately
   distinguished in the schema (the spec's example schema has one `entry`
   block, not `entry_long`/`entry_short`); `backtest.allow_short` is wired
   through the backtester and instrument/position-sizing logic for future
   extension, but the current signal interpretation only opens longs. This
   is documented clearly in code comments in `src/backtester.py`.

3. **Stop loss vs take profit same-bar conflict**: if both would trigger
   intrabar, the stop loss is assumed to execute first (conservative).

4. **IS/OOS correlation metric**: computed as the Pearson correlation between
   the IS and OOS daily-return series (aligned by matching the last `n`
   in-sample returns to the first `n` out-of-sample returns, `n = min` of
   the two lengths), as a simple generalization-consistency proxy. This is
   explicitly documented as a rough proxy, not a formal statistical test, in
   the generated Markdown reports.

5. **PnL model**: when `instruments.<symbol>.point_value` is configured,
   futures-style PnL is used: `(exit - entry) * direction * contracts *
   point_value - costs`. If point_value is missing, a return-based
   approximation against current equity is used instead, as specified.

6. **Position sizing without a stop loss**: falls back to
   `backtest.default_contracts`, capped by `max_contracts`, as specified.

7. **Missing trading days**: never forward-filled; the data is used exactly
   as loaded from the `.dat` file. No resampling is performed since the
   input is already daily.

8. **Symbol mismatch resolution**: if the filename-parsed symbol differs
   from the symbol embedded in the binary header, resolution priority is
   configurable via `config.data.symbol_priority` (default:
   `[filename, binary_metadata, config]`).

9. **Free model detection**: a model is treated as free if its OpenRouter id
   ends with `:free`, or if both `pricing.prompt` and `pricing.completion`
   are `"0"`/`0.0` when that data is available from `GET /models`.

10. **LLM analysis step**: the "analyze results and suggest next directions"
    step is a second, separate LLM call per iteration (not the same call
    that generated strategies), so a JSON-shaped strategy response and a
    free-text-ish analysis response are never conflated in one schema.

11. **Backtest fill fallback**: if a bar's `open` is `NaN`/missing, execution
    falls back to that bar's `close`, per the futures-handling section of
    the spec.

12. **`.dat` binary parsing**: the struct layout and OLE-automation-date
    epoch conversion (`(date - 25569) * 86400` seconds) are taken verbatim
    from the reference parser and never altered.

13. **Charts**: monthly returns heatmap uses `resample("ME")` (pandas
    month-end alias); trade PnL histogram pools IS + OOS trades together
    since a single strategy typically has few trades in either window
    alone.

14. **Logging**: a single named logger tree (`local_quant_ai.*`) is
    configured once via `src/utils.setup_logging`, writing to both stdout
    and `logging.file` from config.

15. **CLI `inspect-data`**: implemented once in `scripts/inspect_dat.py`;
    `main.py inspect-data` dynamically imports and calls that script's
    `main()` so behavior is identical from either entry point.

16. **Donchian channel excludes the current bar** (`shift(1)` before the
    rolling max/min). Including today's own high/low in the channel makes a
    `close > donchian_upper` breakout condition almost unsatisfiable (since
    `close <= high` always), which produced zero trades in practice. Standard
    breakout systems (e.g. Turtle Trading) define the channel over the prior
    N bars for exactly this reason. Still strictly non-lookahead -- only past
    bars are used, one bar more conservatively than before.

17. **Iteration now uses a persistent "near-miss leaderboard"** across the
    whole run (`src/orchestrator.py`, `near_miss_history`), not just the
    previous iteration's text summary. Every rejected-but-validated strategy
    is scored by (fewest rejection reasons, then highest combined IS+OOS
    Sharpe) and the current top 3 are fed back into the next iteration's
    prompt verbatim -- exact indicators/entry/exit JSON plus exact rejection
    reasons -- with an explicit instruction to propose refined variants that
    address those specific reasons, alongside a smaller number of fresh
    exploratory strategies. This is an elitism-style mechanism (borrowed
    from genetic-algorithm search): the closest-to-passing candidates are
    preserved and refined across generations instead of being discarded and
    forgotten each iteration. History is capped at 12 records
    (`MAX_NEAR_MISS_HISTORY`) to keep prompt size bounded. Snapshots are
    saved per iteration to `llm_iterations/iteration_NN_near_miss_leaderboard.json`.

18. **The strategy-generation prompt now states the exact acceptance
    thresholds** (`config.validation.*`) instead of only a vague "make
    something good" framing, and includes explicit guardrails against two
    classes of logically-broken-but-schema-valid strategies observed in
    practice: comparing indicators of mismatched numeric scale (e.g. ATR
    vs. a raw price level), and combining a deep-oversold entry filter with
    a medium-term uptrend filter that is rarely true at the same time.

19. **IS/OOS correlation is now the run's primary optimization objective**,
    not just one of several pass/fail gates. This changes three things:
    - `_near_miss_score` in `src/orchestrator.py` ranks the leaderboard by
      highest correlation first (then fewest failed criteria, then combined
      Sharpe as a final tie-breaker), so refinement effort concentrates on
      the most behaviorally-consistent candidates even if they're still
      failing on trade count or drawdown.
    - The strategy-generation prompt explicitly instructs the LLM to treat
      raising correlation as the main goal when refining near-misses --
      preferring simpler logic and normal-range parameters over intricate,
      narrow, regime-specific conditions -- even at some cost to raw Sharpe.
    - `accepted_strategies.csv` is sorted by `is_oos_correlation` descending,
      so the first row is the most consistent accepted strategy, not
      whichever was accepted first chronologically. The end-of-run log also
      prints the best accepted correlation and its strategy name.
    Acceptance itself still requires `min_is_oos_correlation` to be cleared
    like any other criterion in `config.yaml` -- this change affects what
    the *search* prioritizes among rejected/near-miss candidates and among
    multiple accepted strategies, not the accept/reject boundary itself.

20. **IS/OOS correlation is now reliability-adjusted (shrunk) by trade
    count**, to stop low-sample "lucky" correlation numbers from dominating
    the search (`src/evaluation.py::reliability_adjusted_correlation`):

        adjusted = raw_corr * n / (n + 15),  where n = min(is_trades, oos_trades)

    A raw correlation of 0.9 computed from 2-3 trades shrinks to ~0.11
    (effectively distrusted); a raw 0.3 from 100+ trades stays close to
    0.26 (mostly trusted). The prior of 15 trades matches the ballpark of
    `min_trades_oos` in the default config -- a strategy needs to clear
    roughly that many trades before its correlation is taken near face
    value. `n` is deliberately the trade count, not the number of daily
    return observations, since a single multi-year position produces
    thousands of correlated daily bars but only one real behavioral sample;
    using bar-count would let that degenerate case slip through.

    This adjusted value, not the raw one, is now what's used for: the
    acceptance test against `min_is_oos_correlation`, near-miss leaderboard
    ranking (`_near_miss_score`), the `accepted_strategies.csv` sort order,
    the "best accepted strategy" log line, and what the LLM prompt tells
    the model to maximize. Both raw and adjusted values are still recorded
    everywhere (`is_oos_correlation_raw` / `is_oos_correlation` columns in
    CSVs, both fields in `metadata.json`, both lines in the Markdown report)
    so the discount is always visible and auditable, not hidden.

    Caveat still worth knowing: the underlying correlation itself pairs IS
    and OOS daily returns by relative position within each window (last N
    IS days vs. first N OOS days), not by any real calendar or economic
    correspondence -- it remains a rough consistency proxy, not a formal
    statistical test, as already noted in report captions. The reliability
    adjustment fixes the *sample-size gaming* failure mode specifically; it
    does not turn this into a rigorous walk-forward or cross-validated
    metric. A more rigorous version (e.g. fold-based rank-stability testing
    like in a proper walk-forward harness) is a natural next step if this
    matters more than it currently does here.

21. **IS/OOS correlation is now computed on rolling Sharpe ratio curves, not
    raw daily returns** (`src/evaluation.py::_rolling_sharpe` +
    `_paired_correlation`). Rationale: a single day's return is dominated by
    noise, so correlating raw daily returns mostly measured coincidence, not
    behavioral consistency. A rolling Sharpe ratio over `correlation_
    rolling_window_days` (default 60 trading days, ~one quarter, configurable
    in `config.yaml`) is already a smoothed, risk-adjusted statistic, so
    correlating IS vs. OOS rolling-Sharpe curves asks whether the strategy's
    performance texture -- steady edge vs. choppy, strong stretches vs. weak
    ones -- looks similar in both periods, which is closer to what "IS/OOS
    consistency" should mean.

    This still pairs IS and OOS by relative position within each window
    (last n of IS vs. first n of OOS), not by any real calendar or economic
    correspondence -- that limitation is unchanged and is a separate,
    larger issue from the noise problem this fix addresses (a genuinely
    rigorous fix would need fold-based walk-forward comparison instead of
    a single IS/OOS split). It IS a meaningful improvement over raw-return
    correlation for the noise problem specifically, and stacks with the
    trade-count reliability shrinkage from #20 (unchanged, still trade-count
    based, still applied on top of whichever correlation value is computed).

    Fallback: if either window has fewer daily bars than the rolling window
    (e.g. an intentionally narrow OOS slice shorter than 60 days), rolling
    Sharpe can't be computed and the code falls back to the previous raw
    daily-return correlation, logging a warning. `correlation_method` is
    recorded on every evaluation (`"rolling_sharpe_<window>d"` or
    `"raw_daily_return_fallback"`) and surfaced in `metadata.json` and the
    Markdown report so it's always clear which method actually ran.

22. **`src/strategy_export.py` / `main.py export-strategy`** converts a
    validated strategy JSON into a standalone Python class matching the
    `StratGen*`-style interface (`MODE`, `TAG`, `REQUIRED_COLS`,
    `PRICE_COL`, `domain()`, `validate(p)`, `signals(df, p) ->
    (long_entry, long_exit, short_entry, short_exit)`), for use in other
    frameworks (e.g. a GA-based optimizer). Key decisions:
    - Since local_quant_ai strategies only ever define LONG entry/exit
      logic (see #2), `short_entry`/`short_exit` are always emitted as
      all-`False` Series -- never inferred by mirroring/inverting the long
      conditions, since that would fabricate behavior the source strategy
      never specified.
    - `domain()` maps each indicator parameter to a single-item list
      containing the concrete value from the source strategy (not a search
      range) -- the exported class is a faithful re-encoding of one
      specific strategy, not an automatically-widened search space. Widen
      it by hand if you want the receiving GA framework to explore
      neighboring parameter values.
    - The generated file is fully self-contained (`numpy`/`pandas` only,
      no import of local_quant_ai's own `src/` modules) so it can be
      dropped directly into another codebase.
    - Numeric equivalence to local_quant_ai's own indicator/rule engine was
      verified directly (not just "runs without error") across `and`/`or`
      logic, multi-output indicators (MACD, ADX, Bollinger), and both
      column-vs-column and column-vs-numeric-literal `cross_above`/
      `cross_below` conditions.

23. **`domain()` in exported strategies now holds a logical search
    neighborhood, not just the single concrete value the source strategy
    used** (`_widen_int_domain` / `_widen_float_domain` in
    `src/strategy_export.py`). For integer params (period, fast, slow,
    signal, smooth_k, smooth_d): multiplicative steps (x0.5, x0.75, x1.25,
    x1.5, x2.0) plus small additive steps (+-1, +-2), deduplicated and
    clamped to the same bounds `src/schema.py` enforces at generation time
    (e.g. EMA period stays within [2, 300]). For the one continuous param
    (bollinger_bands' num_std): +-0.5/+-1.0 steps, clamped to [1, 4]. The
    original value is always included. Every generated domain was verified
    exhaustively -- every combination across the full parameter grid passes
    `validate()` and runs cleanly through `signals()` -- not just spot
    checked. This is a heuristic neighborhood, not a principled sensitivity
    analysis; widen/narrow the generated lists by hand if a specific
    strategy calls for a different search radius.

24. **Full short-position support added**, superseding assumption #2
    (previously: entry/exit only ever opened longs). Changes, file by file:
    - `src/schema.py`: `StrategySchema` gains optional `short_entry` /
      `short_exit` fields (same `ConditionGroup` shape as `entry`/`exit`).
      A model validator requires both-or-neither -- a short entry with no
      defined exit (or vice versa) fails validation rather than silently
      producing an unmanageable position. `entry`/`exit` are unchanged and
      still mean "long entry/exit"; every strategy JSON generated before
      this change still validates as-is (short fields default to `None`).
      Added `StrategySchema.has_short_logic` (`True` only when both short
      fields are present) as the single source of truth used everywhere
      else in the codebase that needs to know whether a strategy trades
      short.
    - `src/rules.py`: `compute_signals()` now computes four boolean
      columns instead of two -- `entry_signal`/`exit_signal` (long) and
      `short_entry_signal`/`short_exit_signal` (short). The short columns
      are all-`False` when `strategy.short_entry`/`short_exit` are `None`,
      so downstream code (the backtester) never has to branch on whether
      the columns exist, only on whether they're ever `True`.
    - `src/backtester.py`: position-opening logic now checks both
      `entry_signal` and `short_entry_signal` on the previous bar. A short
      is only actually opened when BOTH `config.backtest.allow_short` is
      true AND the strategy itself defines short logic
      (`strategy.has_short_logic`) -- a strategy with no short rule never
      shorts even if the config allows shorting in general, and shorting
      can be globally disabled via config regardless of what strategies
      define. If both long and short entry signals fire on the same bar
      (a case the strategy's own logic didn't resolve), long wins as the
      deterministic tie-break, matching the backtester's original
      long-only behavior as the default side. Exit-signal checking is now
      direction-aware: a long position checks `exit_signal`, a short
      position checks `short_exit_signal`; stop-loss/take-profit price
      calculation was already direction-generic (used `position_dir` as a
      sign multiplier) and needed no changes. Verified with four targeted
      tests: both directions actually trade, PnL sign is correct for
      shorts (price drop -> profit), `allow_short=False` blocks all shorts
      even when the strategy defines them, and a strategy with no short
      logic never shorts even when `allow_short=True`.
    - `src/llm.py`: the strategy-generation prompt documents `short_entry`/
      `short_exit` as optional fields (both-or-neither), with an explicit
      guardrail against just algebraically flipping long conditions --
      short logic should be independently economically plausible (e.g.
      breaking below a channel/support, a confirmed downtrend, a failed
      overbought bounce), using the same allowed indicators/operators as
      the long side.
    - `src/report.py`: Markdown reports now show separate "Entry Rule
      (long)" / "Exit Rule (long)" and, when `has_short_logic` is true,
      "Entry Rule (short)" / "Exit Rule (short)" sections; long-only
      strategies get an explicit "this strategy is long-only" line instead
      of silently omitting the short section.
    - `src/orchestrator.py`: `_save_accepted_strategy`'s `metadata.json`
      now includes `short_entry`/`short_exit` (previously omitted --
      **this was a real bug caught during testing**: since `metadata.json`
      is exactly what `export-strategy` reads back in, an accepted
      short-capable strategy would have silently lost its short logic the
      moment it was exported, with no error or warning. Fixed and verified
      with a save -> reload -> export round trip that confirms short logic
      survives intact).
    - `src/strategy_export.py`: now translates real short logic into the
      exported class's `short_entry`/`short_exit` outputs when the source
      strategy has it, instead of always forcing them to all-`False`. The
      all-`False` fallback is kept, but now only fires for genuinely
      long-only strategies -- still never inferred by mirroring/inverting
      the long conditions. The generated class's docstring dynamically
      states whether it trades both sides or is long-only. Verified with
      an exact cross-check against the project's own reference engine
      (`src/rules.py`) on a real long+short Donchian breakout/breakdown
      strategy -- all four signal columns (long entry/exit, short
      entry/exit) matched bar-for-bar, not just "ran without error".
    - `strategy_exporter.ipynb`: sanity-check cell now reports real
      short-entry/short-exit signal counts when a strategy has short
      logic, instead of always asserting they're `False`.

25. **Follow-up fix to #24: the prompt now actively encourages short logic,
    not just permits it.** Testing this project's earlier prompt changes
    (e.g. the `ema_close_20` vs `ema_20` naming issue) showed free-tier
    models follow the literal example schema closely and rarely reach for
    a field described only as "optional" -- so the initial short-support
    prompt wording (guardrail #4, "SHORT LOGIC IS OPTIONAL") likely would
    have produced few or no short strategies in practice, even though the
    schema/backtester fully supported them. Fixed in `src/llm.py`:
    - The "YOUR TASK" section now explicitly asks for a MIX across the
      batch -- "roughly half should include short_entry/short_exit where a
      short makes genuine economic sense," while still allowing long-only
      where that's the better fit, rather than forcing short logic onto
      every strategy.
    - `_format_near_misses` now prints an explicit `has_short_logic: True/
      False` line for each near-miss candidate (previously the model would
      have had to infer this by parsing the nested strategy JSON), paired
      with an instruction to preserve a near-miss's long/short shape when
      proposing a refined variant, rather than silently dropping or adding
      short logic without reason.
    Verified by building a real prompt with a short-logic near-miss and
    confirming the MIX instruction, the `has_short_logic: True` flag, and
    the preserve-when-refining instruction all render correctly.

26. **Replaced correlation as the primary acceptance/search objective with
    a direct Sharpe-consistency score.** Correlation (raw or reliability-
    adjusted, raw-return or rolling-Sharpe -- see #19-21) was always an
    indirect proxy, and even the rolling-Sharpe version still paired IS/OOS
    windows by arbitrary relative array position, not real correspondence
    -- an unresolved limitation flagged since #21. Replaced with something
    that directly compares what actually matters: are IS Sharpe and OOS
    Sharpe both strong AND close to each other?

        sharpe_consistency = 1 - |is_sharpe - oos_sharpe| / max(|is_sharpe|, |oos_sharpe|, eps)

    (`src/evaluation.py::sharpe_consistency`). Bounded [0, 1]: 1.0 for
    identical Sharpe in both windows, clipped to 0 for opposite signs or
    >100% relative divergence. `config.yaml` gains `min_sharpe_consistency`
    (default 0.7) as the new acceptance criterion, replacing
    `min_is_oos_correlation` (removed from the acceptance path).
    `min_is_sharpe`/`min_oos_sharpe` defaults raised from 0.3/0.0 to 1.0/1.0
    per explicit request -- both windows must clear a real Sharpe bar, not
    just correlate.

    IS/OOS correlation (raw + reliability-adjusted) is still computed and
    recorded everywhere it was before (CSVs, `metadata.json`, Markdown
    reports) for reference, but is explicitly labeled diagnostic-only and
    no longer gates acceptance, ranks the near-miss leaderboard, or is
    mentioned in the LLM prompt at all -- `sharpe_consistency` replaced it
    in every one of those roles (`_near_miss_score`, `accepted_
    strategies.csv` sort order, the end-of-run "best" log line, the
    PRIMARY OBJECTIVE section and near-miss formatting in `src/llm.py`,
    and the analysis-prompt framing).

    Verified: unit-tested `sharpe_consistency` across normal/edge cases
    (matching Sharpes, diverging Sharpes, opposite signs, both-zero); ran
    the full evaluation pipeline on a strategy with a clean, low-noise
    trend in both IS and OOS windows and confirmed it scores consistency
    ~0.999 and is correctly accepted; confirmed the human-readable
    rejection-reason string renders correctly end to end via the CLI.

27. **Full redesign: multi-fold walk-forward evaluation with a genuinely
    untouched final holdout, replacing the single IS/OOS split, plus a
    significant backtesting speed optimization to keep it fast.**

    Motivation: the previous design (a single IS/OOS split, #19-21/#26) had
    two compounding problems. First, its "consistency" signal was always an
    indirect proxy with real limitations (see #21/#26 history). Second, and
    more fundamentally: the LLM-driven search loop repeatedly saw OOS-derived
    summary statistics (OOS Sharpe, OOS trade count, rejection reasons
    quoting OOS numbers) across iterations and was explicitly instructed to
    adjust future strategies to raise them -- a textbook backtest-overfitting
    pattern (see Bailey/Borwein/Lopez de Prado on probability of backtest
    overfitting): repeatedly adapting search based on a fixed holdout's
    aggregate results eventually contaminates that holdout, even without the
    search ever seeing the holdout's raw data directly.

    **New design** (`src/evaluation.py`, fully rewritten):
    - `split_folds`: splits the full history into `config.validation.
      num_folds` contiguous, time-ordered, non-overlapping folds (never
      shuffled). The LAST fold (most recent data) is, by convention
      everywhere in the codebase, the final holdout.
    - `evaluate_strategy_folds`: backtests a strategy on every SEARCH fold
      (all folds except the holdout) and computes `cross_fold_sharpe_
      consistency` -- a direct generalization of the old pairwise
      `sharpe_consistency` formula to N folds (comparing the best and worst
      fold's Sharpe; verified to reduce to the exact same number as the old
      formula when there are exactly 2 values). This -- not correlation, not
      a single IS/OOS pair -- is what the search loop's acceptance and
      near-miss ranking are driven by now.
    - `run_final_holdout`: runs the backtest on the holdout fold exactly
      ONCE, and only ever from `_save_accepted_strategy` in
      `src/orchestrator.py` -- i.e. only after a strategy has already been
      accepted based on search-fold results alone. Its result is recorded
      (`metadata.json`, `report.md`, the `fold_sharpes.png` chart) but is
      NEVER fed into `near_miss_history`, never appears in any LLM prompt,
      and never influences which strategies get refined or accepted. This
      is what makes it a genuine holdout rather than another number the
      loop can indirectly optimize against.
    - New config keys replace the old IS/OOS date-range and correlation
      config entirely: `num_folds` (default 6 -> 5 search folds + 1
      holdout), `min_trades_per_fold`, `min_avg_fold_sharpe`,
      `min_fold_sharpe_floor` (every individual search fold must clear this,
      so one great fold can't hide several bad ones behind a decent
      average), `min_sharpe_consistency`, `max_fold_drawdown`,
      `min_avg_profit_factor`. `min_is_sharpe`/`min_oos_sharpe`/
      `min_is_oos_correlation`/`in_sample_start` etc. are gone.
    - `src/orchestrator.py`: indicators and entry/exit/short signals are now
      computed ONCE per strategy on the FULL history (`_prepare_full_
      signals`), then fold-sliced downstream -- this is both faster (one
      indicator computation instead of one per fold) AND more correct: the
      previous design computed indicators separately on the raw IS slice
      and the raw OOS slice, which gave the OOS window an artificial "cold
      start" (the first `period` bars of OOS were NaN because a rolling
      window restarted from nothing at the OOS boundary, undercounting real
      trading opportunities right at the start of OOS). This was a genuine
      pre-existing bug, not something introduced by this change, surfaced
      and fixed as a natural consequence of restructuring for folds.
    - `src/report.py`: rewritten for fold-based reporting -- per-fold
      metrics tables, a cross-fold summary, a dedicated "Final Holdout
      Result" section explicitly labeled as never having influenced the
      search, and a new `fold_sharpes.png` chart (bar chart of every search
      fold's Sharpe plus the holdout's, in a visually distinct color, with
      both threshold lines drawn) -- the single most direct visual for "are
      these folds actually consistent, and does the untouched holdout agree
      with them". Equity/drawdown/monthly-returns/trade-histogram charts
      now chain the search folds' equity curves into one continuous series
      (each fold's curve offset to continue where the previous one left
      off, purely for plotting -- metrics are still computed per-fold
      independently) with the buy & hold benchmark trimmed to the same
      plotted date range (an earlier draft of this chart let the benchmark
      extend into the holdout period while the strategy line stopped,
      which visually read as "the strategy collapses" rather than "the
      holdout isn't plotted here on purpose" -- caught by inspecting the
      actual rendered chart, not just the code).
    - `src/llm.py`: PRIMARY OBJECTIVE, near-miss formatting, acceptance
      criteria display, and the analysis prompt all rewritten around
      cross-fold consistency with a concrete example (`fold Sharpes of
      [1.3, 1.1, 1.4, 1.2] is exactly what you're aiming for; [2.5, 0.9,
      -0.3, 1.8] is a failure`), and near-miss records now show each
      candidate's actual per-fold Sharpe list so the model can see exactly
      which fold(s) dragged consistency down.
    - `main.py backtest-strategy` / `backtest_single_strategy_file`: runs
      every search fold plus the holdout check standalone, logging each
      fold's Sharpe individually.

    **Backtester speed optimization** (`src/backtester.py`), done because
    moving from 2 backtests/strategy to `num_folds` backtests/strategy (6 by
    default) meant backtest speed mattered more: the per-bar loop previously
    used `df.iloc[i]` and `row.get(...)` -- reconstructing a pandas Series
    object every iteration, which is a well-known slow access pattern.
    Rewritten to extract every needed column as a raw numpy array once,
    before the loop, and index into those arrays directly. Same exact
    algorithm and semantics, only the access pattern changed.
    - **Verified, not assumed**: regression-tested byte-for-byte against
      the original implementation on both a long-only and a long+short
      strategy at real project scale (6000 bars) -- trades and equity
      curves were exactly identical (`DataFrame.equals` / `np.allclose`),
      while running 15-30x faster (487ms -> 16ms long-only; 388ms -> 25ms
      long+short).
    - Combined with computing indicators once instead of redundantly per
      window/fold, a full per-strategy evaluation under the NEW design
      (6 backtests: 5 search folds + 1 holdout) measured 46.9ms, vs. 434ms
      for the OLD design (2 backtests, redundant indicators, unoptimized
      backtester) doing the equivalent work at the same data scale -- a net
      ~9.3x speedup end-to-end despite running 3x more backtests per
      strategy. At this speed, local compute for even a large run (500
      strategies) is under 30 seconds total -- confirming local backtest
      compute remains negligible next to the real constraint on this
      project, OpenRouter's daily request quota.
    - Deliberately stopped at numpy-array-loop optimization rather than a
      fully vectorized/event-driven rewrite (jumping directly between
      trade events instead of looping every bar) or adding a JIT compiler
      (e.g. numba): position sizing depends on running equity, which
      depends on prior trades' realized PnL, making the loop inherently
      sequential/stateful; a fully vectorized version is possible but adds
      real correctness risk (subtle bugs in intrabar stop/target detection
      between event points) for a backtester whose numbers directly decide
      which strategies get accepted -- not a trade-off worth making for
      marginal additional speed once the dominant overhead (row-wise pandas
      access) was already removed.

    **Bug found and fixed during testing**: profit-factor averaging across
    folds (`evaluate_strategy_folds`) originally filtered out any fold with
    an infinite `profit_factor` (which `compute_metrics` correctly returns
    when a fold has zero losing trades -- not an error, the best possible
    outcome) before averaging. When EVERY search fold happened to have zero
    losing trades (plausible with few trades in a strong, clean trend), all
    of them got filtered out, leaving an empty list, which silently
    defaulted the average to 0.0 -- failing a genuinely excellent strategy
    on `min_avg_profit_factor` for the exact opposite reason it should have
    passed easily. Caught by testing the acceptance path (not just
    rejections) end-to-end and noticing an 11.0-avg-fold-Sharpe strategy
    got rejected. Fixed: infinite profit factors are now capped at a large
    finite value (10.0) and included in the average, rather than dropped.

    **Known limitation, stated plainly**: `num_folds` (default 6) trades
    off fold count against per-fold statistical power the same way it did
    in the companion GA optimizer project -- more folds means each one
    covers less history, which interacts with `min_trades_per_fold` and
    indicator lookback periods (a 200-day EMA needs a fold long enough to
    warm up and still trade). If a strategy's indicators use unusually long
    lookbacks, either raise `min_trades_per_fold` cautiously downward or
    lower `num_folds` -- there's no automatic check that fold length is
    adequate for a given strategy's lookback, the same way there wasn't one
    for the old IS/OOS windows either.

28. **`cross_fold_sharpe_consistency` was scale-biased against strong
    performers -- replaced it, and stopped ranking/selecting on it as the
    primary objective.** Caught by inspecting a real run
    (`local_quant_run_20260903_231211_7d77c9/all_experiments.csv`): "MACD
    Histogram Momentum v2" beat the accepted "ADX Trend Breakout" on EVERY
    single search fold's Sharpe (fold Sharpes [0.475, 1.236, 0.829, 0.956,
    1.851] vs. [0.380, 0.378, 0.518, 0.512, 0.414] -- strict fold-by-fold
    dominance) yet was rejected while the strictly worse strategy was
    accepted and reported as the run's best.

    Root cause: the old formula, `1 - (max-min)/max(|max|,|min|)`, divides
    the best-to-worst spread by the BEST fold's own magnitude. A strategy
    whose best fold is exceptional needs its folds to cluster proportionally
    *tighter* to pass the same bar as a mediocre strategy -- i.e. the better
    a strategy's ceiling, the harder its own floor got before "inconsistent"
    triggered. This is backwards: a strategy that is strongly profitable in
    literally every fold (MACD's worst fold, 0.475, was still better than
    every one of ADX's folds) was penalized purely for having an even
    better fold elsewhere, while a strategy that was merely mediocre in a
    narrow, low-variance band scored as "more consistent" and won.

    **Fix** (`src/evaluation.py::cross_fold_sharpe_consistency`): replaced
    with `min(fold_sharpes) / mean(fold_sharpes)` (clipped to [0, 1], 0 if
    the mean isn't positive) -- "what fraction of the average edge survives
    in the worst fold", independent of how good the best fold happens to
    be. Verified against the same run's data: ADX 0.858 (unchanged
    ordering-wise, still clearly consistent), MACD 0.445 (up from 0.257,
    reflecting that its worst fold is still solidly profitable, just not as
    good as its best), while genuinely fluky strategies stay low (e.g. "ADX
    Trend Filter Donchian Symmetric", worst fold only 14% of its average,
    scores 0.141).

    Two more changes were needed because the formula alone wasn't
    sufficient -- ranking and gating both still had to stop treating
    consistency as more important than actual performance:
    - `min_sharpe_consistency` (`config.yaml`) lowered from 0.7 to 0.4 to
      match the new formula's scale (0.7 under the new formula demands the
      worst fold retain 70% of the average, a much stricter bar than the
      old formula's 0.7 -- recalibrated by checking which previously-
      rejected-for-consistency-only strategies in the same run should now
      pass: MACD Histogram Momentum v2, Bollinger Squeeze Breakout, and
      several others with a solid worst fold now clear the bar, while
      strategies with a near-zero or negative-relative worst fold still
      correctly fail).
    - Ranking was never actually about raw performance even before this
      formula existed: `_near_miss_score` (`src/orchestrator.py`), the
      `accepted_strategies.csv` sort order, and the end-of-run "best"
      log line all used `sharpe_consistency` as the PRIMARY sort key,
      meaning even among strategies that already cleared every bar, the
      flattest one won over the most profitable one. Switched the primary
      key on all three to `avg_fold_sharpe`, with `sharpe_consistency`
      demoted to a tie-breaker -- consistency is now purely a floor to
      clear during acceptance, never a reason to prefer a weaker strategy
      once accepted.
    - `src/llm.py` (PRIMARY OBJECTIVE, near-miss formatting/ranking
      description, refinement instructions, and the analysis prompt)
      rewritten to match: average fold Sharpe is now stated explicitly as
      what strategies are ranked and selected on, consistency is described
      as a floor rather than something to maximize by giving up Sharpe, and
      the worked example was corrected (the old "[2.5, 0.9, -0.3, 1.8] is a
      failure" example was actually failing on the negative fold, not on
      spread -- reworded so the example doesn't imply high variance alone
      is the problem).

29. **A large share of `all_experiments.csv`'s "invalid" rows traced to two
    concrete, fixable mismatches between what the prompt told the LLM and
    what the validator actually required** -- not LLM unreliability in
    general. Audited every "invalid" row across all `results/*/
    all_experiments.csv` (159 total across the project's run history) by
    error message:
    - **28 rows**: `indicators List should have at least 2 items` --
      `StrategySchema.indicators` required `min_length=2`, but the PRIMARY
      OBJECTIVE prompt text explicitly tells the model to "simplify the
      logic or remove an overfit-prone filter" when refining a near-miss
      rejected for low consistency, which can legitimately mean collapsing
      to a single indicator (e.g. a pure Donchian breakout, or a pure RSI
      mean-reversion rule) -- the schema was rejecting the exact simplification
      the model was being told to make. Fixed: `min_length` lowered to 1
      (`src/schema.py`); prompt's "between 2 and 4 indicators" bullet
      updated to "between 1 and 4" (`src/llm.py`).
    - **~30+ rows**: `Unknown column referenced ... 'bb_<period>_2.0_...'`
      -- `INDICATOR_REFERENCE` in `src/llm.py` documented the bollinger_bands
      column name as literally `bb_<period>_<num_std>_...`, but the actual
      naming logic (`src/schema.py::_indicator_output_columns` and
      `src/indicators.py::bollinger_bands`, which already agreed with each
      other) collapses a whole-number `num_std` to an int (2.0 -> "2", so
      the real column is `bb_20_2_upper`, not `bb_20_2.0_upper`) -- the
      model was correctly following the (wrong) documentation and getting
      rejected for it. Fixed: `INDICATOR_REFERENCE`'s bollinger_bands entry
      now states the int-collapse rule explicitly with worked examples for
      both a whole and fractional `num_std`.
    - Smaller, harder-to-fully-eliminate patterns also seen repeatedly
      (added explicit guardrails for each rather than leaving them to
      recur): referencing an indicator's output column without declaring
      that indicator (e.g. `rsi_close_14` used in a condition with no `rsi`
      entry in `indicators`); inventing a suffixed name for a base OHLCV
      column instead of using it plain (e.g. `close_close` instead of
      `close`); and trying to express arithmetic inside a condition (e.g.
      `{"op": "-"}` for a `close - 2*atr` style stop level, which the
      comparison-only condition DSL has no way to represent) -- pointed at
      `stop_loss_pct`/`take_profit_pct` instead.
    Not touched: schema validation errors that are the validator correctly
    doing its job (out-of-bounds parameters, disallowed operators/indicator
    names, malformed JSON shape) -- those are working as intended, not a
    prompt/schema mismatch.


## Universe pipeline (2026-10)

- **Instrument data**: tick size and point value for every symbol come from the
  `.dat` header (`bpv` = point value), not from the hand-maintained
  `instruments:` table (only `@TY` was configured before, so other symbols
  would have been sized/costed with the notional fallback).
- **Per-symbol capital**: `max(initial_capital, backtest.capital_per_notional *
  median 1-contract notional)` so one index-future contract is not a heavily
  levered bet on $100k while a grain contract is a sensible position; keeps
  drawdown thresholds comparable across the universe.
- **Excluded series**: continuous contracts with non-positive prices (> 0.05% of
  bars) or > 0.3% daily moves over 20% are dropped -- percent-based stops and
  signals are meaningless on them (cocoa, heating oil, RBOB, soybeans, soy meal,
  gasoil, Brent/WTI customs, crude, cotton, OJ, milk and a few others; see the
  "Skipping ..." log lines). Results on them (Sharpe 5-7 for a plain Donchian
  breakout) were artifacts.
- **Universe score** = mean over symbols of the per-symbol average search-fold
  Sharpe clipped to [-1.5, 1.5]. **Accepted** = >= 10% of symbols pass every
  per-symbol walk-forward criterion, >= 60% of symbols profitable, score >= 0.15
  (`universe.accept`). Calibrated on classic textbook strategies (donchian, RSI
  pullback, zscore reversion, EMA/ADX, momentum, ...): they score <= ~0.2 with
  0-5 of 51 symbols passing; random-ish ones pass ~1%.
- **Holdout**: the final fold of each symbol is evaluated only for accepted
  strategies and never enters prompts or the leaderboard ranking.
- **Selection bias**: with thousands of candidates some will look good by
  chance; breadth across ~50 markets, the holdout, and the lookahead test are the
  defenses. Treat accepted strategies as candidates for out-of-sample work, not
  as proven edges.
- **Stop/target**: `None` for sl/tp means "use the config default" (2% / 4%),
  exactly as before -- the backtester has no "no stop" mode.

## Frequency + learning changes (2026-10-03)

- **Why**: real runs ranked strategies trading 2-5x/yr first (a few lucky
  positions) and showed no improvement across iterations. Causes found in the
  raw outputs: lost batches (bare `===` delimiters), reasoning models truncating
  at 0-1 usable blocks of 10, helper shadowing (`adx = adx(14)`), operator
  precedence bugs (`a > b | c`), copying of the prompt's example strategy.
- **Frequency**: per-fold minimum trades = max(10, 10/yr * fold years); universe
  score x min(1, trades_per_year/10); acceptance needs >= 10 trades/yr and >= 8%
  of symbols passing every criterion (lowered from 10% because the stricter
  per-fold trade minimum makes passing rarer). Memory format v3 archives the v2
  leaderboard (scored without frequency).
- **Local tuning** picks the best of N parameter neighbours on SEARCH folds, which
  inflates search scores slightly by construction; the reserved holdout is the
  check. Leaderboard/prompt examples are distinct per code body.

## Local search round (2026-10-04)

- **Why**: real runs after the frequency fix traded 20-30x/yr but were unprofitable
  (~38% of markets profitable, mean score -0.08) and the old tuner touched 3
  parents once. LLM requests are the scarce resource; local compute is nearly free.
- **Fitness** = frequency-adjusted score + `fitness_pass_weight` (0.4) x share of
  symbols passing every walk-forward criterion. Leaderboard, parents, tuner and
  top-N all rank by fitness; `score` is still reported. The search can therefore
  trade a little mean Sharpe for broader per-symbol passing.
- **Selection bias**: the tuner promotes the best of ~24 candidates per step on the
  SEARCH folds; scores of tuned strategies are optimistically biased by
  construction. The holdout (reports) is the check. Many tuned strategies in one
  lineage are near-identical; leaderboard views and top-N keep one per code body.
- **Seeds** are textbook strategies (donchian, RSI dip, z-score reversion, ...), not
  novel ideas; they are a baseline for the LLM and a parent supply for the search.
- **Worker processes** reload the universe once each (~0.5s) and receive only small
  strategy specs; on Windows scripts that call the loop need an
  `if __name__ == "__main__":` guard. `run.tune_processes: false` uses threads.
