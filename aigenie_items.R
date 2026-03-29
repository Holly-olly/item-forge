#!/usr/bin/env Rscript
# =============================================================================
#  aigenie_items.R — AI-GENIE Psychometric Validation Pipeline
#  Russell-Lasalandra, Christensen & Golino (2025)
#
#  Adapted from:
#    laralee/AIGENIE (MIT License)
#    https://github.com/laralee/AIGENIE
# =============================================================================
#
#  Runs Steps 2–5 of the AI-GENIE methodology on any item pool:
#    Step 2 — Initial EGA  (TMFG + glasso, pick best NMI/ARI)
#    Step 3 — Iterative UVA  (redundancy removal, wTO ≤ 0.20)
#    Step 4 — Sparse vs full embedding comparison
#    Step 5 — Iterative bootEGA  (stability ≥ 0.75, 100 iterations)
#    Step 6 — Final EGA + NMI/ARI report
#
#  USAGE:
#    Rscript aigenie_items.R --items items.csv --embeddings embeddings.csv
#    Rscript aigenie_items.R --items items.csv --embeddings embeddings.csv \
#                            --output-dir results/ --boot-iter 200 --seed 99
#
#  REQUIRED INPUTS:
#    --items      CSV with at least 2 columns: scale name, item text
#                 (col positions configurable via --scale-col / --item-col)
#    --embeddings CSV produced by efa_items.py (rows=dims, cols=item_0,item_1,…)
#                 OR any matrix in that orientation
#
#  OPTIONAL:
#    --scale-col  Column name or 1-based index for scale labels  [default: 1]
#    --item-col   Column name or 1-based index for item text     [default: 2]
#    --output-dir Directory for output files  [default: aigenie_results_<timestamp>]
#    --uva-cutoff wTO threshold for UVA   [default: 0.20]
#    --boot-iter  Bootstrap iterations    [default: 100]
#    --stab-thr   Stability threshold     [default: 0.75]
#    --seed       Random seed             [default: 42]
#
#  OUTPUTS (in --output-dir):
#    stable_items.csv      — stable item indices + EGA community
#    stable_items_full.csv — stable items with text + scale + match flag
#    ega_results.csv       — pipeline metrics (NMI, ARI, item counts)
#    final_report.txt      — human-readable summary
#    removed_items.csv     — items removed by UVA / bootEGA
#
#  REQUIREMENTS:
#    install.packages(c("EGAnet", "igraph"))
# =============================================================================

# ── Argument parsing ──────────────────────────────────────────────────────────
parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  if (length(raw) == 0 || "--help" %in% raw || "-h" %in% raw) {
    cat(sub("^#!/usr/bin/env Rscript\n", "", readLines(sys.function()(0)), fixed = TRUE))
    cat(readLines(con = textConnection(
      "Usage: Rscript aigenie_items.R --items <file> --embeddings <file> [options]\n"
    )))
    quit(save = "no", status = 0)
  }

  get <- function(flag, default = NULL) {
    i <- which(raw == flag)
    if (length(i) == 0) return(default)
    if (i + 1 > length(raw)) stop(paste("Flag", flag, "requires a value"))
    raw[i + 1]
  }

  list(
    items       = get("--items"),
    embeddings  = get("--embeddings"),
    scale_col   = get("--scale-col",  "1"),
    item_col    = get("--item-col",   "2"),
    output_dir  = get("--output-dir", NULL),
    uva_cutoff  = as.numeric(get("--uva-cutoff", "0.20")),
    boot_iter   = as.integer(get("--boot-iter",  "100")),
    stab_thr    = as.numeric(get("--stab-thr",   "0.75")),
    seed        = as.integer(get("--seed",        "42")),
    no_header   = "--no-header" %in% raw
  )
}

args <- parse_args()

if (is.null(args$items))      stop("--items is required")
if (is.null(args$embeddings)) stop("--embeddings is required")
if (!file.exists(args$items))      stop(paste("File not found:", args$items))
if (!file.exists(args$embeddings)) stop(paste("File not found:", args$embeddings))

# Output directory
ts         <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_dir <- if (!is.null(args$output_dir)) args$output_dir else paste0("aigenie_results_", ts)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("\nOutputs → %s/\n", output_dir))

# ── Package setup ─────────────────────────────────────────────────────────────
for (pkg in c("EGAnet", "igraph")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s ...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}
suppressPackageStartupMessages({ library(EGAnet); library(igraph) })
cat(sprintf("EGAnet %s | igraph %s\n",
            packageVersion("EGAnet"), packageVersion("igraph")))

# ── Step 0: Load data ──────────────────────────────────────────────────────────
cat("\n── Step 0: Load data ──\n")

if (args$no_header) {
  items_df <- read.csv(args$items, header = FALSE, stringsAsFactors = FALSE,
                       check.names = FALSE, col.names = c("scale", "item_text"))
  args$scale_col <- "scale"; args$item_col <- "item_text"
} else {
  items_df <- read.csv(args$items, stringsAsFactors = FALSE, check.names = FALSE)
}
items_df[] <- lapply(items_df, function(x) if (is.character(x)) trimws(x) else x)

resolve_col <- function(spec, df) {
  idx <- suppressWarnings(as.integer(spec))
  if (!is.na(idx)) return(colnames(df)[idx])
  if (spec %in% colnames(df)) return(spec)
  stop(sprintf("Column '%s' not found. Available: %s",
               spec, paste(colnames(df), collapse = ", ")))
}
scale_col <- resolve_col(args$scale_col, items_df)
item_col  <- resolve_col(args$item_col,  items_df)

items_df <- items_df[, c(scale_col, item_col)]
colnames(items_df) <- c("scale", "item_text")
items_df$scale     <- trimws(items_df$scale)
items_df$item_text <- trimws(items_df$item_text)
items_df           <- items_df[complete.cases(items_df), ]
items_df$item_index <- paste0("item_", seq_len(nrow(items_df)) - 1)

scale_names  <- sort(unique(items_df$scale))
n_scales     <- length(scale_names)
scale_to_id  <- setNames(seq_along(scale_names), scale_names)
items_df$community_id <- scale_to_id[items_df$scale]

cat(sprintf("  Items: %d across %d scale(s)\n", nrow(items_df), n_scales))
for (s in scale_names) {
  cat(sprintf("    %s: %d items\n", s, sum(items_df$scale == s)))
}

# Load embeddings: rows=dims, cols=items
embed_raw    <- read.csv(args$embeddings, row.names = 1, check.names = FALSE)
embed_matrix <- as.matrix(embed_raw)   # (dim × n_items)

if (ncol(embed_matrix) != nrow(items_df)) {
  stop(sprintf(
    "Embedding columns (%d) != item rows (%d). Check file orientation: embeddings must be dim × items.",
    ncol(embed_matrix), nrow(items_df)
  ))
}
# Align column names with item indices
colnames(embed_matrix) <- items_df$item_index

known_communities <- setNames(items_df$community_id, items_df$item_index)

cat(sprintf("  Embeddings: %d dims × %d items\n", nrow(embed_matrix), ncol(embed_matrix)))

# ── Helpers ───────────────────────────────────────────────────────────────────

run_ega <- function(data, model) {
  tryCatch(
    EGA(data = data, model = model, algorithm = "walktrap",
        plot.EGA = FALSE, verbose = FALSE),
    error = function(e) { cat(sprintf("  EGA(%s) error: %s\n", model, conditionMessage(e))); NULL }
  )
}

calc_nmi <- function(wc, known) {
  common <- intersect(names(wc), names(known))
  if (length(common) < 2) return(NA_real_)
  tryCatch(
    igraph::compare(as.integer(wc[common]), as.integer(known[common]), method = "nmi"),
    error = function(e) NA_real_
  )
}

calc_ari <- function(wc, known) {
  common <- intersect(names(wc), names(known))
  if (length(common) < 2) return(NA_real_)
  tryCatch(
    igraph::compare(as.integer(wc[common]), as.integer(known[common]), method = "adjusted.rand"),
    error = function(e) NA_real_
  )
}

extract_stabilities <- function(boot_result) {
  for (path_fn in list(
    function(r) r$stability$item.stability$item.stability$empirical.dimensions,
    function(r) r$item.stability$item.stability$empirical.frequencies,
    function(r) r$stability$item.stability$empirical.dimensions,
    function(r) r$item.stability$empirical.frequencies,
    function(r) r$stability$empirical.frequencies
  )) {
    s <- tryCatch(path_fn(boot_result), error = function(e) NULL)
    if (!is.null(s) && length(s) > 0 && is.numeric(s)) return(s)
  }
  NULL
}

sparsify <- function(mat) {
  apply(mat, 2, function(col) {
    thr <- mean(abs(col))
    col[abs(col) < thr] <- 0
    col
  })
}

extract_reduced <- function(uva_result) {
  for (field in c("reduced_data", "data", "EGA")) {
    x <- uva_result[[field]]
    if (!is.null(x) && (is.matrix(x) || is.data.frame(x)))
      return(as.matrix(x))
  }
  NULL
}

fmt_pct <- function(x) if (is.na(x)) "n/a" else sprintf("%.1f%%", x * 100)

# Track removed items
removed_uva  <- character(0)
removed_boot <- character(0)

# ── Step 2: Initial EGA ───────────────────────────────────────────────────────
cat("\n── Step 2: Initial EGA ──\n")

ega_tmfg   <- run_ega(embed_matrix, "TMFG")
ega_glasso <- run_ega(embed_matrix, "glasso")

nmi_tmfg   <- if (!is.null(ega_tmfg))   calc_nmi(ega_tmfg$wc,   known_communities) else NA
nmi_glasso <- if (!is.null(ega_glasso)) calc_nmi(ega_glasso$wc,  known_communities) else NA
ari_tmfg   <- if (!is.null(ega_tmfg))   calc_ari(ega_tmfg$wc,   known_communities) else NA
ari_glasso <- if (!is.null(ega_glasso)) calc_ari(ega_glasso$wc,  known_communities) else NA

cat(sprintf("  TMFG:   NMI=%s  ARI=%s\n", fmt_pct(nmi_tmfg),   fmt_pct(ari_tmfg)))
cat(sprintf("  glasso: NMI=%s  ARI=%s\n", fmt_pct(nmi_glasso), fmt_pct(ari_glasso)))

best_nmi_init <- max(nmi_tmfg, nmi_glasso, na.rm = TRUE)
best_ari_init <- if (!is.na(nmi_tmfg) && !is.na(nmi_glasso) && nmi_tmfg >= nmi_glasso) ari_tmfg else ari_glasso
best_model    <- if (!is.na(nmi_tmfg) && (is.na(nmi_glasso) || nmi_tmfg >= nmi_glasso)) "TMFG" else "glasso"
cat(sprintf("  Best initial model: %s\n", best_model))

initial_nmi <- best_nmi_init
initial_ari <- best_ari_init

# ── Step 3: Iterative UVA ─────────────────────────────────────────────────────
cat(sprintf("\n── Step 3: Iterative UVA (wTO ≤ %.2f) ──\n", args$uva_cutoff))

current_data <- embed_matrix
uva_sweeps   <- 0

repeat {
  uva_result <- tryCatch(
    UVA(data = current_data, cut.off = args$uva_cutoff,
        reduce = TRUE, reduce.method = "remove", verbose = FALSE),
    error = function(e) { cat(sprintf("  UVA error: %s\n", conditionMessage(e))); NULL }
  )
  if (is.null(uva_result)) break

  uva_sweeps <- uva_sweeps + 1
  reduced    <- extract_reduced(uva_result)

  if (is.null(reduced)) {
    cat("  Converged — no more redundancies.\n"); break
  }

  gone <- setdiff(colnames(current_data), colnames(reduced))
  if (length(gone) == 0) { cat("  Converged — no more redundancies.\n"); break }

  removed_uva  <- c(removed_uva, gone)
  current_data <- reduced
  cat(sprintf("  Sweep %d: removed %d → %d remaining\n",
              uva_sweeps, length(gone), ncol(current_data)))
}

cat(sprintf("  UVA done: %d removed in %d sweep(s) → %d items remain\n",
            length(removed_uva), uva_sweeps, ncol(current_data)))

# ── Step 4: Sparse vs full ────────────────────────────────────────────────────
cat("\n── Step 4: Sparse vs Full ──\n")

sparse_data  <- sparsify(current_data)
remaining    <- colnames(current_data)
known_remain <- known_communities[remaining]

ega_full_s4   <- run_ega(current_data, "TMFG")
ega_sparse_s4 <- run_ega(sparse_data,  "TMFG")

nmi_full   <- if (!is.null(ega_full_s4))   calc_nmi(ega_full_s4$wc,   known_remain) else NA
nmi_sparse <- if (!is.null(ega_sparse_s4)) calc_nmi(ega_sparse_s4$wc, known_remain) else NA

cat(sprintf("  Full NMI: %s   Sparse NMI: %s\n", fmt_pct(nmi_full), fmt_pct(nmi_sparse)))

use_sparse    <- !is.na(nmi_sparse) && !is.na(nmi_full) && nmi_sparse > nmi_full
pipeline_data <- if (use_sparse) sparse_data else current_data
cat(sprintf("  Using: %s embeddings\n", if (use_sparse) "sparse" else "full"))

# ── Step 5: Iterative bootEGA ─────────────────────────────────────────────────
cat(sprintf("\n── Step 5: Iterative bootEGA (%d iterations, stability ≥ %.2f) ──\n",
            args$boot_iter, args$stab_thr))

boot_sweeps <- 0

repeat {
  if (ncol(pipeline_data) <= max(4, n_scales)) {
    cat(sprintf("  Safety stop: only %d items left.\n", ncol(pipeline_data))); break
  }

  boot_result <- tryCatch(
    bootEGA(
      data                  = pipeline_data,
      iter                  = args$boot_iter,
      seed                  = args$seed + boot_sweeps,
      model                 = "TMFG",
      algorithm             = "walktrap",
      type                  = "resampling",
      plot.typicalStructure = FALSE,
      verbose               = FALSE
    ),
    error = function(e) { cat(sprintf("  bootEGA error: %s\n", conditionMessage(e))); NULL }
  )
  if (is.null(boot_result)) break

  boot_sweeps  <- boot_sweeps + 1
  stabilities  <- extract_stabilities(boot_result)

  if (is.null(stabilities)) {
    cat("  Cannot extract stabilities. Printing structure:\n")
    str(boot_result, max.level = 3)
    break
  }

  unstable <- names(stabilities[stabilities < args$stab_thr])
  cat(sprintf("  Sweep %d: stable=%d  unstable=%d\n",
              boot_sweeps, sum(stabilities >= args$stab_thr), length(unstable)))

  if (length(unstable) == 0) break

  removed_boot  <- c(removed_boot, unstable)
  pipeline_data <- pipeline_data[
    , !colnames(pipeline_data) %in% unstable, drop = FALSE
  ]
  cat(sprintf("    Removed: %s\n",
              paste(unstable[seq_len(min(10, length(unstable)))], collapse = ", "),
              if (length(unstable) > 10) sprintf("… +%d more", length(unstable) - 10) else ""))
  cat(sprintf("    Remaining: %d items\n", ncol(pipeline_data)))
}

# ── Step 6: Final EGA + NMI ───────────────────────────────────────────────────
cat("\n── Step 6: Final EGA ──\n")

final_remaining <- colnames(pipeline_data)
known_final     <- known_communities[final_remaining]
final_ega       <- run_ega(pipeline_data, "TMFG")

final_nmi <- if (!is.null(final_ega)) calc_nmi(final_ega$wc, known_final) else NA
final_ari <- if (!is.null(final_ega)) calc_ari(final_ega$wc, known_final) else NA

cat(sprintf("  Final NMI: %s  ARI: %s\n", fmt_pct(final_nmi), fmt_pct(final_ari)))
cat(sprintf("  Items: %d start → %d after UVA → %d final\n",
            ncol(embed_matrix), ncol(current_data), length(final_remaining)))

# ── Build community → scale mapping (majority vote) ──────────────────────────
if (!is.null(final_ega)) {
  wc_final <- final_ega$wc[final_remaining]
  comm_ids <- sort(unique(wc_final))
  comm_to_scale <- sapply(comm_ids, function(cid) {
    items_in_comm <- names(wc_final[wc_final == cid])
    scales_in_comm <- known_final[items_in_comm]
    # majority vote: which theoretical scale has most items here
    tbl <- table(
      scale_names[scales_in_comm]
    )
    names(which.max(tbl))
  })
  names(comm_to_scale) <- comm_ids
} else {
  comm_to_scale <- setNames(rep(NA_character_, length(unique(known_final))),
                            unique(known_final))
}

# ── Save outputs ──────────────────────────────────────────────────────────────
cat("\n── Saving outputs ──\n")

# 1. stable_items.csv
stable_df <- data.frame(
  item_index         = final_remaining,
  assigned_community = if (!is.null(final_ega)) final_ega$wc[final_remaining] else NA_integer_,
  stringsAsFactors   = FALSE
)
write.csv(stable_df, file.path(output_dir, "stable_items.csv"), row.names = FALSE)

# 2. stable_items_full.csv — with text, true scale, EGA community, match flag
stable_full <- merge(stable_df, items_df[, c("item_index", "item_text", "scale")],
                     by = "item_index", all.x = TRUE)
stable_full$ega_scale <- comm_to_scale[as.character(stable_full$assigned_community)]
stable_full$match     <- stable_full$scale == stable_full$ega_scale
stable_full           <- stable_full[order(stable_full$scale, stable_full$item_index), ]
write.csv(stable_full, file.path(output_dir, "stable_items_full.csv"), row.names = FALSE)

# 3. removed_items.csv
all_removed <- c(removed_uva, removed_boot)
if (length(all_removed) > 0) {
  removed_df <- merge(
    data.frame(item_index = all_removed,
               removed_by = c(rep("UVA", length(removed_uva)),
                              rep("bootEGA", length(removed_boot))),
               stringsAsFactors = FALSE),
    items_df[, c("item_index", "item_text", "scale")],
    by = "item_index", all.x = TRUE
  )
  write.csv(removed_df, file.path(output_dir, "removed_items.csv"), row.names = FALSE)
}

# 4. ega_results.csv
results_df <- data.frame(
  initial_nmi        = initial_nmi,
  initial_ari        = initial_ari,
  final_nmi          = final_nmi,
  final_ari          = final_ari,
  nmi_improvement    = final_nmi - initial_nmi,
  ari_improvement    = final_ari - initial_ari,
  items_start        = ncol(embed_matrix),
  items_after_uva    = ncol(current_data),
  items_after_boot   = length(final_remaining),
  items_removed_uva  = length(removed_uva),
  items_removed_boot = length(removed_boot),
  uva_sweeps         = uva_sweeps,
  boot_sweeps        = boot_sweeps,
  embedding_type     = if (use_sparse) "sparse" else "full",
  best_model_initial = best_model,
  stringsAsFactors   = FALSE
)
write.csv(results_df, file.path(output_dir, "ega_results.csv"), row.names = FALSE)

# 5. final_report.txt
n_match  <- if (!is.null(stable_full$match)) sum(stable_full$match, na.rm = TRUE) else NA
n_stable <- length(final_remaining)

per_scale_lines <- sapply(scale_names, function(s) {
  n_kept    <- sum(stable_full$scale == s, na.rm = TRUE)
  n_matched <- sum(stable_full$scale == s & stable_full$match, na.rm = TRUE)
  n_start   <- sum(items_df$scale == s)
  sprintf("    %-35s %d/%d kept  (%d match EGA)", s, n_kept, n_start, n_matched)
})

report_lines <- c(
  strrep("=", 65),
  "AI-GENIE VALIDATION REPORT",
  strrep("=", 65),
  sprintf("Date:         %s", format(Sys.time(), "%Y-%m-%d %H:%M")),
  sprintf("Items file:   %s", args$items),
  sprintf("Embeddings:   %s", args$embeddings),
  sprintf("Parameters:   UVA cutoff=%.2f  bootIter=%d  stab≥%.2f  seed=%d",
          args$uva_cutoff, args$boot_iter, args$stab_thr, args$seed),
  "",
  "── Item pool ──",
  sprintf("  Start:        %d items", ncol(embed_matrix)),
  sprintf("  Scales (%d):   %s", n_scales, paste(scale_names, collapse = ", ")),
  per_scale_lines,
  "",
  "── Step 2 — Initial EGA ──",
  sprintf("  TMFG:   NMI=%s  ARI=%s", fmt_pct(nmi_tmfg),   fmt_pct(ari_tmfg)),
  sprintf("  glasso: NMI=%s  ARI=%s", fmt_pct(nmi_glasso), fmt_pct(ari_glasso)),
  sprintf("  Best:   %s", best_model),
  "",
  "── Step 3 — UVA ──",
  sprintf("  Removed: %d items in %d sweep(s)", length(removed_uva), uva_sweeps),
  if (length(removed_uva) > 0) sprintf("  Items:   %s", paste(removed_uva, collapse = ", ")) else "  Items:   (none)",
  "",
  "── Step 4 — Embedding type ──",
  sprintf("  Full NMI: %s   Sparse NMI: %s   → using %s",
          fmt_pct(nmi_full), fmt_pct(nmi_sparse), if (use_sparse) "sparse" else "full"),
  "",
  "── Step 5 — bootEGA ──",
  sprintf("  Removed: %d items in %d sweep(s)", length(removed_boot), boot_sweeps),
  if (length(removed_boot) > 0) sprintf("  Items:   %s", paste(removed_boot, collapse = ", ")) else "  Items:   (none)",
  "",
  "── Step 6 — Final EGA ──",
  sprintf("  NMI:  %s  (Δ %+.1f%%)", fmt_pct(final_nmi),
          if (!is.na(final_nmi) && !is.na(initial_nmi)) (final_nmi - initial_nmi)*100 else NA),
  sprintf("  ARI:  %s  (Δ %+.1f%%)", fmt_pct(final_ari),
          if (!is.na(final_ari) && !is.na(initial_ari)) (final_ari - initial_ari)*100 else NA),
  sprintf("  Items: %d start → %d final", ncol(embed_matrix), n_stable),
  sprintf("  Theory match: %d/%d (%.1f%%)", n_match, n_stable,
          if (!is.na(n_match) && n_stable > 0) n_match/n_stable*100 else NA),
  "",
  "── Stable items ──",
  unlist(lapply(scale_names, function(s) {
    sub <- stable_full[stable_full$scale == s, ]
    header <- sprintf("  %s (%d items):", s, nrow(sub))
    if (nrow(sub) == 0) return(c(header, "    (none)"))
    rows <- apply(sub, 1, function(r) {
      m <- if (!is.na(r["match"]) && r["match"] == "TRUE") "✓" else "✗"
      sprintf("    [%s] %s", m, r["item_text"])
    })
    c(header, rows)
  })),
  "",
  strrep("=", 65),
  sprintf("Outputs saved to: %s/", output_dir),
  "  stable_items.csv         — item indices + EGA community",
  "  stable_items_full.csv    — item text + scale + match flag",
  "  removed_items.csv        — items removed by UVA / bootEGA",
  "  ega_results.csv          — pipeline metrics",
  "  final_report.txt         — this report",
  strrep("=", 65)
)

report_text <- paste(report_lines, collapse = "\n")
cat("\n", report_text, "\n", sep = "")
writeLines(report_text, file.path(output_dir, "final_report.txt"))

cat(sprintf("\n✓ Done.\n  Initial NMI: %s → Final NMI: %s\n  Items: %d → %d  |  Theory match: %d/%d\n",
            fmt_pct(initial_nmi), fmt_pct(final_nmi),
            ncol(embed_matrix), n_stable, n_match, n_stable))
