# Multi-seed parametric bootstrap simulation (AUC only)
# Block A: load data and construct true parameters (once)
# Block B: per-seed response generation -> MCMC -> AUC
# Block C: aggregation across seeds

library(Rcpp)
library(RcppArmadillo)
library(coda)

# =============================================================================
# User configuration
# =============================================================================

USER_CONFIG <- list(
  model_name = "lstm_ae",
  D          = 1,
  PRE        = "robust",

  # workspace_root: project root directory.
  # If NULL, it is resolved as ../../ relative to this script.
  # workspace_root = "/path/to/your/workspace",

  # MCMC settings
  N_iter            = 50000,
  burn_in_ratio     = 0.2,
  thin              = 10,
  proposal_sd_alpha = 1.5,
  proposal_sd_beta  = 0.5,
  proposal_sd_w     = 0.4,
  tau2              = 0.001,
  nu2               = 2.5,

  # Simulation settings
  n_seeds     = 5,
  master_seed = NULL,   # if numeric, seed list generation is reproducible

  # To reproduce a previous run exactly, supply its seed list here
  # (see seed_log.txt of that run). Overrides n_seeds and master_seed.
  # e.g., sim_seeds_override = c(3935997, 6077361, 9769836, 5256305, 7253934),
  sim_seeds_override = NULL,

  # Number of unique actions per item (fixed, independent of D)
  problem_names = c(paste0("ps1_", 1:7), paste0("ps2_", 1:7)),
  n_prob        = 14,
  N_j_vec       = c(119, 141, 124, 47, 48, 182, 200,
                    156, 289,  70, 206, 112, 146, 185)
)

# =============================================================================
# Step 0. Environment setup
# =============================================================================

cat("=============================================================\n")
cat("  Multi-Seed Parametric Bootstrap Simulation (AUC only)\n")
cat("=============================================================\n\n")

if (is.null(USER_CONFIG$workspace_root)) {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[1]) else ""
  script_dir <- if (nzchar(script_path)) normalizePath(dirname(script_path)) else getwd()
  workspace_root <- normalizePath(file.path(script_dir, "..", ".."))
} else {
  workspace_root <- normalizePath(USER_CONFIG$workspace_root)
  script_dir <- getwd()
}

mcmc_cpp_path <- file.path(script_dir, "00_MCMC_edit.cpp")
if (!file.exists(mcmc_cpp_path)) {
  mcmc_cpp_path <- "00_MCMC_edit.cpp"
}
stopifnot(file.exists(mcmc_cpp_path))
sourceCpp(mcmc_cpp_path)

has_pROC <- requireNamespace("pROC", quietly = TRUE)

D <- USER_CONFIG$D

config <- list(
  model_name    = USER_CONFIG$model_name,
  D             = D,
  PRE           = USER_CONFIG$PRE,
  problem_names = USER_CONFIG$problem_names,
  n_prob        = USER_CONFIG$n_prob,
  N_j_vec       = USER_CONFIG$N_j_vec,

  lstm_data_dir = file.path(workspace_root,
                            sprintf("Python_file/97_Result/%s", USER_CONFIG$model_name)),
  all_actions_path = file.path(workspace_root,
                               sprintf("Python_file/97_Result/%s_D%d_all_actions.csv",
                                       USER_CONFIG$model_name, D)),
  significant_actions_path = file.path(workspace_root,
                                       sprintf("Python_file/97_Result/%s_D%d_significant_actions.csv",
                                               USER_CONFIG$model_name, D)),
  actual_result_path = file.path(workspace_root,
                                 sprintf("R_file/05_final_result/%s_%d_%s.RData",
                                         USER_CONFIG$model_name, D, USER_CONFIG$PRE)),

  N_pool = NA_integer_,
  n_resp = NA_integer_,

  tau2   = USER_CONFIG$tau2,
  nu2    = USER_CONFIG$nu2,
  N_iter = USER_CONFIG$N_iter,
  burn_in = as.integer(USER_CONFIG$N_iter * USER_CONFIG$burn_in_ratio),
  thin   = USER_CONFIG$thin,
  proposal_sd_alpha = USER_CONFIG$proposal_sd_alpha,
  proposal_sd_beta  = USER_CONFIG$proposal_sd_beta,
  proposal_sd_w     = USER_CONFIG$proposal_sd_w
)

# Seed list: an explicit override takes precedence for exact reproduction
# of a previous run; otherwise seeds are drawn (optionally under master_seed).
if (!is.null(USER_CONFIG$sim_seeds_override)) {
  sim_seeds <- as.integer(USER_CONFIG$sim_seeds_override)
  stopifnot(length(sim_seeds) > 0, !anyNA(sim_seeds), !anyDuplicated(sim_seeds))
  cat("  Using user-supplied seed list (sim_seeds_override).\n")
} else {
  if (!is.null(USER_CONFIG$master_seed)) {
    set.seed(USER_CONFIG$master_seed)
  }
  sim_seeds <- sample.int(1e7, size = USER_CONFIG$n_seeds)
}
cat(sprintf("  Simulation seeds: %s\n", paste(sim_seeds, collapse = ", ")))
cat(sprintf("  D = %d, n_seeds = %d, N_iter = %d\n\n",
            D, length(sim_seeds), config$N_iter))

output_dir <- file.path(script_dir, "simulation_outputs",
                        sprintf("D%d_%s", D, USER_CONFIG$PRE))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Utility
# =============================================================================

# Rank-based (Mann-Whitney) AUC; fallback when pROC is unavailable
calc_auc <- function(truth, score) {
  ok <- !is.na(truth) & !is.na(score)
  truth <- truth[ok]; score <- score[ok]
  if (length(unique(truth)) < 2) return(NA_real_)
  n1 <- sum(truth == 1); n0 <- sum(truth == 0)
  (sum(rank(score, ties.method = "average")[truth == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# =============================================================================
# Block A: data loading and true parameter construction
# =============================================================================

cat("===========================================================\n")
cat("  Block A: Data loading and true parameter construction\n")
cat("===========================================================\n\n")

cat("  [A1] Loading data\n")

load_actual_lstm_inputs <- function(config) {
  all_actions <- read.csv(config$all_actions_path, stringsAsFactors = FALSE)
  significant_actions <- read.csv(config$significant_actions_path, stringsAsFactors = FALSE)

  all_dfs <- vector("list", config$n_prob)
  for (j in seq_len(config$n_prob)) {
    pn <- config$problem_names[j]
    df <- read.csv(file.path(config$lstm_data_dir,
                             sprintf("long_format_%s_%s_D%d.csv",
                                     config$model_name, pn, config$D)),
                   stringsAsFactors = FALSE)
    all_dfs[[j]] <- df
  }

  data_combined <- do.call(rbind, all_dfs)
  data_combined <- data_combined[complete.cases(data_combined), ]
  all_seq_ids <- unique(as.character(data_combined$seq_id))
  sel <- if (is.na(config$n_resp)) all_seq_ids else head(all_seq_ids, config$n_resp)

  C_col_names <- paste0("C_value", 1:config$D)
  long_data <- vector("list", config$n_prob)

  for (j in seq_len(config$n_prob)) {
    pn <- config$problem_names[j]
    df <- data_combined[data_combined$problem_num == pn, ]
    df$seq_id <- as.character(df$seq_id)
    df$behavior_id <- as.integer(df$behavior_id)
    df$outcome <- as.numeric(df$outcome)

    # Robust scaling within item, independently per latent dimension
    for (col_name in C_col_names) {
      df[[col_name]] <- as.numeric(df[[col_name]])
      vals <- df[[col_name]]
      med <- median(vals, na.rm = TRUE)
      iqr_val <- IQR(vals, na.rm = TRUE)
      df[[col_name]] <- if (iqr_val > 0) (vals - med) / iqr_val else rep(0, length(vals))
    }
    long_data[[j]] <- df
  }
  rm(data_combined, all_dfs)
  n_resp <- length(sel)

  R_obs <- matrix(0L, nrow = n_resp, ncol = config$n_prob)
  C_bar <- R_counts <- vector("list", config$n_prob)
  actual_outcome <- matrix(NA_real_, nrow = n_resp, ncol = config$n_prob)

  for (j in seq_len(config$n_prob)) {
    N_j <- config$N_j_vec[j]
    df_j <- long_data[[j]][long_data[[j]]$seq_id %in% sel, ]
    C_bar[[j]] <- vector("list", n_resp)
    R_counts[[j]] <- vector("list", n_resp)
    sp <- split(df_j, df_j$seq_id)

    for (i in seq_len(n_resp)) {
      cb <- matrix(0, N_j, config$D)
      cnt <- rep(0L, N_j)
      df_ij <- sp[[sel[i]]]

      if (is.null(df_ij) || nrow(df_ij) == 0) {
        C_bar[[j]][[i]] <- cb
        R_counts[[j]][[i]] <- cnt
        next
      }
      R_obs[i, j] <- 1L
      aseq <- as.integer(df_ij$behavior_id)
      cnt <- tabulate(aseq, nbins = N_j)

      for (d in 1:config$D) {
        col_name <- C_col_names[d]
        cm <- tapply(df_ij[[col_name]], df_ij$behavior_id, mean)
        cb[as.integer(names(cm)), d] <- as.numeric(cm)
      }

      yv <- unique(df_ij$outcome[!is.na(df_ij$outcome)])
      if (length(yv) > 0) actual_outcome[i, j] <- yv[1]
      C_bar[[j]][[i]] <- cb
      R_counts[[j]][[i]] <- cnt
    }
  }

  true_delta <- vector("list", config$n_prob)
  for (j in seq_len(config$n_prob)) {
    N_j <- config$N_j_vec[j]
    dj <- rep(0L, N_j)
    si <- significant_actions$action_idx[
      significant_actions$problem_name == config$problem_names[j]]
    dj[si] <- 1L
    true_delta[[j]] <- dj
  }

  list(n_resp = n_resp, respondent_ids = sel, R_obs = R_obs,
       C_bar = C_bar, R_counts = R_counts,
       actual_outcome = actual_outcome, true_delta = true_delta)
}

inp <- load_actual_lstm_inputs(config)
config$n_resp <- inp$n_resp
config$N_pool <- inp$n_resp
R_obs <- inp$R_obs
C_bar <- inp$C_bar
R_counts <- inp$R_counts
actual_outcome <- inp$actual_outcome
true_delta <- inp$true_delta

cat(sprintf("    Respondents: %d, Items: %d, D: %d\n\n", config$n_resp, config$n_prob, config$D))

cat("  [A2] Constructing true parameters (slab-conditional posterior means)\n")

load(config$actual_result_path)  # loads `result`
stopifnot(ncol(result$alpha) == config$n_resp)
stopifnot(ncol(result$beta) == config$n_prob)

true_alpha <- colMeans(result$alpha)
true_beta  <- colMeans(result$beta)
cat(sprintf("    Alpha: [%.3f, %.3f], sd=%.3f\n",
            min(true_alpha), max(true_alpha), sd(true_alpha)))
cat(sprintf("    Beta:  [%.3f, %.3f]\n",
            min(true_beta), max(true_beta)))
cat(sprintf("    Posterior mean of sigma_alpha: %.4f\n", mean(result$sigma[, 1])))

W_slab_cond <- numeric(ncol(result$W))
PIP_actual <- colMeans(result$lambda)

for (k in 1:ncol(result$W)) {
  si <- which(result$lambda[, k] == 1)
  W_slab_cond[k] <- if (length(si) > 0) mean(result$W[si, k]) else 0
}

# Spike-component weights are drawn once with a fixed seed so that
# true parameters are identical across all simulation seeds.
set.seed(99999)
true_lambda <- true_omega <- vector("list", config$n_prob)
w_offset <- 0

for (j in 1:config$n_prob) {
  N_j <- config$N_j_vec[j]
  dj <- true_delta[[j]]
  lf <- of <- numeric(N_j * config$D)

  for (l in 1:N_j) {
    for (d in 1:config$D) {
      fi <- (l - 1) * config$D + d
      gi <- w_offset + fi
      if (dj[l] == 1) {
        of[fi] <- W_slab_cond[gi]
        lf[fi] <- 1
      } else {
        of[fi] <- rnorm(1, 0, sqrt(config$tau2))
        lf[fi] <- 0
      }
    }
  }
  true_lambda[[j]] <- lf
  true_omega[[j]] <- of
  w_offset <- w_offset + N_j * config$D
}

for (j in 1:config$n_prob) {
  so <- true_omega[[j]][true_lambda[[j]] == 1]
  cat(sprintf("    Item %2d: K_j=%2d, |omega| range=[%.4f, %.4f]\n",
              j, sum(true_delta[[j]]),
              ifelse(length(so) > 0, min(abs(so)), NA),
              ifelse(length(so) > 0, max(abs(so)), NA)))
}

# Flattened covariate matrices shared across all seeds
# (action-major ordering: column index = (l - 1) * D + d)
C_sum_list <- lapply(1:config$n_prob, function(j) {
  do.call(rbind, lapply(1:config$n_resp, function(i) {
    as.vector(t(C_bar[[j]][[i]]))
  }))
})

rm(result); gc()
cat("\n")

fixed <- list(
  config         = config,
  R_obs          = R_obs,
  C_bar          = C_bar,
  R_counts       = R_counts,
  actual_outcome = actual_outcome,
  true_delta     = true_delta,
  true_alpha     = true_alpha,
  true_beta      = true_beta,
  true_omega     = true_omega,
  true_lambda    = true_lambda,
  C_sum_list     = C_sum_list
)

# =============================================================================
# Block B: single-seed simulation
# =============================================================================

run_single_seed <- function(seed, fixed) {

  set.seed(seed)
  cfg <- fixed$config
  D   <- cfg$D

  cat(sprintf("\n--- Seed %d ---\n", seed))
  t_start <- Sys.time()

  # B1. Generate bootstrap responses
  data_mat <- matrix(NA, cfg$n_resp, cfg$n_prob)

  for (j in 1:cfg$n_prob) {
    N_j <- cfg$N_j_vec[j]
    for (i in 1:cfg$n_resp) {
      if (fixed$R_obs[i, j] == 0L) next
      at <- 0
      for (l in 1:N_j) {
        if (fixed$R_counts[[j]][[i]][l] > 0) {
          for (d in 1:D) {
            fi <- (l - 1) * D + d
            at <- at + fixed$true_omega[[j]][fi] * fixed$C_bar[[j]][[i]][l, d]
          }
        }
      }
      eta <- fixed$true_alpha[i] + fixed$true_beta[j] + at
      data_mat[i, j] <- rbinom(1, 1, 1 / (1 + exp(-eta)))
    }
  }

  sim_rate <- mean(data_mat, na.rm = TRUE)
  cat(sprintf("  Responses generated: correct rate = %.1f%%\n", sim_rate * 100))

  # B2. Run MCMC
  mcmc_result <- MCMC_action_model_v5(
    N_iter     = cfg$N_iter,
    data       = data_mat,
    C_sum_list = fixed$C_sum_list,
    N_j_vec    = cfg$N_j_vec,
    D          = D,
    alpha_init = rnorm(cfg$n_resp, 0, 0.1),
    beta_init  = rnorm(cfg$n_prob, 0, 0.1),
    W_init     = lapply(cfg$N_j_vec, function(n) rnorm(n * D, 0, 0.1)),
    lambda_init = lapply(cfg$N_j_vec, function(n) rep(0.5, n * D)),
    sigma_alpha_init = var(fixed$true_alpha),
    tau2 = cfg$tau2, nu2 = cfg$nu2,
    proposal_sd_alpha = cfg$proposal_sd_alpha,
    proposal_sd_beta  = cfg$proposal_sd_beta,
    proposal_sd_w     = cfg$proposal_sd_w,
    burn_in = cfg$burn_in, thin = cfg$thin
  )

  cat(sprintf("  MCMC finished: saved samples = %d\n", mcmc_result$n_save))

  # B3. Evaluation (AUC only)

  # Posterior inclusion probabilities
  PIP_flat <- colMeans(mcmc_result$lambda)
  PIP_by_prob <- vector("list", cfg$n_prob)
  wo <- 0
  for (j in 1:cfg$n_prob) {
    np <- cfg$N_j_vec[j] * D
    PIP_by_prob[[j]] <- PIP_flat[(wo + 1):(wo + np)]
    wo <- wo + np
  }

  # AUC at the (l, d) level
  all_tl  <- unlist(fixed$true_lambda)
  all_pip <- PIP_flat

  auc_ld <- if (length(unique(all_tl)) > 1) {
    if (has_pROC) as.numeric(pROC::auc(pROC::roc(all_tl, all_pip, quiet = TRUE)))
    else calc_auc(all_tl, all_pip)
  } else NA_real_

  # AUC at the action level (max PIP across dimensions)
  ta <- sum(cfg$N_j_vec)
  all_td <- integer(ta); all_ap <- numeric(ta); idx <- 0L
  for (j in 1:cfg$n_prob) {
    for (l in 1:cfg$N_j_vec[j]) {
      idx <- idx + 1L
      di <- ((l - 1) * D + 1):(l * D)
      all_td[idx] <- fixed$true_delta[[j]][l]
      all_ap[idx] <- max(PIP_by_prob[[j]][di])
    }
  }

  auc_act <- if (length(unique(all_td)) > 1) {
    if (has_pROC) as.numeric(pROC::auc(pROC::roc(all_td, all_ap, quiet = TRUE)))
    else calc_auc(all_td, all_ap)
  } else NA_real_

  t_end <- Sys.time()
  elapsed_min <- as.numeric(difftime(t_end, t_start, units = "mins"))
  cat(sprintf("  Elapsed: %.1f min\n", elapsed_min))

  metrics <- data.frame(
    seed             = seed,
    D                = D,
    sim_correct_rate = sim_rate,
    auc_ld           = auc_ld,
    auc_act          = auc_act,
    elapsed_min      = elapsed_min,
    stringsAsFactors = FALSE
  )

  seed_file <- file.path(output_dir, sprintf("seed_%d.rds", seed))
  saveRDS(list(
    seed        = seed,
    metrics     = metrics,
    mcmc_result = mcmc_result,
    data_mat    = data_mat,
    PIP_flat    = PIP_flat,
    PIP_by_prob = PIP_by_prob
  ), file = seed_file)
  cat(sprintf("  Saved: %s\n", seed_file))

  return(metrics)
}

# =============================================================================
# Block B execution
# =============================================================================

cat("===========================================================\n")
cat("  Block B: Per-seed simulation\n")
cat("===========================================================\n")

all_metrics <- vector("list", length(sim_seeds))

for (s in seq_along(sim_seeds)) {
  cat(sprintf("\n[%d / %d]\n", s, length(sim_seeds)))
  all_metrics[[s]] <- run_single_seed(sim_seeds[s], fixed)
}

metrics_df <- do.call(rbind, all_metrics)

# =============================================================================
# Block C: aggregation
# =============================================================================

cat("\n\n===========================================================\n")
cat("  Block C: Aggregation\n")
cat("===========================================================\n\n")

numeric_cols <- c(
  "sim_correct_rate",
  "auc_ld", "auc_act",
  "elapsed_min"
)

summarize_col <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(mean = NA, sd = NA, median = NA, min = NA, max = NA))
  c(mean   = mean(x),
    sd     = sd(x),
    median = median(x),
    min    = min(x),
    max    = max(x))
}

summary_list <- lapply(numeric_cols, function(col) {
  summarize_col(metrics_df[[col]])
})
names(summary_list) <- numeric_cols

summary_df <- as.data.frame(do.call(rbind, summary_list))
summary_df$metric <- rownames(summary_df)
summary_df <- summary_df[, c("metric", "mean", "sd", "median", "min", "max")]
rownames(summary_df) <- NULL

cat("  [Variable selection (AUC)]\n")
for (m in c("auc_ld", "auc_act")) {
  r <- summary_list[[m]]
  cat(sprintf("    %-12s: %.4f \u00b1 %.4f  [%.4f, %.4f]\n",
              m, r["mean"], r["sd"], r["min"], r["max"]))
}

cat("\n  [Simulated correct rate]\n")
r <- summary_list[["sim_correct_rate"]]
cat(sprintf("    %-12s: %.4f \u00b1 %.4f  [%.4f, %.4f]\n",
            "sim_rate", r["mean"], r["sd"], r["min"], r["max"]))

cat("\n  [Saving]\n")

csv_path <- file.path(output_dir, "all_seeds_metrics.csv")
write.csv(metrics_df, file = csv_path, row.names = FALSE)
cat(sprintf("    Per-seed metrics: %s\n", csv_path))

summary_csv_path <- file.path(output_dir, "summary_statistics.csv")
write.csv(summary_df, file = summary_csv_path, row.names = FALSE)
cat(sprintf("    Summary: %s\n", summary_csv_path))

full_rds_path <- file.path(output_dir, "multi_seed_results.rds")
saveRDS(list(
  sim_seeds   = sim_seeds,
  D           = D,
  config      = config,
  metrics_df  = metrics_df,
  summary_df  = summary_df,
  true_params = list(
    alpha  = true_alpha,
    beta   = true_beta,
    delta  = true_delta,
    lambda = true_lambda,
    omega  = true_omega
  )
), file = full_rds_path)
cat(sprintf("    Full results: %s\n", full_rds_path))

seed_log_path <- file.path(output_dir, "seed_log.txt")
writeLines(c(
  sprintf("Run time: %s", Sys.time()),
  sprintf("D: %d", D),
  sprintf("n_seeds: %d", length(sim_seeds)),
  sprintf("N_iter: %d, burn_in: %d, thin: %d", config$N_iter, config$burn_in, config$thin),
  sprintf("seed_source: %s",
          if (!is.null(USER_CONFIG$sim_seeds_override)) "sim_seeds_override"
          else if (!is.null(USER_CONFIG$master_seed)) sprintf("master_seed = %d", USER_CONFIG$master_seed)
          else "random"),
  sprintf("sim_seeds: %s", paste(sim_seeds, collapse = ", "))
), con = seed_log_path)
cat(sprintf("    Seed log: %s\n", seed_log_path))

cat("\n=== Done ===\n")
