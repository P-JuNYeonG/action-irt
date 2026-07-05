# =============================================================================
# Convergence diagnostics: Gelman-Rubin (R-hat) and PIP stability
# Model: MCMC_action_model_v5 (fixed beta prior, multi-dimensional D)
# Burn-in and thinning are applied inside the C++ sampler.
# =============================================================================

library(Rcpp)
library(dplyr)
library(coda)

# ---- Configuration ----------------------------------------------------------
model_name <- "lstm_ae"   # lstm_ae, mlp_ae, pca
D          <- 1           # latent dimension
n_chains   <- 5
iteration  <- 50000
burn_in    <- iteration * 0.2
thin       <- 10

cpp_path  <- "00_MCMC_edit.cpp"
base_path <- file.path("data", model_name)  # directory with long_format_*.csv

sourceCpp(cpp_path)

# ---- 1. Load data and robust-scale latent values within each item -----------
files <- list.files(
  path = base_path,
  pattern = sprintf("^long_format_%s_ps[0-9]_[0-9]+_D%d\\.csv$", model_name, D),
  full.names = TRUE
)
data_raw <- do.call(rbind, lapply(files, read.csv))
data     <- data_raw[complete.cases(data_raw), ]

C_col_names <- paste0("C_value", 1:D)

for (col_name in C_col_names) {
  data <- data %>%
    group_by(problem_num) %>%
    mutate(
      !!col_name := {
        vals    <- .data[[col_name]]
        med     <- median(vals, na.rm = TRUE)
        iqr_val <- IQR(vals, na.rm = TRUE)
        if (iqr_val > 0) (vals - med) / iqr_val else rep(0, length(vals))
      }
    ) %>%
    ungroup()
}

# ---- 2. Build response matrix, N_j_vec, and flattened C matrices ------------
unique_students <- unique(data$seq_id)
unique_problems <- unique(data$problem_num)
n_students      <- length(unique_students)
n_problems      <- length(unique_problems)

response_matrix <- matrix(NA, nrow = n_students, ncol = n_problems)
rownames(response_matrix) <- unique_students
colnames(response_matrix) <- unique_problems

response_data <- data %>%
  group_by(seq_id, problem_num) %>%
  summarise(outcome = first(outcome), .groups = "drop")

for (i in 1:nrow(response_data)) {
  s_idx <- which(unique_students == response_data$seq_id[i])
  p_idx <- which(unique_problems == response_data$problem_num[i])
  response_matrix[s_idx, p_idx] <- response_data$outcome[i]
}

# Assumes behavior_id runs 1..N_j within each item
N_j_vec <- sapply(unique_problems, function(p) {
  max(data$behavior_id[data$problem_num == p])
})
names(N_j_vec) <- unique_problems

C_list <- vector("list", n_problems)
names(C_list) <- unique_problems

for (p in 1:n_problems) {
  prob_name    <- unique_problems[p]
  N_j          <- N_j_vec[p]
  problem_data <- data %>% filter(problem_num == prob_name)

  # Flattened layout: column (l-1)*D + d matches the C++ indexing
  C_matrix_flat <- matrix(0, nrow = n_students, ncol = N_j * D)
  rownames(C_matrix_flat) <- unique_students

  for (d in 1:D) {
    aggregated <- problem_data %>%
      group_by(seq_id, behavior_id) %>%
      summarise(C_agg = mean(.data[[C_col_names[d]]], na.rm = TRUE),
                .groups = "drop")

    for (i in 1:nrow(aggregated)) {
      s_idx      <- which(unique_students == aggregated$seq_id[i])
      action_idx <- aggregated$behavior_id[i]
      if (length(s_idx) > 0 && action_idx <= N_j) {
        C_matrix_flat[s_idx, (action_idx - 1) * D + d] <- aggregated$C_agg[i]
      }
    }
  }
  C_list[[p]] <- C_matrix_flat
}

cat(sprintf("Respondents: %d, Items: %d, D: %d\n", n_students, n_problems, D))
cat(sprintf("Total W parameters: %d\n", sum(N_j_vec) * D))

# ---- 3. Run multiple chains with overdispersed initializations --------------
run_multiple_chains <- function(n_chains, n_iter, data, C_sum_list,
                                N_j_vec, D, burn_in, thin) {
  chains <- vector("list", n_chains)
  n_resp <- nrow(data)
  n_prob <- ncol(data)

  for (chain_id in 1:n_chains) {
    cat(sprintf("Running chain %d / %d...\n", chain_id, n_chains))
    set.seed(1000 * chain_id)

    chains[[chain_id]] <- MCMC_action_model_v5(
      N_iter            = n_iter,
      data              = data,
      C_sum_list        = C_sum_list,
      N_j_vec           = N_j_vec,
      D                 = D,
      alpha_init        = rnorm(n_resp, 0, 2),
      beta_init         = rnorm(n_prob, 0, 2),
      W_init            = lapply(N_j_vec, function(n) rnorm(n * D, 0, 2)),
      lambda_init       = lapply(N_j_vec, function(n) rbinom(n * D, 1, 0.5)),
      sigma_alpha_init  = runif(1, 0.5, 5),
      tau2              = 0.001,
      nu2               = 2.5,
      proposal_sd_alpha = 1.5,
      proposal_sd_beta  = 0.5,
      proposal_sd_w     = 0.4,
      burn_in           = burn_in,
      thin              = thin
    )
  }
  chains
}

start_time <- Sys.time()
chains <- run_multiple_chains(n_chains, iteration, response_matrix, C_list,
                              N_j_vec, D, burn_in, thin)
cat(sprintf("Total elapsed: %.2f min\n",
            as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# ---- 4. Gelman-Rubin R-hat (coda::gelman.diag) -------------------------------
# autoburnin = FALSE: burn-in already removed inside the C++ sampler
compute_rhat_vector <- function(chains, param_name) {
  n_param <- ncol(chains[[1]][[param_name]])
  rhat    <- numeric(n_param)
  for (k in 1:n_param) {
    mcmc_list <- mcmc.list(lapply(chains, function(ch) {
      mcmc(ch[[param_name]][, k])
    }))
    rhat[k] <- gelman.diag(mcmc_list, autoburnin = FALSE,
                           multivariate = FALSE)$psrf[1, 1]
  }
  rhat
}

compute_rhat_scalar <- function(chains, param_name, col_idx) {
  mcmc_list <- mcmc.list(lapply(chains, function(ch) {
    mcmc(ch[[param_name]][, col_idx])
  }))
  gelman.diag(mcmc_list, autoburnin = FALSE,
              multivariate = FALSE)$psrf[1, 1]
}

# ---- 5. PIP stability: between-chain SD of chain-specific PIPs ---------------
compute_pip_stability <- function(chains) {
  # pip_mat: (n_chains x n_W), chain-specific posterior inclusion probabilities
  pip_mat <- do.call(rbind, lapply(chains, function(ch) {
    colMeans(ch[["lambda"]])
  }))

  mean_pip <- colMeans(pip_mat)
  sd_pip   <- apply(pip_mat, 2, sd)

  list(
    mean_pip         = mean_pip,
    sd_pip           = sd_pip,
    mean_sd          = mean(sd_pip),
    median_sd        = median(sd_pip),
    max_sd           = max(sd_pip),
    n_selected       = sum(mean_pip >= 0.5),
    mean_sd_selected = mean(sd_pip[mean_pip >= 0.5]),
    max_sd_selected  = max(sd_pip[mean_pip >= 0.5])
  )
}

# ---- 6. Compute diagnostics ---------------------------------------------------
rhat_alpha       <- compute_rhat_vector(chains, "alpha")
rhat_beta        <- compute_rhat_vector(chains, "beta")
rhat_W_all       <- compute_rhat_vector(chains, "W")
rhat_sigma_alpha <- compute_rhat_scalar(chains, "sigma", 1)  # col 1 = sigma_alpha^2

# Selected subset: mean PIP across chains >= 0.5 (slab component)
pip_mean_for_select <- colMeans(do.call(rbind, lapply(chains, function(ch) {
  colMeans(ch[["lambda"]])
})))
rhat_W_selected <- rhat_W_all[pip_mean_for_select >= 0.5]

pip_stability <- compute_pip_stability(chains)

# ---- 7. Report: R-hat by parameter block --------------------------------------
print_rhat_row <- function(name, rhat) {
  cat(sprintf("%-45s  median = %.4f   max = %.4f\n",
              name, median(rhat), max(rhat)))
}

cat("\n===== Gelman-Rubin R-hat by parameter block =====\n")
cat(sprintf("Model: %s, D = %d, Chains: %d\n", model_name, D, n_chains))
print_rhat_row("Respondent abilities alpha_i",              rhat_alpha)
print_rhat_row("Item difficulties beta_j",                  rhat_beta)
print_rhat_row("Ability variance sigma_alpha^2",            rhat_sigma_alpha)
print_rhat_row("Action weights omega (selected, PIP>=0.5)", rhat_W_selected)
print_rhat_row("Action weights omega (all)",                rhat_W_all)

all_rhat <- c(rhat_alpha, rhat_beta, rhat_W_all, rhat_sigma_alpha)
cat(sprintf("\nOverall max R-hat: %.4f  (criterion: R-hat < 1.10)\n",
            max(all_rhat)))

# ---- 8. Report: PIP stability --------------------------------------------------
cat("\n===== PIP stability (between-chain SD of PIP) =====\n")
cat(sprintf("Number of W parameters      : %d\n",   length(pip_stability$sd_pip)))
cat(sprintf("Mean SD                     : %.4f\n", pip_stability$mean_sd))
cat(sprintf("Median SD                   : %.4f\n", pip_stability$median_sd))
cat(sprintf("Max SD                      : %.4f\n", pip_stability$max_sd))
cat(sprintf("Selected actions (PIP>=0.5) : %d\n",   pip_stability$n_selected))
cat(sprintf("Mean SD (selected)          : %.4f\n", pip_stability$mean_sd_selected))
cat(sprintf("Max SD (selected)           : %.4f\n", pip_stability$max_sd_selected))