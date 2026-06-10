# =============================================================================
# Multi-Seed Parametric Bootstrap 시뮬레이션
# =============================================================================
# 구조:
#   Block A — 데이터 로드 + True 파라미터 구성 (1회, seed 독립)
#   Block B — seed별 Y 생성 → MCMC → 평가 → 진단 (반복)
#   Block C — 전체 seed 결과 결합 및 집계
# =============================================================================

library(Rcpp)
library(RcppArmadillo)
library(coda)

# =============================================================================
# 사용자 설정 영역
# =============================================================================

USER_CONFIG <- list(
  # --- 모델/차원 설정 (D를 바꾸면 자동 대응) ---
  model_name = "lstm_ae",
  D          = 1,
  PRE        = "robust",
  

  # --- MCMC 설정 ---
  N_iter           = 50000,
  burn_in_ratio    = 0.2,
  thin             = 10,
  proposal_sd_alpha = 1.5,
  proposal_sd_beta  = 0.5,
  proposal_sd_w     = 0.4,
  tau2             = 0.001,
  nu2              = 2.5,
  
  # --- 시뮬레이션 설정 ---
  n_seeds          = 5,        # 반복 시뮬레이션 횟수
  master_seed      = NULL,     # NULL이면 완전 랜덤, 숫자면 재현 가능
  
  # --- 문제별 행동 수 (D에 무관, 고정) ---
  problem_names = c(paste0("ps1_", 1:7), paste0("ps2_", 1:7)),
  n_prob        = 14,
  N_j_vec       = c(119, 141, 124, 47, 48, 182, 200,
                     156, 289,  70, 206, 112, 146, 185)
)

# =============================================================================
# Step 0. 환경 설정
# =============================================================================

cat("=============================================================\n")
cat("  Multi-Seed Parametric Bootstrap Simulation\n")
cat("=============================================================\n\n")

# --- workspace_root 결정 ---
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

# --- C++ 컴파일 ---
mcmc_cpp_path <- file.path(script_dir, "00_MCMC_edit.cpp")
if (!file.exists(mcmc_cpp_path)) {
  # 현재 디렉토리에서도 탐색
  mcmc_cpp_path <- "00_MCMC_edit.cpp"
}
stopifnot(file.exists(mcmc_cpp_path))
sourceCpp(mcmc_cpp_path)

# --- pROC 확인 ---
has_pROC <- requireNamespace("pROC", quietly = TRUE)

# --- config 구성 ---
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
                                 sprintf("R_file/95_multi_result/%s_%d_%s.RData",
                                         USER_CONFIG$model_name, D, USER_CONFIG$PRE)),
  
  N_pool = NA_integer_,
  n_resp = NA_integer_,
  
  tau2  = USER_CONFIG$tau2,
  nu2   = USER_CONFIG$nu2,
  N_iter = USER_CONFIG$N_iter,
  burn_in = as.integer(USER_CONFIG$N_iter * USER_CONFIG$burn_in_ratio),
  thin   = USER_CONFIG$thin,
  proposal_sd_alpha = USER_CONFIG$proposal_sd_alpha,
  proposal_sd_beta  = USER_CONFIG$proposal_sd_beta,
  proposal_sd_w     = USER_CONFIG$proposal_sd_w
)

# --- seed 생성 ---
if (!is.null(USER_CONFIG$master_seed)) {
  set.seed(USER_CONFIG$master_seed)
}
sim_seeds <- sample.int(1e7, size = USER_CONFIG$n_seeds)
cat(sprintf("  시뮬레이션 seed 목록: %s\n", paste(sim_seeds, collapse = ", ")))
cat(sprintf("  D = %d, n_seeds = %d, N_iter = %d\n\n",
            D, USER_CONFIG$n_seeds, config$N_iter))

# --- 출력 디렉토리 ---
output_dir <- file.path(script_dir, "simulation_outputs",
                        sprintf("D%d_%s", D, USER_CONFIG$PRE))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# 유틸리티 함수
# =============================================================================

calc_auc <- function(truth, score) {
  ok <- !is.na(truth) & !is.na(score)
  truth <- truth[ok]; score <- score[ok]
  if (length(unique(truth)) < 2) return(NA_real_)
  n1 <- sum(truth == 1); n0 <- sum(truth == 0)
  (sum(rank(score, ties.method = "average")[truth == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

calc_sensitivity <- function(truth, score, thr = 0.5) {
  pred <- as.numeric(score > thr)
  tp <- sum(pred == 1 & truth == 1)
  fn <- sum(pred == 0 & truth == 1)
  ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
}

calc_sensitivity_bin <- function(truth, pred) {
  tp <- sum(pred == 1 & truth == 1)
  fn <- sum(pred == 0 & truth == 1)
  ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
}

calc_hpd <- function(x, prob = 0.95) {
  h <- coda::HPDinterval(coda::as.mcmc(x), prob = prob)
  c(lower = unname(h[1, "lower"]), upper = unname(h[1, "upper"]))
}

# =============================================================================
# Block A: 데이터 로드 + True 파라미터 구성 (1회)
# =============================================================================

cat("===========================================================\n")
cat("  Block A: 데이터 로드 + True 파라미터 구성\n")
cat("===========================================================\n\n")

# --- A1. 데이터 로드 ---
cat("  [A1] 데이터 로드\n")

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
    
    # robust scaling (차원별 독립)
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

cat(sprintf("    응답자: %d, 문제: %d, D: %d\n\n", config$n_resp, config$n_prob, config$D))

# --- A2. True 파라미터 구성 ---
cat("  [A2] True 파라미터 구성 (slab 조건부 사후 평균)\n")

load(config$actual_result_path)  # result 로드
stopifnot(ncol(result$alpha) == config$n_resp)
stopifnot(ncol(result$beta) == config$n_prob)

true_alpha <- colMeans(result$alpha)
true_beta  <- colMeans(result$beta)
cat(sprintf("    Alpha: [%.3f, %.3f], sd=%.3f\n",
            min(true_alpha), max(true_alpha), sd(true_alpha)))
cat(sprintf("    Beta:  [%.3f, %.3f]\n",
            min(true_beta), max(true_beta)))
cat(sprintf("    sigma_alpha 사후 평균: %.4f\n", mean(result$sigma[, 1])))

# Omega: slab 조건부 사후 평균
W_slab_cond <- numeric(ncol(result$W))
PIP_actual <- colMeans(result$lambda)

for (k in 1:ncol(result$W)) {
  si <- which(result$lambda[, k] == 1)
  W_slab_cond[k] <- if (length(si) > 0) mean(result$W[si, k]) else 0
}

# true_omega, true_lambda 구성
# spike 성분은 고정 seed로 1회만 생성 (모든 sim seed에 대해 동일)
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

# K_j 확인 출력
for (j in 1:config$n_prob) {
  so <- true_omega[[j]][true_lambda[[j]] == 1]
  cat(sprintf("    문제 %2d: K_j=%2d, |omega| 범위=[%.4f, %.4f]\n",
              j, sum(true_delta[[j]]),
              ifelse(length(so) > 0, min(abs(so)), NA),
              ifelse(length(so) > 0, max(abs(so)), NA)))
}

# C_sum_list 사전 구성 (모든 seed에 공유)
C_sum_list <- lapply(1:config$n_prob, function(j) {
  do.call(rbind, lapply(1:config$n_resp, function(i) {
    as.vector(t(C_bar[[j]][[i]]))
  }))
})

rm(result); gc()
cat("\n")

# --- 고정 객체 묶음 ---
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
# Block B: 단일 seed 시뮬레이션 함수
# =============================================================================

run_single_seed <- function(seed, fixed) {
  
  set.seed(seed)
  cfg <- fixed$config
  D   <- cfg$D
  
  cat(sprintf("\n--- Seed %d 시작 ---\n", seed))
  t_start <- Sys.time()
  
  # =========================================================================
  # B1. Y 생성
  # =========================================================================
  data_mat <- eta_ab <- eta_act <- matrix(NA, cfg$n_resp, cfg$n_prob)
  
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
      eta_ab[i, j]  <- fixed$true_alpha[i] + fixed$true_beta[j]
      eta_act[i, j] <- at
      eta <- eta_ab[i, j] + at
      data_mat[i, j] <- rbinom(1, 1, 1 / (1 + exp(-eta)))
    }
  }
  
  sim_rate <- mean(data_mat, na.rm = TRUE)
  cat(sprintf("  Y 생성 완료: 정답률=%.1f%%\n", sim_rate * 100))
  
  # =========================================================================
  # B2. MCMC 실행
  # =========================================================================
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
  
  cat(sprintf("  MCMC 완료: 저장 샘플=%d\n", mcmc_result$n_save))
  
  # =========================================================================
  # B3. 평가
  # =========================================================================
  
  # --- 수락률 ---
  acc_alpha <- mean(mcmc_result$accept_alpha)
  acc_beta  <- mean(mcmc_result$accept_beta)
  acc_w     <- mean(mcmc_result$accept_w)
  
  # --- PIP ---
  PIP_flat <- colMeans(mcmc_result$lambda)
  PIP_by_prob <- vector("list", cfg$n_prob)
  wo <- 0
  for (j in 1:cfg$n_prob) {
    np <- cfg$N_j_vec[j] * D
    PIP_by_prob[[j]] <- PIP_flat[(wo + 1):(wo + np)]
    wo <- wo + np
  }
  
  # --- (l,d) 수준: AUC, Sensitivity ---
  all_tl  <- unlist(fixed$true_lambda)
  all_pip <- PIP_flat
  
  auc_ld <- if (length(unique(all_tl)) > 1) {
    if (has_pROC) as.numeric(pROC::auc(pROC::roc(all_tl, all_pip, quiet = TRUE)))
    else calc_auc(all_tl, all_pip)
  } else NA_real_
  
  sens_ld <- calc_sensitivity(all_tl, all_pip)
  
  # --- 행동 수준: AUC, Sensitivity ---
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
  
  auc_act  <- if (length(unique(all_td)) > 1) {
    if (has_pROC) as.numeric(pROC::auc(pROC::roc(all_td, all_ap, quiet = TRUE)))
    else calc_auc(all_td, all_ap)
  } else NA_real_
  
  sens_act <- calc_sensitivity(all_td, all_ap)
  
  # --- HPD 기반 선별: Sensitivity ---
  all_hpd <- integer(ta); idx <- 0L
  for (j in 1:cfg$n_prob) {
    N_j <- cfg$N_j_vec[j]
    ci <- which(data_mat[, j] == 1)
    ii <- which(data_mat[, j] == 0)
    for (l in 1:N_j) {
      idx <- idx + 1L
      di <- ((l - 1) * D + 1):(l * D)
      sd_dims <- which(PIP_by_prob[[j]][di] >= 0.5)
      ed <- rep(0, mcmc_result$n_save)
      
      if (length(ci) > 0 && length(ii) > 0 && length(sd_dims) > 0) {
        for (d in sd_dims) {
          fij <- (l - 1) * D + d
          gix <- sum(cfg$N_j_vec[seq_len(j - 1)] * D) + fij
          cc <- mean(sapply(ci, function(i) fixed$C_bar[[j]][[i]][l, d]))
          ic <- mean(sapply(ii, function(i) fixed$C_bar[[j]][[i]][l, d]))
          ed <- ed + mcmc_result$W[, gix] *
            as.numeric(mcmc_result$lambda[, gix] == 1) * (cc - ic)
        }
      }
      hpd <- calc_hpd(ed)
      all_hpd[idx] <- as.integer(hpd["lower"] > 0 || hpd["upper"] < 0)
    }
  }
  
  sens_hpd <- calc_sensitivity_bin(all_td, all_hpd)
  
  # --- 파라미터 복원: Alpha, Beta ---
  a_pm <- colMeans(mcmc_result$alpha)
  b_pm <- colMeans(mcmc_result$beta)
  
  alpha_bias <- mean(a_pm - fixed$true_alpha)
  alpha_rmse <- sqrt(mean((a_pm - fixed$true_alpha)^2))
  beta_bias  <- mean(b_pm - fixed$true_beta)
  beta_rmse  <- sqrt(mean((b_pm - fixed$true_beta)^2))
  
  # =========================================================================
  # B4. 진단
  # =========================================================================
  
  # --- Eta 분포 ---
  obs <- !is.na(data_mat)
  rel_cont <- abs(eta_act[obs]) / (abs(eta_ab[obs]) + abs(eta_act[obs]) + 1e-10)
  action_contrib_mean <- mean(rel_cont)
  
  # --- C̄ 부호 일치 ---
  sign_match <- 0; sign_total <- 0
  for (j in 1:cfg$n_prob) {
    imp <- which(fixed$true_delta[[j]] == 1)
    sc <- which(data_mat[, j] == 1); si <- which(data_mat[, j] == 0)
    ac <- which(fixed$actual_outcome[, j] == 1)
    ai <- which(fixed$actual_outcome[, j] == 0)
    for (l in imp) {
      if (length(sc) > 0 && length(si) > 0 && length(ac) > 0 && length(ai) > 0) {
        sd_ <- mean(sapply(sc, function(i) fixed$C_bar[[j]][[i]][l, 1])) -
               mean(sapply(si, function(i) fixed$C_bar[[j]][[i]][l, 1]))
        ad_ <- mean(sapply(ac, function(i) fixed$C_bar[[j]][[i]][l, 1])) -
               mean(sapply(ai, function(i) fixed$C_bar[[j]][[i]][l, 1]))
        sign_total <- sign_total + 1
        if (sign(sd_) == sign(ad_)) sign_match <- sign_match + 1
      }
    }
  }
  cbar_sign_match_rate <- ifelse(sign_total > 0, sign_match / sign_total, NA_real_)
  
  # --- Lambda mixing ---
  imp_gi <- c(); wo <- 0
  for (j in 1:cfg$n_prob) {
    for (l in 1:cfg$N_j_vec[j]) {
      if (fixed$true_delta[[j]][l] == 1) {
        for (d in 1:D) imp_gi <- c(imp_gi, wo + (l - 1) * D + d)
      }
    }
    wo <- wo + cfg$N_j_vec[j] * D
  }
  
  lam_switch <- sapply(imp_gi, function(gi) {
    lc <- mcmc_result$lambda[, gi]
    sum(diff(lc) != 0) / (length(lc) - 1)
  })
  lambda_switch_median <- median(lam_switch)
  lambda_switch_low_pct <- mean(lam_switch < 0.01)
  
  # --- Sensitivity 손실 경로 ---
  n_imp <- sum(all_td == 1)
  s1_pass <- all_ap >= 0.5
  s1_tp   <- sum(all_td == 1 & s1_pass)
  s1_miss <- sum(all_td == 1 & !s1_pass)
  s2_cand <- which(all_td == 1 & s1_pass)
  s2_tp   <- sum(all_hpd[s2_cand] == 1)
  s2_miss <- sum(all_hpd[s2_cand] == 0)
  
  bottleneck <- ifelse(s1_miss > s2_miss, "PIP", "HPD")
  
  # =========================================================================
  # 결과 반환
  # =========================================================================
  t_end <- Sys.time()
  elapsed_min <- as.numeric(difftime(t_end, t_start, units = "mins"))
  cat(sprintf("  소요 시간: %.1f분\n", elapsed_min))
  
  metrics <- data.frame(
    seed             = seed,
    D                = D,
    sim_correct_rate = sim_rate,
    
    # 변수 선별
    auc_ld           = auc_ld,
    sens_ld          = sens_ld,
    auc_act          = auc_act,
    sens_act         = sens_act,
    sens_hpd         = sens_hpd,
    
    # 파라미터 복원
    alpha_bias       = alpha_bias,
    alpha_rmse       = alpha_rmse,
    beta_bias        = beta_bias,
    beta_rmse        = beta_rmse,
    
    # 수락률
    acc_alpha        = acc_alpha,
    acc_beta         = acc_beta,
    acc_w            = acc_w,
    
    # 진단
    action_contrib   = action_contrib_mean,
    cbar_sign_match  = cbar_sign_match_rate,
    lam_switch_med   = lambda_switch_median,
    lam_switch_low   = lambda_switch_low_pct,
    s1_tp            = s1_tp,
    s1_miss          = s1_miss,
    s2_tp            = s2_tp,
    s2_miss          = s2_miss,
    bottleneck       = bottleneck,
    elapsed_min      = elapsed_min,
    
    stringsAsFactors = FALSE
  )
  
  # 개별 seed 저장
  seed_file <- file.path(output_dir, sprintf("seed_%d.rds", seed))
  saveRDS(list(
    seed       = seed,
    metrics    = metrics,
    mcmc_result = mcmc_result,
    data_mat   = data_mat,
    PIP_flat   = PIP_flat,
    PIP_by_prob = PIP_by_prob,
    all_hpd    = all_hpd
  ), file = seed_file)
  cat(sprintf("  저장: %s\n", seed_file))
  
  return(metrics)
}


# =============================================================================
# Block B 실행: 순차 반복
# =============================================================================

cat("===========================================================\n")
cat("  Block B: Seed별 시뮬레이션 실행\n")
cat("===========================================================\n")

all_metrics <- vector("list", length(sim_seeds))

for (s in seq_along(sim_seeds)) {
  cat(sprintf("\n[%d / %d]\n", s, length(sim_seeds)))
  all_metrics[[s]] <- run_single_seed(sim_seeds[s], fixed)
}

metrics_df <- do.call(rbind, all_metrics)


# =============================================================================
# Block C: 결과 집계
# =============================================================================

cat("\n\n===========================================================\n")
cat("  Block C: 결과 집계\n")
cat("===========================================================\n\n")

# 수치형 지표 열 선택 (문자열 열 제외)
numeric_cols <- c(
  "sim_correct_rate",
  "auc_ld", "sens_ld", "auc_act", "sens_act", "sens_hpd",
  "alpha_bias", "alpha_rmse", "beta_bias", "beta_rmse",
  "acc_alpha", "acc_beta", "acc_w",
  "action_contrib", "cbar_sign_match",
  "lam_switch_med", "lam_switch_low",
  "s1_tp", "s1_miss", "s2_tp", "s2_miss",
  "elapsed_min"
)

# 집계 함수
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

# --- 출력 ---
cat("  [변수 선별]\n")
for (m in c("auc_ld", "sens_ld", "auc_act", "sens_act", "sens_hpd")) {
  r <- summary_list[[m]]
  cat(sprintf("    %-12s: %.4f ± %.4f  [%.4f, %.4f]\n",
              m, r["mean"], r["sd"], r["min"], r["max"]))
}

cat("\n  [파라미터 복원]\n")
for (m in c("alpha_bias", "alpha_rmse", "beta_bias", "beta_rmse")) {
  r <- summary_list[[m]]
  cat(sprintf("    %-12s: %.4f ± %.4f  [%.4f, %.4f]\n",
              m, r["mean"], r["sd"], r["min"], r["max"]))
}

cat("\n  [수락률]\n")
for (m in c("acc_alpha", "acc_beta", "acc_w")) {
  r <- summary_list[[m]]
  cat(sprintf("    %-12s: %.4f ± %.4f\n", m, r["mean"], r["sd"]))
}

cat("\n  [진단]\n")
for (m in c("action_contrib", "cbar_sign_match", "lam_switch_med", "lam_switch_low")) {
  r <- summary_list[[m]]
  cat(sprintf("    %-18s: %.4f ± %.4f\n", m, r["mean"], r["sd"]))
}

cat("\n  [Sensitivity 손실 경로 (평균)]\n")
cat(sprintf("    1단계 통과: %.1f, 탈락: %.1f\n",
            summary_list$s1_tp["mean"], summary_list$s1_miss["mean"]))
cat(sprintf("    2단계 통과: %.1f, 탈락: %.1f\n",
            summary_list$s2_tp["mean"], summary_list$s2_miss["mean"]))

# 병목 빈도
bt_tab <- table(metrics_df$bottleneck)
cat(sprintf("    병목 빈도: %s\n",
            paste(sprintf("%s=%d", names(bt_tab), bt_tab), collapse = ", ")))

# --- 저장 ---
cat("\n  [저장]\n")

# 개별 seed 지표 csv
csv_path <- file.path(output_dir, "all_seeds_metrics.csv")
write.csv(metrics_df, file = csv_path, row.names = FALSE)
cat(sprintf("    개별 지표: %s\n", csv_path))

# 집계 결과 csv
summary_csv_path <- file.path(output_dir, "summary_statistics.csv")
write.csv(summary_df, file = summary_csv_path, row.names = FALSE)
cat(sprintf("    집계 결과: %s\n", summary_csv_path))

# 전체 결과 rds
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
cat(sprintf("    전체 결과: %s\n", full_rds_path))

# seed 기록
seed_log_path <- file.path(output_dir, "seed_log.txt")
writeLines(c(
  sprintf("실행 시각: %s", Sys.time()),
  sprintf("D: %d", D),
  sprintf("n_seeds: %d", length(sim_seeds)),
  sprintf("N_iter: %d, burn_in: %d, thin: %d", config$N_iter, config$burn_in, config$thin),
  sprintf("master_seed: %s", ifelse(is.null(USER_CONFIG$master_seed), "NULL (random)", 
                                     as.character(USER_CONFIG$master_seed))),
  sprintf("sim_seeds: %s", paste(sim_seeds, collapse = ", "))
), con = seed_log_path)
cat(sprintf("    Seed 기록: %s\n", seed_log_path))

cat("\n=== 완료 ===\n")
