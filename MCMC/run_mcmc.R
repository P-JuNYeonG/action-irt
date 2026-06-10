library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(coda)
library(ggplot2)
library(gridExtra)
library(glue)

print("Fixed Beta, Robust Scaling")

## -- Data loading -- ##
model_name <- "lstm_ae" #lstm_ae, mlp_ae, pca
D <- 1 # D = 1, 2, 3, 4, 5
PRE <- "robust" # robust, winsor, wo_pre

## -- C++ Compile -- ## 
sourceCpp("/Users/jun_yeong/Desktop/Log_Process/02_Module/03_IRT/00_MCMC.cpp")

save_path <- glue("/home/zmzm106/R_file/95_multi_result/{model_name}_{D}_{PRE}.RData")
OUTPUT_DIR <- "/home/zmzm106/R_file/92_multi_conv"

# 2. 설정
set.seed(2025)
iteration <- 50000

base_path <- file.path("/Users/jun_yeong/Desktop/Log_Process/Workspace/Legacy/01_Data/model_output", model_name)

files <- list.files(
  path = base_path,
  pattern = sprintf("^long_format_%s_ps[0-9]_[0-9]+_D%d\\.csv$", model_name, D),
  full.names = TRUE
)

data_list <- lapply(files, read.csv)
data_raw <- do.call(rbind, data_list)

data <- data_raw[complete.cases(data_raw), ]

# C_value1, C_value2, ..., C_valueD 열 이름 생성
C_col_names <- paste0("C_value", 1:D)

# ─── 로버스트 스케일링 ───
for (col_name in C_col_names) {
  data <- data %>%
    group_by(problem_num) %>% # 각 문제 각각 별도로 scaling 된다.
    mutate(
      !!col_name := {
        vals <- .data[[col_name]] # 현재 열의 값을 가져오고,
        med <- median(vals, na.rm = TRUE) # 중앙값, IQR 계산
        iqr_val <- IQR(vals, na.rm = TRUE)
        if (iqr_val > 0) {
          (vals - med) / iqr_val
        } else {
          rep(0, length(vals))
        }
      }
    ) %>%
    ungroup()
}

#####################################################################
# 고유 응답자 및 문제 추출
#####################################################################
unique_students <- unique(data$seq_id)
unique_problems <- unique(data$problem_num)

n_students <- length(unique_students)
n_problems <- length(unique_problems)

#####################################################################
# 1. 응답 데이터 행렬 생성 (n_students × n_problems)
# 2. N_j_vec 생성: 각 문제의 고유 행동 종류 수
#####################################################################
response_matrix <- matrix(NA, nrow = n_students, ncol = n_problems)
rownames(response_matrix) <- unique_students
colnames(response_matrix) <- unique_problems

response_data <- data %>%
  group_by(seq_id, problem_num) %>%
  summarise(outcome = first(outcome), .groups = "drop")

for(i in 1:nrow(response_data)) {
  student_idx <- which(unique_students == response_data$seq_id[i])
  problem_idx <- which(unique_problems == response_data$problem_num[i])
  response_matrix[student_idx, problem_idx] <- response_data$outcome[i]
}

#####################################################################
N_j_vec <- sapply(unique_problems, function(p) {
  max(data$behavior_id[data$problem_num == p])
})
names(N_j_vec) <- unique_problems

#####################################################################
# 3. C_list 생성: 펼친 행렬 (n_students × N_j * D)
#####################################################################
C_list <- vector("list", n_problems)
names(C_list) <- unique_problems

for(p in 1:n_problems) {
  prob_name <- unique_problems[p]
  N_j <- N_j_vec[p]
  
  problem_data <- data %>% filter(problem_num == prob_name)
  
  # 다차원으로 확장을 위해서 열의 개수를 D만큼 multiply
  C_matrix_flat <- matrix(0, nrow = n_students, ncol = N_j * D)
  rownames(C_matrix_flat) <- unique_students
  
  for (d in 1:D) {
    col_name <- C_col_names[d]
    
    # 고유 행동의 대표값(평균)을 구하기 위해서 평균으로 집계
    aggregated <- problem_data %>%
      group_by(seq_id, behavior_id) %>%
      summarise(C_agg = mean(.data[[col_name]], na.rm = TRUE), .groups = "drop")
    
    for (i in 1:nrow(aggregated)) {
      student_idx <- which(unique_students == aggregated$seq_id[i])
      action_idx <- aggregated$behavior_id[i]
      
      if (length(student_idx) > 0 && action_idx <= N_j) {
        flat_col <- (action_idx - 1) * D + d
        C_matrix_flat[student_idx, flat_col] <- aggregated$C_agg[i]
      }
    }
  }
  
  C_list[[p]] <- C_matrix_flat
}

total_W_params <- sum(N_j_vec) * D
print(paste("총 W 파라미터 수 (N_j * D):", total_W_params))
print(paste("Alpha 파라미터 수:", n_students))
print(paste("Beta 파라미터 수:", n_problems))
print(paste("Latent Dimension D:", D))

#####################################################################
# Model loading
#####################################################################

print("MCMC 시작...")
start_time <- Sys.time()

result <- MCMC_action_model_v5(
  N_iter = iteration,
  data = response_matrix,
  C_sum_list = C_list,
  N_j_vec = N_j_vec,
  D = D,
  alpha_init = rnorm(n_students, 0, 0.1),
  beta_init = rnorm(n_problems, 0, 0.1),
  W_init = lapply(N_j_vec, function(n) rnorm(n * D, 0, 0.1)),
  lambda_init = lapply(N_j_vec, function(n) rep(0.5, n * D)),
  sigma_alpha_init = 1.0,
  tau2 = 0.001,
  nu2 = 2.5,
  proposal_sd_alpha = 1.5,
  proposal_sd_beta = 0.5,
  proposal_sd_w = 0.4,
  burn_in = iteration * 0.2,
  thin = 10
)

end_time <- Sys.time()
print(paste("MCMC 완료! 소요 시간:", round(difftime(end_time, start_time, units = "mins"), 2), "분"))

save(result, file = save_path)
print(paste("결과 저장:", save_path))

#####################################################################
# 6. 수렴 진단 및 결과 요약
#####################################################################

# 사후 샘플 추출
post_alpha  <- result$alpha
post_beta   <- result$beta
post_W      <- result$W
post_lambda <- result$lambda
post_sigma  <- result$sigma

N_j_vec <- result$N_j_vec
D_result <- result$D
n_prob <- length(N_j_vec)
n_resp <- ncol(post_alpha)

# 수락률 확인
cat("\n===== 수락률 요약 =====\n")
cat("Beta 수락률:\n")
print(summary(result$accept_beta))

cat("\nAlpha 수락률:\n")
print(summary(result$accept_alpha))

cat("\nW 수락률:\n")
print(summary(result$accept_w))

# ESS 계산
cat("\n", "=", rep("=", 60), "\n", sep = "")
cat("Alpha, Beta, W ESS 계산\n")
cat("=", rep("=", 60), "\n", sep = "")

calc_ess <- function(samples) {
  if (is.vector(samples)) {
    return(effectiveSize(as.mcmc(samples)))
  } else {
    return(apply(samples, 2, function(x) effectiveSize(as.mcmc(x))))
  }
}

alpha_ess <- calc_ess(post_alpha)
cat("\nAlpha ESS:\n")
cat("  Mean:", round(mean(alpha_ess), 2), "\n")
cat("  Min:", round(min(alpha_ess), 2), "\n")
cat("  Max:", round(max(alpha_ess), 2), "\n")

beta_ess <- calc_ess(post_beta)
cat("\nBeta ESS:\n")
cat("  Mean:", round(mean(beta_ess), 2), "\n")
cat("  Min:", round(min(beta_ess), 2), "\n")
cat("  Max:", round(max(beta_ess), 2), "\n")

w_ess <- calc_ess(post_W)
cat("\nW ESS:\n")
cat("  Mean:", round(mean(w_ess), 2), "\n")
cat("  Min:", round(min(w_ess), 2), "\n")
cat("  Max:", round(max(w_ess), 2), "\n")

cat("\nSigma 요약:\n")
cat("  sigma_alpha: mean =", round(mean(post_sigma[, 1]), 4), "\n")
cat("  sigma_beta:  mean =", round(mean(post_sigma[, 2]), 4), "\n")

# Burn-in/Thinning 정보 출력
cat("\n===== Burn-in / Thinning 정보 =====\n")
cat("  burn_in:", result$burn_in, "\n")
cat("  thin:", result$thin, "\n")
cat("  저장된 샘플 수 (n_save):", result$n_save, "\n")


# -----------------------------------------------------------------------------
# Alpha 시각화 (랜덤 9명 선택)
# -----------------------------------------------------------------------------

set.seed(2025)
sample_alpha_idx <- sample(1:n_resp, min(18, n_resp))

alpha_path <- glue("{model_name}_{D}_{PRE}_alpha.pdf")
pdf(file.path(OUTPUT_DIR, alpha_path), width = 12, height = 10)

# Trace plots
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
for (i in sample_alpha_idx) {
  plot(post_alpha[, i], type = "l", 
       main = paste0("Alpha[", i, "] Trace"),
       xlab = "Iteration", ylab = "Value",
       col = "steelblue")
}

# Density plots
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
for (i in sample_alpha_idx) {
  plot(density(post_alpha[, i]), 
       main = paste0("Alpha[", i, "] Density"),
       xlab = "Value", ylab = "Density",
       col = "steelblue", lwd = 2)
}

dev.off()

# -----------------------------------------------------------------------------
# Beta 시각화 (전체)
# -----------------------------------------------------------------------------

beta_path <- glue("{model_name}_{D}_{PRE}_beta.pdf")
pdf(file.path(OUTPUT_DIR, beta_path), width = 14, height = 10)

# Trace plots
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
for (j in 1:n_prob) {
  plot(post_beta[, j], type = "l", 
       xlab = "Iteration", ylab = "Value",
       col = "darkred")
}

# Density plots
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
for (j in 1:n_prob) {
  plot(density(post_beta[, j]), 
       xlab = "Value", ylab = "Density",
       col = "darkred", lwd = 2)
}

dev.off()