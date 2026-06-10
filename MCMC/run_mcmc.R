library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(coda)
library(ggplot2)
library(gridExtra)
library(glue)

print("Fixed Beta, Robust Scaling")

## -- Data loading -- ##
model_name <- "lstm_ae"
D <- 1
PRE <- "robust"

## -- Compile C++ sampler -- ##
sourceCpp("/path/to/MCMC.cpp")

save_path <- glue("/path/to/results/{model_name}_{D}_{PRE}.RData")
OUTPUT_DIR <- "/path/to/output"

# 2. Settings
set.seed(2025)
iteration <- 50000

base_path <- file.path("/path/to/model_output", model_name)

files <- list.files(
  path = base_path,
  pattern = sprintf("^long_format_%s_ps[0-9]_[0-9]+_D%d\\.csv$", model_name, D),
  full.names = TRUE
)

data_list <- lapply(files, read.csv)
data_raw <- do.call(rbind, data_list)

data <- data_raw[complete.cases(data_raw), ]

# Generate column names: C_value1, C_value2, ..., C_valueD
C_col_names <- paste0("C_value", 1:D)

# --- Robust Scaling ---
for (col_name in C_col_names) {
  data <- data %>%
    group_by(problem_num) %>%
    mutate(
      !!col_name := {
        vals    <- .data[[col_name]]              # Retrieve values for the current column
        med     <- median(vals, na.rm = TRUE)     # Compute median and IQR
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
# Extract unique respondents and items
#####################################################################
unique_students <- unique(data$seq_id)
unique_problems <- unique(data$problem_num)

n_students <- length(unique_students)
n_problems <- length(unique_problems)

#####################################################################
# 1. Construct response data matrix (n_students x n_problems)
# 2. Construct N_j_vec: number of unique action types per item
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
# 3. Construct C_list: flattened matrix (n_students x N_j * D)
#####################################################################
C_list <- vector("list", n_problems)
names(C_list) <- unique_problems

for(p in 1:n_problems) {
  prob_name <- unique_problems[p]
  N_j <- N_j_vec[p]
  
  problem_data <- data %>% filter(problem_num == prob_name)
  
  # Expand to D dimensions by multiplying the number of columns by D
  C_matrix_flat <- matrix(0, nrow = n_students, ncol = N_j * D)
  rownames(C_matrix_flat) <- unique_students
  
  for (d in 1:D) {
    col_name <- C_col_names[d]
    
    # Aggregate per unique action using the mean as the representative value
    aggregated <- problem_data %>%
      group_by(seq_id, behavior_id) %>%
      summarise(C_agg = mean(.data[[col_name]], na.rm = TRUE), .groups = "drop")
    
    for (i in 1:nrow(aggregated)) {
      student_idx <- which(unique_students == aggregated$seq_id[i])
      action_idx  <- aggregated$behavior_id[i]
      
      if (length(student_idx) > 0 && action_idx <= N_j) {
        flat_col <- (action_idx - 1) * D + d
        C_matrix_flat[student_idx, flat_col] <- aggregated$C_agg[i]
      }
    }
  }
  
  C_list[[p]] <- C_matrix_flat
}

print(paste("Total W parameters (N_j * D):", sum(N_j_vec) * D))
print(paste("Number of Alpha parameters:", n_students))
print(paste("Number of Beta parameters:", n_problems))
print(paste("Latent Dimension D:", D))

#####################################################################
# Run MCMC sampler
#####################################################################

print("Starting MCMC...")
start_time <- Sys.time()

result <- MCMC_action_model(
  N_iter            = iteration,
  data              = response_matrix,
  C_sum_list        = C_list,
  N_j_vec           = N_j_vec,
  D                 = D,
  alpha_init        = rnorm(n_students, 0, 0.1),
  beta_init         = rnorm(n_problems, 0, 0.1),
  W_init            = lapply(N_j_vec, function(n) rnorm(n * D, 0, 0.1)),
  lambda_init       = lapply(N_j_vec, function(n) rep(0.5, n * D)),
  sigma_alpha_init  = 1.0,
  tau2              = 0.001,
  nu2               = 2.5,
  proposal_sd_alpha = 1.5,
  proposal_sd_beta  = 0.5,
  proposal_sd_w     = 0.4,
  burn_in           = iteration * 0.2,
  thin              = 10
)

end_time <- Sys.time()
save(result, file = save_path)