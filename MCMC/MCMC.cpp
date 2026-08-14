#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::depends("RcppArmadillo")]]

// =============================================================================
// Model Structure (Multi-dimensional Action IRT)
// =============================================================================
// Fixed effect Beta / Random effect Alpha
// logit(pi_ij) = alpha_i + beta_j + sum_{l=1}^{N_j} sum_{d=1}^{D} w^(d)_jl * C_bar^(d)_ijl * I(l in A_ij)
//
// alpha_i ~ N(0, sigma_alpha)                 : respondent ability
// beta_j  ~ N(0, 1)                           : item difficulty
// w^(d)_jl | lambda^(d)_jl ~ spike-and-slab   : action weight (independent across dimensions)
// lambda^(d)_jl ~ Bernoulli(0.5)
// sigma_alpha ~ Inverse-Gamma(2, 1)
//
// =============================================================================
// Data Structure (flattening strategy: action-major ordering)
// =============================================================================
// data: (n_resp x n_prob) response matrix
// C_sum_list[[j]]: (n_resp x N_j * D) matrix
//   column order: [l=0,d=0] [l=0,d=1] ... [l=0,d=D-1] [l=1,d=0] ... [l=N_j-1,d=D-1]
//   indexing: col_index = l * D + d
// N_j_vec: (n_prob,) number of unique action types per item (before flattening)
// D: number of latent dimensions

// =============================================================================
// Numerically Stable Helper Functions
// =============================================================================

inline double log_sigmoid(double x) {
  if (x > 0) {
    return -log1p(exp(-x));
  } else {
    return x - log1p(exp(x));
  }
}

inline double log_one_minus_sigmoid(double x) {
  if (x > 0) {
    return -x - log1p(exp(-x));
  } else {
    return -log1p(exp(x));
  }
}

inline double log_lik_single(double y, double eta) {
  if (y == 1.0) {
    return log_sigmoid(eta);
  } else {
    return log_one_minus_sigmoid(eta);
  }
}

// =============================================================================
// 1. Log Posterior Computation Functions
// =============================================================================

// Log posterior for Beta_j
// eta_ij = alpha_i + beta_j + WC_cache_j[i]
// [[Rcpp::export]]
double log_post_beta(const vec& data_j,
                        double candi_beta,
                        const vec& alpha,
                        const vec& WC_cache,
                        double sigma_beta) {
  
  int n_resp = data_j.n_elem;
  double log_likelihood = 0.0;
  
  for (int i = 0; i < n_resp; i++) {
    if (std::isnan(data_j[i])) continue;
    double eta = alpha[i] + candi_beta + WC_cache[i];
    log_likelihood += log_lik_single(data_j[i], eta);
  }
  
  double log_prior = -0.5 * log(2.0 * M_PI * sigma_beta) - 0.5 * candi_beta * candi_beta / sigma_beta;
  return log_likelihood + log_prior;
}

// Log posterior for Alpha_i
// [[Rcpp::export]]
double log_post_alpha(const vec& data_i,
                        double candi_alpha,
                        const vec& beta,
                        const vec& WC_i_vec,
                        double sigma_alpha) {
  
  int n_prob = data_i.n_elem;
  double log_likelihood = 0.0;
  
  for (int j = 0; j < n_prob; j++) {
    if (std::isnan(data_i[j])) continue;
    double eta = candi_alpha + beta[j] + WC_i_vec[j];
    log_likelihood += log_lik_single(data_i[j], eta);
  }
  
  double log_prior = -0.5 * log(2.0 * M_PI * sigma_alpha) - 0.5 * candi_alpha * candi_alpha / sigma_alpha;
  return log_likelihood + log_prior;
}

// Log posterior for W^(d)_jl (incremental computation)
// Update for a single (j, l, d) component
// C_sum_col: the (l*D+d)-th column of C_sum_list[j]
// [[Rcpp::export]]
double log_post_w(const vec& data_j,
                    double candi_w,
                    double current_w,
                    const vec& alpha,
                    double beta_j,
                    const vec& WC_cache,
                    const vec& C_sum_col,
                    double lambda_jld,
                    double tau2,
                    double nu2) {
  
  int n_resp = data_j.n_elem;
  double log_likelihood = 0.0;
  double delta_w = candi_w - current_w;
  
  for (int i = 0; i < n_resp; i++) {
    if (std::isnan(data_j[i])) continue;
    double eta = alpha[i] + beta_j + WC_cache[i] + delta_w * C_sum_col[i];
    log_likelihood += log_lik_single(data_j[i], eta);
  }
  
  // Spike-and-slab prior
  double var = (lambda_jld < 0.5) ? tau2 : nu2;
  double log_prior = -0.5 * log(2.0 * M_PI * var) - 0.5 * candi_w * candi_w / var;
  
  return log_likelihood + log_prior;
}

// =============================================================================
// 2. Cache Management Functions
// =============================================================================

// Initial computation of WC_cache
// C_sum_j: (n_resp x N_j*D), W_j: (N_j*D,)
// WC_cache[i] = sum_{l,d} W_j[l*D+d] * C_sum_j(i, l*D+d)
vec compute_WC_cache(const mat& C_sum_j, const vec& W_j) {
  return C_sum_j * W_j;
}

// Incremental update for a single component
void update_WC_cache(vec& WC_cache, const vec& C_sum_col, double delta_w) {
  WC_cache += delta_w * C_sum_col;
}

// =============================================================================
// 3. Main MCMC Sampler
// =============================================================================

// [[Rcpp::export]]
List MCMC_action_model(int N_iter,
                          mat data,                    // (n_resp x n_prob) response matrix
                          vector<mat> C_sum_list,      // C_sum_list[[j]]: (n_resp x N_j*D) flattened matrix
                          vec N_j_vec,                 // (n_prob,) number of unique action types per item (before flattening)
                          int D,                       // number of latent dimensions
                          vec alpha_init,              // (n_resp,)
                          vec beta_init,               // (n_prob,)
                          vector<vec> W_init,          // W_init[[j]]: (N_j*D,) flattened vector
                          vector<vec> lambda_init,     // lambda_init[[j]]: (N_j*D,) flattened vector
                          double sigma_alpha_init,
                          double tau2,                 // spike variance
                          double nu2,                  // slab variance
                          double proposal_sd_alpha = 0.4,
                          double proposal_sd_beta = 0.4,
                          double proposal_sd_w = 0.4,
                          int burn_in = 0,             // burn-in period
                          int thin = 1) {              // thinning interval
  
  int n_resp = data.n_rows;
  int n_prob = data.n_cols;
  
  // Current parameter values
  vec now_alpha = alpha_init;
  vec now_beta = beta_init;
  vector<vec> now_W = W_init;           // now_W[j]: length N_j*D
  vector<vec> now_lambda = lambda_init; // now_lambda[j]: length N_j*D
  double now_sigma_alpha = sigma_alpha_init;
  
  // Beta prior: fixed as N(0, 1)
  double now_sigma_beta = 1.0;
  
  // ==========================================================================
  // Initialize WC cache
  // ==========================================================================
  vector<vec> WC_caches(n_prob);
  for (int j = 0; j < n_prob; j++) {
    WC_caches[j] = compute_WC_cache(C_sum_list[j], now_W[j]);
  }
  
  // Compute total number of flattened W parameters: sum(N_j) * D
  int w_total_flat = 0;
  for (int j = 0; j < n_prob; j++) {
    w_total_flat += (int)N_j_vec[j] * D;
  }
  
  // ==========================================================================
  // Allocate result storage (post burn-in, with thinning)
  // ==========================================================================
  int n_save = 0;
  for (int iter = burn_in; iter < N_iter; iter++) {
    if ((iter - burn_in) % thin == 0) n_save++;
  }
  
  mat result_alpha(n_save, n_resp);
  mat result_beta(n_save, n_prob);
  mat result_W(n_save, w_total_flat);
  mat result_lambda(n_save, w_total_flat);
  mat result_sigma(n_save, 2);
  
  int save_idx = 0;
  
  // Acceptance rate counters
  vec accept_alpha = zeros(n_resp);
  vec accept_beta = zeros(n_prob);
  vec accept_w = zeros(w_total_flat);
  
  // MCMC iterations
  for (int iter = 0; iter < N_iter; iter++) {
    
    if ((iter + 1) % 500 == 0) {
      Rcout << "Iteration: " << (iter + 1) << " / " << N_iter << std::endl;
    }
    
    // =========================================================================
    // Step 1: Update Beta (MH) - fixed prior N(0, 1)
    // =========================================================================
    for (int j = 0; j < n_prob; j++) {
      double candi_beta = R::rnorm(now_beta[j], proposal_sd_beta);
      
      double log_num = log_post_beta(data.col(j), candi_beta,
                                        now_alpha, WC_caches[j], now_sigma_beta);
      double log_den = log_post_beta(data.col(j), now_beta[j],
                                        now_alpha, WC_caches[j], now_sigma_beta);
      
      if (log(R::runif(0, 1)) < (log_num - log_den)) {
        now_beta[j] = candi_beta;
        accept_beta[j] += 1.0;
      }
    }
    
    // =========================================================================
    // Step 2: Update Alpha (MH)
    // =========================================================================
    for (int i = 0; i < n_resp; i++) {
      double candi_alpha = R::rnorm(now_alpha[i], proposal_sd_alpha);
      
      vec WC_i_vec(n_prob);
      for (int j = 0; j < n_prob; j++) {
        WC_i_vec[j] = WC_caches[j][i];
      }
      
      double log_num = log_post_alpha(data.row(i).t(), candi_alpha,
                                        now_beta, WC_i_vec, now_sigma_alpha);
      double log_den = log_post_alpha(data.row(i).t(), now_alpha[i],
                                        now_beta, WC_i_vec, now_sigma_alpha);
      
      if (log(R::runif(0, 1)) < (log_num - log_den)) {
        now_alpha[i] = candi_alpha;
        accept_alpha[i] += 1.0;
      }
    }
    
    // =========================================================================
    // Step 3: Update W (MH) - triple loop over (j, l, d)
    // =========================================================================
    int w_idx = 0;
    for (int j = 0; j < n_prob; j++) {
      int N_j = (int)N_j_vec[j];
      
      for (int l = 0; l < N_j; l++) {
        for (int d = 0; d < D; d++) {
          int flat_idx = l * D + d;  // flattened index
          
          double candi_w = R::rnorm(now_W[j][flat_idx], proposal_sd_w);
          
          vec C_sum_col = C_sum_list[j].col(flat_idx);
          
          double log_num = log_post_w(data.col(j), candi_w, now_W[j][flat_idx],
                                        now_alpha, now_beta[j], WC_caches[j],
                                        C_sum_col, now_lambda[j][flat_idx], tau2, nu2);
          double log_den = log_post_w(data.col(j), now_W[j][flat_idx], now_W[j][flat_idx],
                                        now_alpha, now_beta[j], WC_caches[j],
                                        C_sum_col, now_lambda[j][flat_idx], tau2, nu2);
          
          if (log(R::runif(0, 1)) < (log_num - log_den)) {
            double delta_w = candi_w - now_W[j][flat_idx];
            update_WC_cache(WC_caches[j], C_sum_col, delta_w);
            now_W[j][flat_idx] = candi_w;
            accept_w[w_idx] += 1.0;
          }
          
          w_idx++;
        }
      }
    }
    
    // =========================================================================
    // Step 4: Update Lambda (Gibbs) - independently for each (j, l, d)
    // =========================================================================
    for (int j = 0; j < n_prob; j++) {
      int N_j = (int)N_j_vec[j];
      
      for (int l = 0; l < N_j; l++) {
        for (int d = 0; d < D; d++) {
          int flat_idx = l * D + d;
          double w_jld = now_W[j][flat_idx];
          
          double log_spike = -0.5 * log(tau2) - 0.5 * w_jld * w_jld / tau2;
          double log_slab  = -0.5 * log(nu2)  - 0.5 * w_jld * w_jld / nu2;
          
          double max_log = std::max(log_spike, log_slab);
          double spike_lik = exp(log_spike - max_log);
          double slab_lik  = exp(log_slab - max_log);
          
          double posterior_prob = slab_lik / (spike_lik + slab_lik);
          now_lambda[j][flat_idx] = R::rbinom(1, posterior_prob);
        }
      }
    }
    
    // =========================================================================
    // Step 5: Update variance parameters (Gibbs)
    // =========================================================================
    
    // sigma_alpha ~ IG(2, 1)
    double ss_alpha = sum(pow(now_alpha, 2));
    double shape_alpha = n_resp / 2.0 + 2.0;
    double rate_alpha = ss_alpha / 2.0 + 1.0;
    now_sigma_alpha = 1.0 / R::rgamma(shape_alpha, 1.0 / rate_alpha);
    
    // =========================================================================
    // Save samples (post burn-in, with thinning)
    // =========================================================================
    if (iter >= burn_in && (iter - burn_in) % thin == 0) {
      result_alpha.row(save_idx) = now_alpha.t();
      result_beta.row(save_idx) = now_beta.t();
      result_sigma(save_idx, 0) = now_sigma_alpha;
      result_sigma(save_idx, 1) = now_sigma_beta;
      
      w_idx = 0;
      for (int j = 0; j < n_prob; j++) {
        int N_j = (int)N_j_vec[j];
        for (int l = 0; l < N_j; l++) {
          for (int d = 0; d < D; d++) {
            int flat_idx = l * D + d;
            result_W(save_idx, w_idx) = now_W[j][flat_idx];
            result_lambda(save_idx, w_idx) = now_lambda[j][flat_idx];
            w_idx++;
          }
        }
      }
      save_idx++;
    }
  }
  
  // Compute acceptance rates
  accept_alpha /= N_iter;
  accept_beta /= N_iter;
  accept_w /= N_iter;
  
  return List::create(
    Named("alpha") = result_alpha,
    Named("beta") = result_beta,
    Named("W") = result_W,
    Named("lambda") = result_lambda,
    Named("sigma") = result_sigma,
    Named("accept_alpha") = accept_alpha,
    Named("accept_beta") = accept_beta,
    Named("accept_w") = accept_w,
    Named("N_j_vec") = N_j_vec,
    Named("D") = D,
    Named("n_save") = n_save,
    Named("burn_in") = burn_in,
    Named("thin") = thin
  );
}