# ============================================================
# R Environment Setup
# ============================================================
# This project uses renv for R package version management.
# To initialize: run `renv::init()` in the project root.
#
# Required R packages and tested versions:
#
# MCMC and computation:
#   Rcpp          >= 1.0.12
#   RcppArmadillo >= 0.12.8
#
# Data manipulation:
#   dplyr         >= 1.1.4
#
# MCMC diagnostics:
#   coda          >= 0.19-4
#
# Visualization:
#   ggplot2       >= 3.5.0
#   gridExtra     >= 2.3
#
# Utilities:
#   glue          >= 1.7.0
#
# R version tested: >= 4.3.0
# ============================================================

packages <- c(
  "Rcpp",
  "RcppArmadillo",
  "dplyr",
  "coda",
  "ggplot2",
  "gridExtra",
  "glue"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

invisible(lapply(packages, install_if_missing))

cat("All required R packages are installed.\n")
