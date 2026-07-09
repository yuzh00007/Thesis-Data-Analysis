library(brms)
library(bayesplot)
library(tidybayes)
library(ggplot2)
library(dplyr)

currentDir = "/home/yzhang/Documents/Uni-Saarland/thesis/modeling/BMRS/"
setwd(currentDir)

set.seed(123)

# stvincent <- read_csv('https://usda-ree-ars.github.io/SEAStats/brms_crash_course/stvincent.csv')
stvincent <- stvincent %>%
  mutate(across(c(site, block, plot, N, P, K), factor))

fit_interceptonly <- brm(yield ~ 1 + (1 | site) + (1 | block:site),
                         data = stvincent,
                         chains = 2,
                         iter = 200,
                         warmup = 100,
                         init = 'random',
                         seed = 1)

# Fit a Bayesian linear regression
#   car miles per gallon ~ weight of car + no. of cylinders
model1 <- brm(
  mpg ~ wt + cyl,
  data = mtcars,
  family = gaussian(),  # normal distribution for lin.reg
  chains = 4,
  iter = 2000,
  seed = 123
)

summary(model1)  # includes r-hat for all coefficients (should be at least < 1.05, best close to 1.0)
fixef(model1)  # simplified summary, includes 95% "credible interval" (NOT confidence interval) betw Q2.5 and Q97.5 values
rhat(model1)
neff_ratio(model1)

# credible interval: 
#  - bayesian
#  - "given the observed data, the effect has XX% probability of falling within this range"
#  - a direct statement about the parameter, not about hypothetical repeated samples.
# confidence interval: 
#  - frequentist
#  - "there is an XX% probability that when computing a confidence interval from data of this sort, the effect falls within this range"
#  - with a large number of repeated samples, XX% of such calculated confidence intervals would include the true value of the parameter.

# Plot posterior distributions
mcmc_plot(model1, type = "areas", pars = "^b_")

# Posterior density plots
mcmc_areas(
  model1, 
  pars = c("b_wt", "b_cyl"), 
  prob = 0.95
) + labs(title = "Posterior Distributions of Coefficients")


# Get posterior predictive samples
pp <- posterior_predict(model1, newdata = data.frame(wt = c(2, 3, 4), cyl = c(4, 6, 8)))

# pp is a matrix: rows = iterations, columns = new observations
dim(pp)  # [1] 4000    3

# Summarize predictions
apply(pp, 2, mean)
apply(pp, 2, quantile, probs = c(0.025, 0.975))

# Trace plots
plot(model1)
mcmc_trace(model1, pars = c("b_wt", "b_cyl"))
mcmc_rank_overlay(as.array(model1))  # Ideal rank plots are uniform across ranks

# posterior predictive distribution overlayed with observations
pp_check(model1)


# specifying priors
priors <- c(
  prior(normal(0, 10), class = Intercept),
  prior(normal(-3, 1), class = b, coef = "wt"),
  prior(normal(0, 2), class = b, coef = "cyl"),
  prior(exponential(1), class = sigma)
)
model2 <- brm(
  mpg ~ wt + cyl,
  data = mtcars,
  family = gaussian(),
  prior = priors,
  chains = 4,
  seed = 123
)
summary(model2)
fixef(model2)
