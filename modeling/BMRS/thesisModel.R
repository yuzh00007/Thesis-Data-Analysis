library(brms)
library(bayesplot)
library(tidybayes)
library(ggplot2)
library(dplyr)

# getwd()
# currentDir = "/home/yzhang/Documents/Uni-Saarland/thesis/modeling/BMRS/"
# setwd(currentDir)

results <- read.csv("./formatted_results.csv")
set.seed(123)

# print(
#   get_prior(
#     derivationStrength ~ speakerType + speakerGender + nathan_score_full + freqRatio + itemType
#     + speakerType * itemType
#     + speakerType * freqRatio
#     + speakerType * speakerGender
#     + speakerType * nathan_score_full
#     + itemType * nathan_score_full
#     + speakerType * nathan_score_full * itemType
#     + speakerType * freqRatio * itemType
#     + (1|itemId)
#     + (1+speakerType+speakerGender|participantId),
#     data=results, family=brmsfamily("normal", "log")
#   ),
#   options("width"=2000)
# )

# FORMULA
formula1 <- brmsformula(
  derivationStrength ~ speakerType + speakerGender + nathan_score_full + freqRatio + itemType
  + speakerType * itemType
  + speakerType * freqRatio
  + speakerType * speakerGender
  + speakerType * nathan_score_full
  + itemType * nathan_score_full
  + speakerType * nathan_score_full * itemType
  + speakerType * freqRatio * itemType
  + (1|itemId)
  + (1+speakerType+speakerGender|participantId)
)

# PRIORS
model1_priors <- c(
  # fixed effects
  prior(normal(87.5, 12.5), class = "Intercept"),
  prior(normal(0, 5), class = "b"),
  prior(normal(0, 12.5), class = "b", coef="speakerTypenonnative"),
  prior(normal(-25, 12.5), class = "b", coef="itemTypeunambiguous"),
  prior(normal(0, 12.5), class = "b", coef="speakerTypenonnative:itemTypeunambiguous"),

  # random effects
  prior(student_t(3, 0, 12.5), class = "sd", group="itemId"),
  prior(student_t(3, 0, 12.5), class = "sd", group="participantId"),

  # residual standard deviation?
  prior(student_t(3, 0, 2.5), class = "sd"),

  # correlation structure
  prior_string("lkj(2)", class = "cor")
)


# MODEL
model1 <- brm(
  formula1,
  data = results,
  family = gaussian(),
  prior = model1_priors,
  chains = 4,
  warmup=1000,
  iter = 4000,
  seed = 123
)

sink("./output/modelSummary.txt")
print(summary(model1, options(width=2000)))
sink()



pp_check(model1, ndraws = 30)
# citation()
