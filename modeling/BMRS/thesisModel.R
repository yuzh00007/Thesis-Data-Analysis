library(brms)
library(bayesplot)
library(tidybayes)
library(ggplot2)
library(dplyr)

getwd()
results <- read.csv("./formatted_results.csv")
set.seed(123)

# FORMULA 
formula1 <- brmsformula(
  derivationStrength ~ speakerType + speakerGender + ToM + freqRatio + itemType
  + speakerType * itemType
  + speakerType * freqRatio
  + speakerType * speakerGender
  + speakerType * ToM
  + itemType * ToM
  + speakerType * ToM * itemType
  + speakerType * freqRatio * itemType
  + (1|itemId)
  # + (1|participantId) throws error 
  + (1+speakerType+speakerGender|participantId)
)

# PRIORS
model1_priors <- c(
  prior(normal(0, 5), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior_string("lkj(2)", class = "cor")
)


# MODEL
model1 <- brm(
  formula1,
  data = results,
  family = gaussian(),
  prior = model1_priors, 
  chains = 4,
  iter = 2000,
  seed = 123
)

summary(model1) 
fixef(model1)

pp_check(model1, ndraws = 30)





citation()
