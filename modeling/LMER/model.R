# install.packages(c('brms', 'devtools'))
install.packages("rstan")

library(brms)

getwd()
results <- read.csv("Documents/Uni-Saarland/thesis/modeling/LMER/formatted_results.csv")
which(is.na(results)==T)

boxplot(derivationStrength ~ speakerType, col=c("white","lightgray"),results)
boxplot(derivationStrength ~ speakerType*speakerGender, col=c("white","lightgray"),results)

results.fullmodel = lmer(
  derivationStrength ~ speakerType + speakerGender + ToM + freqRatio + itemType
                      + speakerType * itemType
                      + speakerType * freqRatio
                      + speakerType * speakerGender
                      + speakerType * ToM
                      + itemType * ToM
                      + speakerType * ToM * itemType
                      + speakerType * freqRatio * itemType
                      + (1|itemId) + (1|participantId)
                      + (1+speakerType+speakerGender|participantId), # random order of speakers impact on participants
  data=results,
  REML=FALSE
)

# H1 null model, removing speakerType
results.H1nullmodel = lmer(
  derivationStrength ~ speakerGender + ToM + freqRatio + itemType
  + itemType * ToM
  + (1|itemId) + (1|participantId) 
  + (1+speakerType+speakerGender|participantId),
  data=results,
  REML=FALSE
)
# H1 speakerType vs null
anova(results.fullmodel,results.H1nullmodel)


# H2 null model, removing freqRatio
results.H2nullmodel = lmer(
  derivationStrength ~ speakerType + speakerGender + ToM + itemType
                      + speakerType * itemType
                      + speakerType * speakerGender
                      + speakerType * ToM
                      + itemType * ToM
                      + speakerType * ToM * itemType
                      + (1|itemId) + (1|participantId) 
                      + (1+speakerType+speakerGender|participantId),
  data=results,
  REML=FALSE
)
# H2 freqRatio vs null
anova(results.fullmodel,results.H2nullmodel)


# H3 null model, removing ToM
results.H3nullmodel = lmer(
  derivationStrength ~ speakerType + speakerGender + freqRatio + itemType
                      + speakerType * itemType
                      + speakerType * freqRatio
                      + speakerType * speakerGender
                      + speakerType * freqRatio * itemType
                      + (1|itemId) + (1|participantId) 
                      + (1+speakerType+speakerGender|participantId),
  data=results,
  REML=FALSE
)
# H3 ToM vs null
anova(results.fullmodel,results.H3nullmodel)

