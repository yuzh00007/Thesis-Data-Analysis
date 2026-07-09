# data creation
pitch = c(233,204,242,130,112,142)  # concatenation
sex = c(rep("female",3),rep("male",3))

my.df = data.frame(sex,pitch)


# linear model
xmdl = lm(pitch ~ sex, my.df)
summary(xmdl)



# age and pitch
age = c(14,23,35,48,52,67)
pitch = c(252,244,240,233,212,204)
my.df = data.frame(age,pitch)
xmdl = lm(pitch ~ age, my.df)
summary(xmdl)

# meaningful intercepts by subtracting mean age from all ages
# centered data, intercept = predicted voice pitch at avg age
my.df$age.c = my.df$age - mean(my.df$age)
xmdl = lm(pitch ~ age.c, my.df)
summary(xmdl)

# residual plot
plot(fitted(xmdl),residuals(xmdl))

################
# mixed models #
################

# the new lme4 2.0-1 does not work LOL, have to downgrade
packageurl <- "https://cran.r-project.org/src/contrib/Archive/lme4/lme4_1.1-38.tar.gz"
install.packages(packageurl, repos=NULL, type="source")

library(lme4)

politeness= read.csv("http://www.bodowinter.com/tutorial/politeness_data.csv")
which(is.na(politeness)==T)

boxplot(frequency ~ attitude*gender, col=c("white","lightgray"),politeness)

politeness.model = lmer(
  frequency ~ attitude + gender 
              + (1|subject) + (1|scenario), 
  data=politeness
)
summary(politeness.model)


# comparing two models - likelihood ratio test
politeness.null = lmer(
  frequency ~ gender +
              (1|subject) + (1|scenario), 
  data=politeness,
  REML=FALSE
)
politeness.model = lmer(
  frequency ~ attitude + gender 
  + (1|subject) + (1|scenario), 
  data=politeness,
  REML=FALSE
)
anova(politeness.null,politeness.model)

politeness.interaction = lmer(
  frequency ~ attitude*gender
              + (1|subject) + (1|scenario), 
  data=politeness,
  REML=FALSE
)
anova(politeness.interaction,politeness.model)

# random slopes model
coef(politeness.model)

politeness.model = lmer(
  frequency ~ attitude + gender + 
              (1+attitude|subject) + (1+attitude|scenario),
  data=politeness,
  REML=FALSE
)
anova(politeness.null,politeness.model)
coef(politeness.model)