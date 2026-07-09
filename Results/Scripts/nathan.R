library(tidyverse)
library(jsonlite)
library(dplyr)
library(stringr) 
library(purrr) 
library(tidyr)

getwd()
currentDir = "/home/yzhang/Documents/Uni-Saarland/thesis/Results/Scripts/"
setwd(currentDir)

data_raw <- read.csv("tomResults.csv")

data <- data_raw %>%
    filter(!workerid %in% c("test"))


df_parsed <- data %>%
  mutate(
    clean_json = str_replace_all(answer, '^"|"$', ""), 
    clean_json = str_replace_all(clean_json, '\\\\"', '"')
  )
df_parsed <- df_parsed %>%
  mutate(answer_list = map(clean_json, ~ safely(fromJSON)(.)$result))

extract <- \(field, default) {
  \(x) if (is.list(x) && field %in% names(x)) x[[field]] else default
}

# null value in 
df_parsed


df_parsed <- df_parsed %>%
  mutate(
    answer_value = map_chr(answer_list, extract("answer", NA_character_)),
    rt           = map_dbl(answer_list, extract("RT", NA_real_)),
    timeout      = map_dbl(answer_list, extract("timeout", NA_real_)),
    feedback     = map(answer_list, extract("feedback", 0))
  )

#df_parsed <- df_parsed %>%
#  unnest_wider(feedback, names_sep = "_")


#feedback
# df_feedback <- df_parsed %>%
#   filter(map_lgl(feedback, ~ !is.null(.)))
# feedback <- df_feedback %>%
#   dplyr::select(workerid, feedback)
# feedback <- feedback %>%
#   unnest_wider(feedback, names_sep = "_")
# 
# write.csv(feedback, "nathan2_feedback.csv")

################################################
# problems with videos

fail_summary <- df_parsed %>%
  filter(answer_value == -1) %>%
  
  group_by(workerid, questiontype) %>%
  summarise(n_failed_type = n(), .groups = "drop") %>%
  
  pivot_wider(
    names_from = questiontype,
    values_from = n_failed_type,
    values_fill = 0
  ) %>%
  
  mutate(n_failed_total = rowSums(across(where(is.numeric)))) %>%
  
  arrange(desc(n_failed_total))
write.csv(fail_summary, "failed_questions.csv")

# add a correct column, if answer is right
df_parsed$correct <- ifelse(df_parsed$correctanswer == df_parsed$answer_value, 1, 0)

#analyze control questions (exclusion criteria: <50% correct)
df_control <- df_parsed %>%
  filter(questiontype =="control") %>%
  group_by(workerid) %>%
  summarise(mean_correct_control = mean(correct, na.rm = TRUE), .groups = "drop")
# print the participants who failed to pass the minimum threshold
low_accuracy <- df_control%>%
  filter(mean_correct_control<0.5)
print(low_accuracy)


#correct answers
df_summary <- df_parsed %>%
  group_by(workerid, questiontype) %>%
  summarise(num_correct = sum(correct, na.rm = TRUE), mean_rt = mean(rt))

write.csv(df_summary, "nathan_scores_by_question_type.csv")

#all ToM questions
full_score <- df_parsed %>%
  filter(!questiontype %in% c("control")) %>%
  group_by(workerid) %>%
  summarise(nathan_score_full = mean(correct, na.rm = TRUE),mean_rt_nathan_full = mean(rt, na.rm = TRUE), .groups = "drop")
  

# Cognitive ToM only
ctom_score <- df_parsed %>%
  filter(!questiontype %in% c("control","emotion")) %>%
  group_by(workerid) %>%
  summarise(nathan_score_ctom = mean(correct, na.rm = TRUE),mean_rt_nathan_ctom = mean(rt, na.rm = TRUE), .groups = "drop")


df <- full_score %>%
  left_join(ctom_score, by = "workerid")
write.csv(df, "nathan_score_cog_aff.csv")

