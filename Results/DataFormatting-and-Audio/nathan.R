library(jsonlite)
library(dplyr)
library(stringr) 
library(purrr) 
library(tidyr)

getwd()
currentDir = "/home/yzhang/Documents/Uni-Saarland/thesis/Results/DataFormatting-and-Audio"
setwd(currentDir)

data_raw <- read.csv("./dataRaw/experiment_6301_results.csv")

data <- data_raw %>%
    filter(!workerid %in% c("test", "1", "1234", "testFirefox", "testURL", "testFinal"))


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

df_parsed <- df_parsed %>%
  mutate(
    answer_value = map(answer_list, extract("answer", NA_character_)),
    rt           = map_dbl(answer_list, extract("RT", NA_real_)),
    timeout      = map_dbl(answer_list, extract("timeout", NA_real_)),
    feedback     = map(answer_list, extract("feedback", 0))
  )


################################################
# PARTICIPANT FEEDBACK

df_feedback <- df_parsed %>%
  filter(map_lgl(feedback, ~ !is.null(.)))
feedback <- df_feedback %>%
  dplyr::select(workerid, feedback)
feedback <- feedback %>%
  unnest_wider(feedback, names_sep = "_")

write.csv(feedback, "./dataOutput/ToM/nathan2_feedback.csv")

################################################
# TECHNICAL PROBLEMS WITH VIDEO

fail_summary <- df_parsed %>%
  filter(answer_value == "-1")

fail_summary <- df_parsed %>%
  filter(answer_value == "-1") %>%
  
  group_by(workerid, questiontype) %>%
  summarise(n_failed_type = n(), .groups = "drop") %>%
  
  pivot_wider(
    names_from = questiontype,
    values_from = n_failed_type,
    values_fill = 0
  ) %>%
  
  mutate(n_failed_total = rowSums(across(where(is.numeric)))) %>%
  
  arrange(desc(n_failed_total))
write.csv(fail_summary, "./dataOutput/ToM/failed_questions.csv")

################################################
# CONTROL CHECK

# add a correct column, if answer is right
df_parsed$correct <- ifelse(df_parsed$correctanswer == df_parsed$answer_value, 1, 0)

# analyze control questions (exclusion criteria: <80% correct)
df_control <- df_parsed %>%
  filter(questiontype =="control") %>%
  group_by(workerid) %>%
  summarise(mean_correct_control = mean(correct, na.rm = TRUE), .groups = "drop")
# print the participants who failed to pass the minimum threshold
low_accuracy <- df_control%>%
  filter(mean_correct_control<0.8)
print(low_accuracy$workerid)

################################################
# FINAL OUTPUT REPORTS

# correct answers by question type
df_summary <- df_parsed %>%
  group_by(workerid, questiontype) %>%
  summarise(num_correct = sum(correct, na.rm = TRUE), mean_rt = mean(rt))

write.csv(df_summary, "./dataOutput/ToM/nathan_scores_by_question_type.csv")

# score for all critical ToM questions
full_score <- df_parsed %>%
  filter(!questiontype %in% c("control")) %>%
  # filter(!workerid %in% low_accuracy$workerid)
  group_by(workerid) %>%
  summarise(nathan_score_full = mean(correct, na.rm = TRUE),mean_rt_nathan_full = mean(rt, na.rm = TRUE), .groups = "drop")

write.csv(full_score, "./dataOutput/ToM/nathan_scores.csv")  
