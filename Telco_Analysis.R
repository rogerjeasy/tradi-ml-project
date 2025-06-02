source("clean_telco.R")
if (!require('caTools')) install.packages('caTools', dependencies=TRUE)
if (!require('e1071')) install.packages('e1071', dependencies=TRUE)
if (!require('dplyr')) install.packages('dplyr', dependencies=TRUE)
if (!require('rcompanion')) install.packages('rcompanion', dependencies=TRUE)
if (!require('tidyr')) install.packages('tidyr', dependencies=TRUE)
if (!require('effectsize')) install.packages('effectsize', dependencies=TRUE)
if (!require('mgcv')) install.packages('mgcv', dependencies=TRUE)
library(tidyr)
library(effectsize)
library(rcompanion)
library(caTools)
library(e1071)
library(dplyr)
library(ggplot2)
library(mgcv) 

View(head(telco))  
str(telco)


# ===== Feature Selection: Churn Analysis =====

factor_vars <- c(
  "Married",
  "Gender",
  "Multiple.Lines",
  "Device.Protection.Plan",
  "Internet.Service",
  "Online.Backup",
  "Online.Security",
  "Paperless.Billing",
  "Partner",
  "Phone.Service",
  "Premium.Tech.Support",
  "Referred.a.Friend",
  "Senior.Citizen",
  "Streaming.Movies",
  "Streaming.Music",
  "Streaming.TV",
  "Unlimited.Data",
  "age_group",
  "Dependents",
  "Internet.Type",
  "Payment.Method",
  "Satisfaction.Score",
  "Under.30"
)

results <- data.frame(
  variable  = factor_vars,
  statistic = NA_real_,
  df        = NA_real_,
  p_value   = NA_real_
)


# Initialize results table
results <- data.frame(
  variable   = factor_vars,
  statistic  = numeric(length(factor_vars)),
  df         = numeric(length(factor_vars)),
  p_value    = numeric(length(factor_vars)),
  cramers_v  = numeric(length(factor_vars)),
  stringsAsFactors = FALSE
)

# Loop over variables
for (i in seq_along(factor_vars)) {
  var <- factor_vars[i]
  tbl <- table(telco[[var]], telco$Churn)
  test <- suppressWarnings(chisq.test(tbl, correct = FALSE))
  v <- suppressWarnings(cramerV(tbl))
  
  results$statistic[i] <- unname(test$statistic)
  results$df[i]        <- unname(test$parameter)
  results$p_value[i]   <- test$p.value
  results$cramers_v[i] <- v
}

# Filter and sort
results <- results %>%
  filter(p_value < 0.05) %>%
  arrange(desc(cramers_v))

results

# Interior correlation

in_results <- data.frame(
  variable    = character(),
  variable_2  = character(),
  statistic   = numeric(),
  df          = numeric(),
  p_value     = numeric(),
  cramers_v   = numeric(),
  stringsAsFactors = FALSE
)

f_interior_correlation <- function(data, factor_vars) {
  library(rcompanion)
  library(dplyr)
  
  in_results <- data.frame(
    variable    = character(),
    variable_2  = character(),
    statistic   = numeric(),
    df          = numeric(),
    p_value     = numeric(),
    cramers_v   = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(factor_vars)) {
    for (j in seq_along(factor_vars)) {
      var1 <- factor_vars[i]
      var2 <- factor_vars[j]
      
      if (var1 == var2) next  # skip identical pairs
      
      tbl <- table(data[[var1]], data[[var2]])
      test <- suppressWarnings(chisq.test(tbl, correct = FALSE))
      v <- suppressWarnings(cramerV(tbl))
      
      in_results <- rbind(in_results, data.frame(
        variable    = var1,
        variable_2  = var2,
        statistic   = unname(test$statistic),
        df          = unname(test$parameter),
        p_value     = test$p.value,
        cramers_v   = v,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  rownames(in_results) <- NULL
  
  # Apply filter and sort
  in_results <- in_results %>%
    filter(p_value < 0.05) %>%
    arrange(desc(cramers_v))
  
  return(in_results)
}

result_table <- f_interior_correlation(telco, factor_vars)
View(result_table %>% 
       filter(cramers_v > 0.4))

in_results <- f_interior_correlation(telco, factor_vars)
View(in_results %>% 
       filter(cramers_v > 0.4))

# Print selected relevant columns
View(in_results %>% 
       select(variable, variable_2, statistic, df, p_value, cramers_v) %>% 
       filter(p_value < 0.05) %>% 
       filter(cramers_v > 0.4)) 
#Table shows that there are values which are intrinsically linked due to high cramers V and p-value and should be dropped

model_factor_vars <- c(
  "Device.Protection.Plan",
  "Online.Backup",
  "Online.Security",
  "Paperless.Billing",
  "Partner",
  "Premium.Tech.Support",
  "Senior.Citizen",
  "Streaming.Movies",
  "Dependents",
  "Internet.Type",
  "Payment.Method",
  "Satisfaction.Score",
  "Under.30"
)


View(f_interior_correlation(telco, model_factor_vars)) #final table looks good


continuous_vars <- c(
  "Avg.Monthly.GB.Download",
  "Avg.Monthly.Long.Distance.Charges",
  "CLTV",
  "Monthly.Charge",
  "Total.Charges",
  "Total.Long.Distance.Charges",
  "Total.Revenue"
)


pb_results <- lapply(continuous_vars, function(var) {
  test <- cor.test(telco[[var]], as.numeric(as.character(telco$Churn)), method = "pearson")
  c(correlation = test$estimate, p_value = test$p.value)
})

pb_cor_matrix <- do.call(rbind, pb_results)
pb_cor_matrix <- data.frame(Variable = continuous_vars, pb_cor_matrix)
pb_cor_matrix

#Internal correlation
cor_matrix <- cor(telco[ , continuous_vars], use = "complete.obs")
cor_df <- as.data.frame(cor_matrix)
cor_df
View(cor_df)

keep_cont_vars <- c(
  "Total.Revenue",
  "Monthly.Charge",
  "CLTV",
  "Avg.Monthly.GB.Download"  # optional
)
# Set up grid
par(mfrow = c(3, 4))  

for (var in continuous_vars) {
  boxplot(telco[[var]] ~ telco$Churn,
          main = var,
          xlab = "Churn",
          ylab = var)
}

count_vars <- c(
  "Age",
  "Number.of.Dependents",
  "Number.of.Referrals",
  "Tenure.in.Months",
  "Total.Extra.Data.Charges"
)

pb_count_results <- lapply(count_vars, function(var) {
  test <- cor.test(telco[[var]], as.numeric(as.character(telco$Churn)), method = "pearson")
  c(correlation = test$estimate, p_value = test$p.value)
})

pb_count_matrix <- do.call(rbind, pb_count_results)
pb_count_matrix <- data.frame(Variable = count_vars, pb_count_matrix)

pb_count_matrix

count_data <- telco[ , count_vars]
count_cor_matrix <- cor(count_data, use = "complete.obs", method = "pearson")
count_cor_matrix
count_cor_df <- as.data.frame(count_cor_matrix)
View(count_cor_df)  


# ===== GLM Model: Categorical =====

# Assuming Churn is a factor with values like "Yes"/"No"
telco %>%
  select(Churn, Satisfaction.Score) %>%
  ggplot(aes(x = as.factor(Satisfaction.Score), fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(x = "Satisfaction Score", y = "Count", title = "Churn by Satisfaction Score") +
  theme_minimal()
#comparison of models showed that satisfaction score is an absolute must for feasibility
categorical_glm <- glm(
  Churn ~ Device.Protection.Plan +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Partner +
    Premium.Tech.Support +
    Senior.Citizen +
    Streaming.Movies +
    Dependents +
    Internet.Type +
    Payment.Method +
    Satisfaction.Score +
    Under.30,
  data = telco,
  family = binomial
)

summary(categorical_glm)
categorical_glm_cleaned <- glm(
  Churn ~ Device.Protection.Plan +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Internet.Type +
    Satisfaction.Score,
  data = telco,
  family = binomial
)
summary(categorical_glm_cleaned)


AIC(categorical_glm_cleaned, categorical_glm)
cat_glm_vars <- c(
  "Device.Protection.Plan",
  "Online.Backup",
  "Online.Security",
  "Paperless.Billing",
  "Premium.Tech.Support",
  "Dependents",
  "Internet.Type",
  "Satisfaction.Score"
)


summary(categorical_glm_cleaned)
AIC(categorical_glm, categorical_glm_cleaned) #cleaned model has same/minimally better performance with 6 fewer parameters
anova(categorical_glm, categorical_glm_cleaned, test = "Chisq")

# ===== Feature Selection 2 =====

# Compute η² for each categorical–continuous pairing with descriptive names
effect_size_tbl <- expand.grid(
  categorical_var = cat_glm_vars,
  continuous_var  = keep_cont_vars,
  stringsAsFactors = FALSE
)
effect_size_tbl$eta_squared_value <- NA_real_

for (idx in seq_len(nrow(effect_size_tbl))) {
  cat_name <- effect_size_tbl$categorical_var[idx]
  cont_name <- effect_size_tbl$continuous_var[idx]
  
  anova_fit <- aov(
    formula = as.formula(paste(cont_name, "~", cat_name)),
    data    = telco
  )
  es <- eta_squared(anova_fit, partial = FALSE)
  effect_size_tbl$eta_squared_value[idx] <- es$Eta2[1]
}

effect_size_tbl <- effect_size_tbl %>%
  mutate(
    eta_squared_pct = round(eta_squared_value * 100, 2)
  ) %>%
  arrange(desc(eta_squared_pct))

print(effect_size_tbl)
View(effect_size_tbl)
# Drop internet_type

# ==== GLM combined =====

cat_con_glm <- glm(
  Churn ~ Device.Protection.Plan +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Satisfaction.Score +
    Total.Revenue +
    Monthly.Charge +
    CLTV +
    Avg.Monthly.GB.Download,
  data = telco,
  family = binomial)

summary(cat_con_glm)

AIC(cat_con_glm, categorical_glm_cleaned) #shows new one is better

cat_con_glm_cleaned <- glm(
  Churn ~ 
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Satisfaction.Score +
    Total.Revenue +
    Monthly.Charge,
  data = telco,
  family = binomial)

AIC(cat_con_glm_cleaned, cat_con_glm) 

# ===== GLM Model: Count =====


complete_glm <- glm(
  Churn ~ Number.of.Referrals +
    Tenure.in.Months +
    Age +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Satisfaction.Score +
    Total.Revenue +
    Monthly.Charge,
  data = telco,
  family = binomial)

plot(telco$Total.Revenue, telco$Monthly.Charge,
     xlab = "Total Revenue",
     ylab = "Monthly Charge",
     main = "Monthly Charge vs Total Revenue",
     pch = 16, col = telco$Churn)

summary(complete_glm)
library(car)
vif(complete_glm)
detach("package:car", unload = TRUE)

boxplot(Age ~ Churn, data = telco, col = "grey")

telco %>% 
  filter(Number.of.Dependents < 2) %>% 
  boxplot(Number.of.Dependents ~ Churn, data=, col="red")

telco %>% 
  filter(Age < 20) %>% 
  select(Age, Churn) %>% 
  View()

AIC(cat_con_glm_cleaned, complete_glm) 


TotalRev_glm <- glm(
  Churn ~ Number.of.Referrals +
    Age +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Satisfaction.Score +
    Total.Revenue,
  data = telco,
  family = binomial)

comb_glm <- glm(
  Churn ~ Number.of.Referrals +
    Tenure.in.Months +
    Age +
    Online.Backup +
    Online.Security +
    Paperless.Billing +
    Premium.Tech.Support +
    Dependents +
    Satisfaction.Score +
    Monthly.Charge,
  data = telco,
  family = binomial)

AIC(comb_glm, TotalRev_glm)

# ===== Testing: Final GLM Model =====

telco_test <- read.csv("test.csv")
telco_test <- telco_test %>%
  mutate(
    Churn = as.factor(Churn),
    Married = as.factor(Married),
    Gender = as.factor(Gender),
    Multiiple.Lines = as.factor(Multiple.Lines),
    Device.Protection.Plan = as.factor(Device.Protection.Plan),
    Internet.Service = as.factor(Internet.Service),
    Online.Backup = as.factor(Online.Backup),
    Online.Security = as.factor(Online.Security),
    Paperless.Billing = as.factor(Paperless.Billing),
    Partner = as.factor(Partner),
    Phone.Service = as.factor(Phone.Service),
    Premium.Tech.Support = as.factor(Premium.Tech.Support),
    Referred.a.Friend = as.factor(Referred.a.Friend),
    Senior.Citizen = as.factor(Senior.Citizen),
    Streaming.Movies = as.factor(Streaming.Movies),
    Streaming.Music = as.factor(Streaming.Music),
    Streaming.TV = as.factor(Streaming.TV),
    Unlimited.Data = as.factor(Unlimited.Data),
    Dependents = as.factor(Dependents),
    Internet.Type = as.factor(Internet.Type),
    Payment.Method = as.factor(Payment.Method),
    Satisfaction.Score = as.factor(Satisfaction.Score),
    Under.30 = as.factor(Under.30),
  )

# Predicted probabilities
test_prob <- predict(comb_glm, newdata = telco_test, type = "response")

# Class predictions using 0.5 cut-off
test_pred <- ifelse(test_prob >= 0.5, 1, 0)

cm <- table(Actual = telco_test$Churn,
            Predicted = test_pred)

print(cm)

library(pROC)

# test_prob = predicted probabilities
# telco_test$Churn = actual labels (0 or 1)
roc_obj <- roc(telco_test$Churn, test_prob)
plot(roc_obj, col = "blue")
auc(roc_obj)   # prints AUC value

detach("package:pROC", unload = TRUE)


# ===== Evaluation: GAM =====

# log-odds transform churn so patterns pop out (optional)
library(ggplot2)
plot_smooth <- function(var) {
  ggplot(telco, aes_string(x = var, y = "as.numeric(as.character(Churn))")) +
    geom_jitter(height = 0.05, alpha = 0.2) +
    geom_smooth(method = "loess", se = FALSE) +
    labs(y = "Churn", x = var)
}
plot_smooth("Monthly.Charge")
plot_smooth("Tenure.in.Months")
plot_smooth("Age")
plot_smooth("Number.of.Referrals")

plot_bar <- function(var) {
  ggplot(telco, aes(x = var, fill = as.factor(Churn))) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", fill = "Churn")
}
plot_bar("Online.Backup")
plot_bar("Online.Security")
plot_bar("Paperless.Billing")
plot_bar("Premium.Tech.Support")

telco %>% count(Online.Backup, Churn)

gam_fit <- gam(Churn ~
  s(Monthly.Charge, k = 10) +
  s(Tenure.in.Months, k = 10) +
  s(Age, k = 10) +
  s(Number.of.Referrals, k =  8) +
  Online.Backup +
  Online.Security +
  Paperless.Billing +
  Premium.Tech.Support +
  Dependents +
  Satisfaction.Score,
  data   = telco,
  family = binomial,   
  method = "REML"
)
concurvity(gam_fit)

gam_test_prob <- predict(gam_fit, telco_test, type = "response")
gam_test_pred <- ifelse(gam_test_prob >= 0.50, 1, 0)

gam_conf_mat  <- table(Actual = telco_test$Churn, Predicted = test_pred)
print(gam_conf_mat)

