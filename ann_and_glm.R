# ==============================================================================
# ML1 Telco Customer Analysis - Neural Network & Poisson GLM
# ==============================================================================

# Load required libraries
library(dplyr)
library(ggplot2)
library(neuralnet)
library(caret)
library(corrplot)
library(VIM)
library(faraway)
library(tidyr)

set.seed(123)

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

# Load the cleaned telco dataset
source("clean_telco.R")

# Load the data (assuming 'telco' is the cleaned dataset)
tr_clean <- telco

# Display dataset structure
cat("Dataset Structure:\n")
str(tr_clean)
cat("\nDataset dimensions:", nrow(tr_clean), "rows,", ncol(tr_clean), "columns\n")

# ==============================================================================
# FEATURE ENGINEERING AND PREPARATION
# ==============================================================================

# Categorical variables (exclude Customer.ID and target variables)
categorical_vars <- c("Contract", "Gender", "Internet.Service", "Internet.Type", 
                      "Married", "Online.Backup", "Online.Security", "Paperless.Billing",
                      "Partner", "Payment.Method", "Phone.Service", "Premium.Tech.Support",
                      "Referred.a.Friend", "Satisfaction.Score", "Senior.Citizen",
                      "Streaming.Movies", "Streaming.Music", "Streaming.TV", 
                      "Under.30", "Unlimited.Data", "Dependents", "Device.Protection.Plan")

# Converting categorical variables to factors
for(var in categorical_vars) {
  if(var %in% names(tr_clean)) {
    tr_clean[[var]] <- as.factor(tr_clean[[var]])
  }
}

# Convert Churn to factor for classification
tr_clean$Churn <- as.factor(tr_clean$Churn)

# Remove Customer.ID for modeling
tr_clean <- tr_clean %>% select(-Customer.ID)

# ==============================================================================
# EXPLORATORY DATA ANALYSIS
# ==============================================================================

cat("\n", strrep("=", 80), "\n")
cat("EXPLORATORY DATA ANALYSIS\n")
cat(strrep("=", 80), "\n")

# Summary of key variables
cat("Summary of Number of Referrals (target for Poisson GLM):\n")
print(summary(tr_clean$Number.of.Referrals))

cat("\nChurn distribution:\n")
print(table(tr_clean$Churn))

# Visualize key relationships
p1 <- ggplot(tr_clean, aes(x = Number.of.Referrals)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Number of Referrals",
       x = "Number of Referrals", y = "Frequency") +
  theme_minimal()

print(p1)
# Save the plot as a PNG file
ggsave("Plots/number_referrals_distribution.png", plot = p1, width = 8, height = 6, dpi = 300)

# Churn by key variables
p2 <- ggplot(tr_clean, aes(x = Tenure.in.Months, fill = Churn)) +
  geom_histogram(position = "identity", alpha = 0.7, bins = 30) +
  labs(title = "Tenure Distribution by Churn Status",
       x = "Tenure (Months)", y = "Count") +
  theme_minimal()

print(p2)
# Save the plot as a PNG file
ggsave("Plots/tenure_churn_distribution.png", plot = p2, width = 8, height = 6, dpi = 300)


# ==============================================================================
# NEURAL NETWORK ANALYSIS - CHURN PREDICTION
# ==============================================================================

cat("\n", strrep("=", 80), "\n")
cat("NEURAL NETWORK ANALYSIS - CHURN PREDICTION\n")
cat(strrep("=", 80), "\n")

# Prepare data for neural network
create_nn_data_selective <- function(data) {
  
  demographic_features <- c("Age", "Gender", "Senior.Citizen", "Partner", "Married")
  service_features <- c("Tenure.in.Months", "Contract", "Internet.Service", "Online.Security", 
                        "Phone.Service", "Payment.Method")
  financial_features <- c("Monthly.Charge", "Total.Charges")
  satisfaction_features <- c("Satisfaction.Score", "Number.of.Referrals")
  
  key_features <- c(
    "Tenure.in.Months",           
    "Monthly.Charge",             
    "Contract",                   
    "Payment.Method",             
    "Internet.Service",           
    "Satisfaction.Score",         
    
    "Age",                        
    "Senior.Citizen",             
    "Partner",                    
    
    "Online.Security",            
    "Total.Charges",              
    "Number.of.Referrals",        
    
    "Phone.Service",              
    "Gender"                      
  )
  
  # Filter available features
  available_features <- key_features[key_features %in% names(data)]
  
  cat("Selected", length(available_features), "features based on business relevance:\n")
  for(i in 1:length(available_features)) {
    cat(sprintf("%d. %s\n", i, available_features[i]))
  }
  
  # Create formula for dummy variables
  formula_str <- paste("~", paste(available_features, collapse = " + "), "- 1")
  
  # Create dummy variables for categorical predictors
  dummy_vars <- model.matrix(as.formula(formula_str), data = data)
  
  # Combine with target
  nn_data <- data.frame(
    dummy_vars,
    Churn = as.numeric(as.character(data$Churn))
  )
  
  # Scale all numeric variables to [0,1] for better neural network performance
  numeric_vars <- names(nn_data)[sapply(nn_data, is.numeric)]
  numeric_vars <- numeric_vars[numeric_vars != "Churn"]  # Exclude target
  
  # Create a temporary dataset with selected features + target
  temp_data <- data[c(available_features, "Churn")]
  temp_data$Churn <- as.numeric(as.character(temp_data$Churn))
  
  # Calculate correlations for numeric features only
  numeric_features <- available_features[sapply(temp_data[available_features], is.numeric)]
  
  if(length(numeric_features) > 0) {
    correlations <- sapply(numeric_features, function(var) {
      cor(temp_data[[var]], temp_data$Churn, use = "complete.obs")
    })
    
    cat("Correlations with Churn (for interpretation only):\n")
    correlation_df <- data.frame(
      Feature = names(correlations),
      Correlation = round(correlations, 4)
    )
    print(correlation_df[order(abs(correlation_df$Correlation), decreasing = TRUE), ])
    
  }
  
  for(var in numeric_vars) {
    var_range <- max(nn_data[[var]], na.rm = TRUE) - min(nn_data[[var]], na.rm = TRUE)
    if(var_range > 0) {  # Avoid division by zero
      nn_data[[var]] <- (nn_data[[var]] - min(nn_data[[var]], na.rm = TRUE)) / var_range
    }
  }
  
  return(nn_data)
}

# Create comprehensive neural network dataset (following assignment guidelines)
nn_data <- create_nn_data_selective(tr_clean)

cat("Neural network data dimensions:", nrow(nn_data), "rows,", ncol(nn_data), "columns\n")

# Split data for neural network
train_index <- createDataPartition(nn_data$Churn, p = 0.8, list = FALSE)
nn_train <- nn_data[train_index, ]
nn_test <- nn_data[-train_index, ]

# Create neural network formula using all available features (except target)
all_features <- names(nn_train)[names(nn_train) != "Churn"]

# Limit to maximum 15 features for neural network
if(length(all_features) > 15) {
  priority_features <- c(
    "Tenure.in.Months", "Monthly.Charge", "Total.Charges", "Satisfaction.Score", "Age"
  )
  
  selected_features <- all_features[all_features %in% priority_features]
  remaining_features <- setdiff(all_features, selected_features)
  additional_needed <- min(15 - length(selected_features), length(remaining_features))
  
  final_features <- c(selected_features, head(remaining_features, additional_needed))
} else {
  final_features <- all_features
}

nn_formula <- as.formula(paste("Churn ~", paste(final_features, collapse = " + ")))

cat("\nNeural Network will use", length(final_features), "domain-selected features (max 15)\n")
cat("Features used:", paste(final_features, collapse = ", "), "\n")

# Train multiple neural network architectures
cat("\nTraining Neural Networks with different architectures...\n")

# Architecture 1: Simple network
cat("Training Architecture 1: Single hidden layer (8 neurons)...\n")
nn_model_1 <- neuralnet(
  formula = nn_formula,
  data = nn_train,
  hidden = 8,
  linear.output = FALSE,
  threshold = 0.1,
  stepmax = 1e6,
  rep = 1,
  algorithm = "rprop+"
)

# Architecture 2: Two hidden layers
cat("Training Architecture 2: Two hidden layers (6, 4 neurons)...\n")
nn_model_2 <- neuralnet(
  formula = nn_formula,
  data = nn_train,
  hidden = c(6, 4),
  linear.output = FALSE,
  threshold = 0.1,
  stepmax = 1e6,
  rep = 1,
  algorithm = "rprop+"
)

# Architecture 3: Three hidden layers (more complex)
cat("Training Architecture 3: Three hidden layers (8, 5, 3 neurons)...\n")
nn_model_3 <- try({
  neuralnet(
    formula = nn_formula,
    data = nn_train,
    hidden = c(8, 5, 3),
    linear.output = FALSE,
    threshold = 0.1,
    stepmax = 1e6,
    rep = 1,
    algorithm = "rprop+"
  )
}, silent = TRUE)

# Evaluate all models
evaluate_nn_model <- function(model, test_data, model_name) {
  if(inherits(model, "try-error")) {
    cat(sprintf("%s: Failed to converge\n", model_name))
    return(list(accuracy = NA, sensitivity = NA, specificity = NA))
  }
  
  # Make predictions
  predictions <- predict(model, test_data)
  pred_class <- ifelse(predictions > 0.5, 1, 0)
  
  # Evaluate performance
  pred_factor <- factor(pred_class, levels = c(0, 1))
  actual_factor <- factor(test_data$Churn, levels = c(0, 1))
  
  confusion <- confusionMatrix(pred_factor, actual_factor)
  
  accuracy <- confusion$overall['Accuracy']
  sensitivity <- confusion$byClass['Sensitivity']
  specificity <- confusion$byClass['Specificity']
  
  cat(sprintf("%s Performance:\n", model_name))
  cat(sprintf("  Accuracy: %.4f\n", accuracy))
  cat(sprintf("  Sensitivity: %.4f\n", sensitivity))
  cat(sprintf("  Specificity: %.4f\n", specificity))
  
  return(list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    confusion = confusion,
    predictions = predictions
  ))
}

# Evaluate all models
cat("\n=== MODEL EVALUATION ===\n")
results_1 <- evaluate_nn_model(nn_model_1, nn_test, "Architecture 1 (8)")
results_2 <- evaluate_nn_model(nn_model_2, nn_test, "Architecture 2 (6,4)")
results_3 <- evaluate_nn_model(nn_model_3, nn_test, "Architecture 3 (8,5,3)")

# Select best model
valid_models <- list(
  list(model = nn_model_1, results = results_1, name = "Architecture 1"),
  list(model = nn_model_2, results = results_2, name = "Architecture 2")
)

if(!inherits(nn_model_3, "try-error")) {
  valid_models[[3]] <- list(model = nn_model_3, results = results_3, name = "Architecture 3")
}

# Find best performing model
best_accuracy <- 0
best_model_info <- NULL

for(model_info in valid_models) {
  if(!is.na(model_info$results$accuracy) && model_info$results$accuracy > best_accuracy) {
    best_accuracy <- model_info$results$accuracy
    best_model_info <- model_info
  }
}

cat(sprintf("\nBest performing model: %s (Accuracy: %.4f)\n", 
            best_model_info$name, best_accuracy))

# Use best model for final analysis
final_nn_model <- best_model_info$model
final_results <- best_model_info$results

# Plot the best neural network architecture
cat("\nPlotting best neural network architecture...\n")
plot(final_nn_model, rep = "best")
title(paste("Best Neural Network Architecture:", best_model_info$name))

# Detailed confusion matrix for best model
cat("\nDetailed Confusion Matrix for Best Model:\n")
print(final_results$confusion)

# Store final model performance
nn_final_accuracy <- final_results$accuracy
nn_final_sensitivity <- final_results$sensitivity
nn_final_specificity <- final_results$specificity

cat("\n=== NEURAL NETWORK FINAL SUMMARY ===\n")
cat(sprintf("Best Architecture: %s\n", paste(best_model_info$name, collapse = ", ")))
cat(sprintf("Test Set Accuracy: %.4f\n", nn_final_accuracy))
cat(sprintf("Test Set Sensitivity: %.4f\n", nn_final_sensitivity))
cat(sprintf("Test Set Specificity: %.4f\n", nn_final_specificity))

# ==============================================================================
# POISSON GLM ANALYSIS - NUMBER OF REFERRALS PREDICTION
# ==============================================================================

cat("\n", strrep("=", 80), "\n")
cat("POISSON GLM ANALYSIS - REFERRAL COUNT PREDICTION\n")
cat(strrep("=", 80), "\n")

# Explore the count variable (Number of Referrals)
cat("Distribution of Number of Referrals:\n")
referral_table <- table(tr_clean$Number.of.Referrals)
print(referral_table)

# Check Poisson assumptions
mean_referrals <- mean(tr_clean$Number.of.Referrals, na.rm = TRUE)
var_referrals <- var(tr_clean$Number.of.Referrals, na.rm = TRUE)

cat(sprintf("\nMean of Number.of.Referrals: %.4f\n", mean_referrals))
cat(sprintf("Variance of Number.of.Referrals: %.4f\n", var_referrals))
cat(sprintf("Variance/Mean Ratio: %.4f\n", var_referrals/mean_referrals))

if(var_referrals/mean_referrals > 1.5) {
  cat("Note: Variance > Mean suggests potential overdispersion.\n")
}

# Visualize referrals distribution
p3 <- ggplot(tr_clean, aes(x = Number.of.Referrals)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Number of Referrals",
       x = "Number of Referrals", y = "Frequency") +
  theme_minimal()

print(p3)

# Split data for Poisson GLM
train_index_poisson <- createDataPartition(tr_clean$Number.of.Referrals, p = 0.8, list = FALSE)
poisson_train <- tr_clean[train_index_poisson, ]
poisson_test <- tr_clean[-train_index_poisson, ]

# Select predictors for Poisson model based on domain knowledge
poisson_predictors <- c("Age", "Tenure.in.Months", "Monthly.Charge", 
                        "Contract", "Satisfaction.Score", "Churn",
                        "Internet.Service", "Payment.Method")

# Filter available predictors
available_predictors <- poisson_predictors[poisson_predictors %in% names(poisson_train)]

# Create Poisson formula
poisson_formula <- as.formula(paste("Number.of.Referrals ~", 
                                    paste(available_predictors, collapse = " + ")))

cat("Poisson GLM Formula:\n")
print(poisson_formula)

# Fit Poisson GLM
cat("\nFitting Poisson GLM...\n")
poisson_model <- glm(poisson_formula, 
                     data = poisson_train, 
                     family = "poisson")

# Model summary
cat("\nPoisson GLM Summary:\n")
print(summary(poisson_model))

# Check for overdispersion
residual_deviance <- poisson_model$deviance
df_residual <- poisson_model$df.residual
overdispersion_ratio <- residual_deviance / df_residual

cat(sprintf("\nOverdispersion Check:\n"))
cat(sprintf("Residual Deviance: %.4f\n", residual_deviance))
cat(sprintf("Degrees of Freedom: %d\n", df_residual))
cat(sprintf("Overdispersion Ratio: %.4f\n", overdispersion_ratio))

# Fit quasi-Poisson if overdispersed
if(overdispersion_ratio > 1.2) {
  cat("\nFitting Quasi-Poisson GLM due to overdispersion...\n")
  
  quasi_poisson_model <- glm(poisson_formula, 
                             data = poisson_train, 
                             family = "quasipoisson")
  
  cat("\nQuasi-Poisson GLM Summary:\n")
  print(summary(quasi_poisson_model))
  
  final_poisson_model <- quasi_poisson_model
} else {
  final_poisson_model <- poisson_model
}

# Make predictions
poisson_predictions <- predict(final_poisson_model, 
                               newdata = poisson_test, 
                               type = "response")

# Round predictions to integers
poisson_pred_rounded <- round(poisson_predictions)

# Evaluate performance
poisson_rmse <- sqrt(mean((poisson_test$Number.of.Referrals - poisson_predictions)^2))
poisson_mae <- mean(abs(poisson_test$Number.of.Referrals - poisson_predictions))

cat("\nPoisson GLM Performance Metrics:\n")
cat(sprintf("RMSE: %.4f\n", poisson_rmse))
cat(sprintf("MAE: %.4f\n", poisson_mae))

# Prediction accuracy table
poisson_confusion_table <- table(Actual = poisson_test$Number.of.Referrals, 
                                 Predicted = poisson_pred_rounded)
cat("\nPoisson GLM - Count Prediction Table (top 10x10):\n")
print(poisson_confusion_table[1:min(10, nrow(poisson_confusion_table)), 
                              1:min(10, ncol(poisson_confusion_table))])

# Coefficient interpretation
cat("\nCoefficient Interpretation (Rate Ratios):\n")
coef_exp <- exp(coef(final_poisson_model))
for(i in 1:length(coef_exp)) {
  if(names(coef_exp)[i] != "(Intercept)") {
    cat(sprintf("%s: %.4f (%.1f%% change in rate)\n", 
                names(coef_exp)[i], 
                coef_exp[i], 
                (coef_exp[i] - 1) * 100))
  }
}

# ==============================================================================
# MODEL VALIDATION AND CROSS-VALIDATION
# ==============================================================================

cat("\n", strrep("=", 80), "\n")
cat("MODEL VALIDATION AND CROSS-VALIDATION\n")
cat(strrep("=", 80), "\n")

# Cross-validation for Neural Network
set.seed(123)
cv_folds <- 5

cat("Performing Cross-Validation for Neural Network...\n")
nn_cv_accuracy <- numeric(cv_folds)
folds <- createFolds(nn_data$Churn, k = cv_folds)

for(i in 1:cv_folds) {
  train_cv <- nn_data[-folds[[i]], ]
  test_cv <- nn_data[folds[[i]], ]
  
  # Simplified neural network for CV
  nn_cv <- try({
    neuralnet(nn_formula, 
              data = train_cv, 
              hidden = c(3), 
              linear.output = FALSE,
              threshold = 0.1,
              stepmax = 1e5)
  }, silent = TRUE)
  
  if(!inherits(nn_cv, "try-error")) {
    pred_cv <- predict(nn_cv, test_cv)
    pred_class_cv <- ifelse(pred_cv > 0.5, 1, 0)
    nn_cv_accuracy[i] <- mean(pred_class_cv == test_cv$Churn, na.rm = TRUE)
  } else {
    nn_cv_accuracy[i] <- NA
  }
}

cat(sprintf("Neural Network CV Accuracy: %.4f ± %.4f\n", 
            mean(nn_cv_accuracy, na.rm = TRUE), 
            sd(nn_cv_accuracy, na.rm = TRUE)))

# Cross-validation for Poisson GLM
cat("Performing Cross-Validation for Poisson GLM...\n")
poisson_cv_rmse <- numeric(cv_folds)
folds_poisson <- createFolds(tr_clean$Number.of.Referrals, k = cv_folds)

for(i in 1:cv_folds) {
  train_cv_p <- tr_clean[-folds_poisson[[i]], ]
  test_cv_p <- tr_clean[folds_poisson[[i]], ]
  
  model_cv_p <- glm(poisson_formula, 
                    data = train_cv_p, 
                    family = if(overdispersion_ratio > 1.2) "quasipoisson" else "poisson")
  
  pred_cv_p <- predict(model_cv_p, newdata = test_cv_p, type = "response")
  poisson_cv_rmse[i] <- sqrt(mean((test_cv_p$Number.of.Referrals - pred_cv_p)^2))
}

cat(sprintf("Poisson GLM CV RMSE: %.4f ± %.4f\n", 
            mean(poisson_cv_rmse), 
            sd(poisson_cv_rmse)))

# ==============================================================================
# VISUALIZATION AND DIAGNOSTICS
# ==============================================================================

cat("\n", strrep("=", 80), "\n")
cat("MODEL DIAGNOSTICS AND VISUALIZATION\n")
cat(strrep("=", 80), "\n")

# Poisson GLM diagnostic plots
par(mfrow = c(2, 2))
plot(final_poisson_model, main = "Poisson GLM Diagnostics")
par(mfrow = c(1, 1))

# Feature importance visualization for Poisson GLM
coef_df <- data.frame(
  Variable = names(coef(final_poisson_model))[-1],
  Coefficient = coef(final_poisson_model)[-1],
  Rate_Ratio = exp(coef(final_poisson_model)[-1])
)

p4 <- ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Poisson GLM Coefficients",
       x = "Variables", y = "Coefficient Value") +
  theme_minimal()

print(p4)
