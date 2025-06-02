# Linear Model ####################################################################
# Description:
# This script builds and evaluates multiple linear regression models to predict 
# Customer Lifetime Value (CLTV), a continuous numeric variable. The analysis 
# follows these steps:
# 1. Data Processing:
#    - Load the dataset and inspect its structure.
# 2. Data Preparation:
#    - Convert categorical variables (Contract, Payment Method, Internet Type) 
#      into factors to ensure proper encoding in the regression model.
# 3. Initial Model Fitting:
#    - Fit a multiple linear regression model using selected predictors based on 
#      domain knowledge, including customer demographics, usage behavior, and service types.
# 4. Multicollinearity Check:
#    - Use Variance Inflation Factor (VIF) to assess and identify highly correlated predictors.
# 5. Non-linearity Exploration:
#    - Introduce a 4th-degree polynomial transformation of "Tenure in Months" to 
#      capture potential non-linear effects.
#    - Visualize the polynomial relationship between CLTV and tenure using ggplot2.
# 6. Model Selection:
#    - Apply stepwise model selection (both directions) on the initial and non-linear models 
#      to improve model parsimony.
# 7. Model Diagnostics:
#    - Evaluate residual plots, histograms, Q-Q plots, and Cook’s distance 
#      to check model assumptions, normality, and influential data points.

# 1. Data Processing
tr_data <- read.csv("train.csv", stringsAsFactors = FALSE)
str(tr_data)
head(tr_data)
tr_clean <- tr_data
# 2. Convert categorical variables into factors
tr_clean$Contract <- as.factor(tr_clean$Contract)
tr_clean$Payment.Method <- as.factor(tr_clean$Payment.Method)
tr_clean$Internet.Type <- as.factor(tr_clean$Internet.Type)
# 3. Fit the linear model
initial_model <- lm(CLTV ~ Age + Avg.Monthly.GB.Download + Avg.Monthly.Long.Distance.Charges + 
                      Churn.Score + Monthly.Charge + Tenure.in.Months + Total.Charges + 
                      Total.Long.Distance.Charges + Under.30 + Unlimited.Data + 
                      Device.Protection.Plan + Contract + Payment.Method + Internet.Type, 
                    data = tr_clean)
summary(initial_model)

# 4. Check for multicollinearity
library(car)
vif(initial_model)

# 5. Non-linearity exploration

model_nonlinear <-lm(CLTV ~ Age + poly(Tenure.in.Months, 4) + 
                      Avg.Monthly.GB.Download + Total.Charges + 
                      Total.Long.Distance.Charges + Under.30 + Unlimited.Data + 
                      Device.Protection.Plan + Contract, data = tr_clean)
summary(model_nonlinear)
vif(model_nonlinear)

# visualization for CLTV against Tenure.in.Months
library(ggplot2)
ggplot(tr_clean, aes(x = Tenure.in.Months, y = CLTV)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), color = "red") +
  labs(title = "Effect of Tenure on CLTV (Polynomial)", x = "Tenure in Months", y = "CLTV")


# 6.  Model Selection:
# Stepwise model selection (both directions)
step_model <- step(initial_model, direction = "both")
summary(step_model)
# stepwise selection on the non-linear model
step_model_nonlinear <- step(model_nonlinear, direction = "both")
summary(step_model_nonlinear)
# 7. Model Diagnostics

plot(model_nonlinear$residuals)
hist(model_nonlinear$residuals)
qqnorm(model_nonlinear$residuals)
qqline(model_nonlinear$residuals)
plot(model_nonlinear, which = 4)  
plot(model_nonlinear, which = 5)
################################################################################
# Remarks: model_nonlinear performs far better than the initial model
# as it has adjusted R2 = 0.20, while the initial model has adjusted R2 = 0.164


################################################################################
#svm

# 1. Libraries and Data Pre-processing

library(dplyr)
library(caret)
library(e1071)
install.packages("glmnet")
library(glmnet)
library(reshape2)

tr_clean$Churn <- as.factor(tr_clean$Churn)  # Ensure binary factor
tr_clean <- tr_clean[complete.cases(tr_clean), ]  # Remove NAs
numeric_data <- tr_clean[sapply(tr_clean, is.numeric)]
# 2. Correlation Matrix and Feature Selection

cor_matrix <- cor(numeric_data)
high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)
numeric_data_reduced <- numeric_data[, -high_cor]

# display correlation

# 3.Correlation Matrix Transformation and Sorting
melted_cor_matrix <- melt(cor_matrix, varnames = c("Variable1", "Variable2"))
melted_cor_matrix_unique <- melted_cor_matrix[as.character(melted_cor_matrix$Variable1) < as.character(melted_cor_matrix$Variable2), ]
sorted_cor_list <- melted_cor_matrix_unique[order(-abs(melted_cor_matrix_unique$value)), ]
print("\nCorrelation List (Sorted by Absolute Value):")
print(sorted_cor_list)



# 4. Preparing Data for Modeling
X <- as.matrix(tr_data[, -which(names(tr_data) == "Churn")])  # Remove target column (Churn)
y <- tr_data$Churn  # Target variable (Churn)

# 5. Fitting Lasso Regression Model
cv_lasso <- cv.glmnet(X, y, alpha = 1)
best_lambda <- cv_lasso$lambda.min
print(paste("Best lambda:", best_lambda))

# 6. Plotting and Interpreting Lasso Results
plot(cv_lasso)
lasso_coefficients <- coef(cv_lasso, s = "lambda.min")
print(lasso_coefficients)

# 7. Feature Selection Using Lasso and Elastic Net Regularization
lasso_coefficients_matrix <- as.matrix(lasso_coefficients)  # Convert sparse matrix to a regular matrix
selected_features <- rownames(lasso_coefficients_matrix)[lasso_coefficients_matrix != 0]  # Get the non-zero coefficients
print("Selected features:")
print(selected_features)

# Elastic Net
elastic_net <- cv.glmnet(X, y, alpha = 0.5)  # alpha = 0.5 gives an equal balance between Lasso and Ridge
best_lambda <- elastic_net$lambda.min
coef(elastic_net, s = "lambda.min")


# 8. Select the relevant columns based on the refined feature set
selected_features <- c("Churn", "Churn.Score", "Dependents", "Device.Protection.Plan", 
                       "Internet.Service", "Married", "Monthly.Charge", 
                       "Multiple.Lines", "Number.of.Referrals", 
                       "Online.Backup", "Online.Security", "Paperless.Billing", "Phone.Service", 
                       "Population", "Premium.Tech.Support", "Referred.a.Friend", 
                       "Satisfaction.Score", "Senior.Citizen", 
                       "Streaming.Music", "Tenure.in.Months", "Total.Charges", "Total.Refunds", "Under.30", "Unlimited.Data", 
                       "Zip.Code")

# Subset the data to keep only selected features
subset_data <- tr_data[, selected_features]

# 9. Create training and testing splits using createDataPartition
set.seed(123)  
train_index <- createDataPartition(subset_data$Churn, p = 0.8, list = FALSE)  # 80% training data
train_data <- subset_data[train_index, ]
test_data <- subset_data[-train_index, ]


# 10. visualization 

# Visualize the distribution of Churn (target variable)
ggplot(subset_data, aes(x = Churn)) +
  geom_bar(fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Churn", x = "Churn", y = "Count")
# Visualize Monthly Charge vs Churn
ggplot(train_data, aes(x = Monthly.Charge, fill = Churn)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Density of Monthly Charge by Churn", x = "Monthly Charge", y = "Density")

# Visualize Tenure vs Churn
ggplot(train_data, aes(x = Tenure.in.Months, fill = Churn)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Density of Tenure by Churn", x = "Tenure in Months", y = "Density")


# 11. Support Vector Machine (SVM) Model Training
X_train <- train_data[, -which(names(train_data) == "Churn")]
y_train <- train_data$Churn
X_test <- test_data[, -which(names(test_data) == "Churn")]
y_test <- test_data$Churn  # Define y_test

svm_model_rbf <- svm(X_train, y_train, type = "C-classification", kernel = "radial")
summary(svm_model_rbf)

# 12. Confusion Matrix Evaluation for Model Performance
# Make predictions on the test data
svm_predictions <- predict(svm_model_rbf, X_test)

# Ensure consistent factor levels
svm_predictions <- factor(svm_predictions, levels = c(0, 1))
y_test <- factor(y_test, levels = c(0, 1))

confusion_matrix <- confusionMatrix(svm_predictions, y_test)
print(confusion_matrix)

# 13. SVM model with cross-validation

train_control <- trainControl(method = "cv", number = 5)

# Train the SVM model using caret's train() with radial kernel
set.seed(123)
train_data$Churn <- factor(train_data$Churn, levels = c(0, 1))  # Convert to factor

svm_cv_model <- train(
  Churn ~ ., 
  data = train_data,
  method = "svmRadial",
  trControl = train_control,
  preProcess = c("center", "scale"),
  tuneLength = 5  # Tries 5 different combinations of hyperparameters
)

# Print CV results
print(svm_cv_model)


# 14. 8. Final SVM Model Training with Optimal Parameters
svm_final_model <- svm(Churn ~ ., data = train_data, kernel = "radial", cost = 2, gamma = 0.02196428)

# Print the summary of the final model
summary(svm_final_model)
# Predict on test data
svm_predictions_final <- predict(svm_final_model, test_data)

# Convert predictions to factors
svm_predictions_final <- factor(svm_predictions_final, levels = c(0, 1))
y_test <- factor(test_data$Churn, levels = c(0, 1))

# Evaluate performance using confusion matrix
confusion_matrix_svm <- confusionMatrix(svm_predictions_final, y_test)
print(confusion_matrix_svm)

###############################################################################



