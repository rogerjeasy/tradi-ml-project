## ── install/load ───────────────────────────────────────────────────
if (!require('caTools'))       install.packages('caTools',       dependencies = TRUE)
if (!require('e1071'))         install.packages('e1071',         dependencies = TRUE)
if (!require('dplyr'))         install.packages('dplyr',         dependencies = TRUE)
if (!require('rcompanion'))    install.packages('rcompanion',    dependencies = TRUE)
if (!require('tidyr'))         install.packages('tidyr',         dependencies = TRUE)
if (!require('effectsize'))    install.packages('effectsize',    dependencies = TRUE)
if (!require('mgcv'))          install.packages('mgcv',          dependencies = TRUE)
if (!require('ggplot2'))       install.packages('ggplot2',       dependencies = TRUE)

library(dplyr)
library(tidyr)
library(ggplot2)
library(effectsize)
library(rcompanion)
library(caTools)
library(e1071)
library(mgcv)

## ── load ───────────────────────────────────────────────────────────
telco <- read.csv("train.csv")

## ── blanks → NA ────────────────────────────────────────────────────
telco <- telco %>% mutate(across(where(is.character), ~ na_if(.x, "")))

## ── visualise missingness ─────────────────────────────────────────
miss_pct <- telco %>% 
  summarise(across(everything(), ~ mean(is.na(.x))*100)) %>% 
  pivot_longer(everything(), names_to = "variable", values_to = "pct")

ggplot(miss_pct, aes(x = reorder(variable, pct), y = pct)) +
  geom_col() +
  coord_flip() +
  labs(x = NULL, y = "Percent missing")

## ── treat sparse character columns ────────────────────────────────
telco <- telco %>% mutate(
  Churn.Category = replace_na(Churn.Category, "None"),
  Churn.Reason   = replace_na(Churn.Reason,   "None"),
  Offer          = replace_na(Offer,          "None")
)

## ── factor conversions ────────────────────────────────────────────
binary_cols <- c("Churn","Married","Partner","Phone.Service","Senior.Citizen",
                 "Paperless.Billing","Unlimited.Data","Multiple.Lines",
                 "Online.Backup","Online.Security","Device.Protection.Plan",
                 "Premium.Tech.Support","Streaming.Movies","Streaming.Music",
                 "Streaming.TV","Referred.a.Friend","Dependents","Under.30")


## ── numeric median impute (if any remain) ─────────────────────────
num_cols <- telco %>% select(where(is.numeric)) %>% names()
telco    <- telco %>% mutate(across(all_of(num_cols), ~ replace_na(.x, median(.x, na.rm = TRUE))))

#Find which values are still missing
na_report <- telco |>
  summarise(across(everything(), \(x) sum(is.na(x)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") |>
  filter(n_missing > 0) |>
  arrange(desc(n_missing))

na_report

#Add "Unknown" to the factor’s levels
telco$Internet.Type <- factor(telco$Internet.Type,
                              levels = c(levels(telco$Internet.Type), "Unknown"))

#Replace NA values
telco$Internet.Type[is.na(telco$Internet.Type)] <- "Unknown"

sum(is.na(telco))      

#Prepare factors for analysis

telco <- telco %>%
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

## ── train/test split ──────────────────────────────────────────────
set.seed(42)
split <- sample.split(telco$Churn, SplitRatio = 0.7)
telco_training <- telco[split, ]
telco_testing  <- telco[!split, ]

