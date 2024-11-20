![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
# Overview
This project predicts housing prices using a linear regression (lm) model, incorporating feature engineering, cross-validation, and regularization techniques such as Lasso Regression and Ridge Regression. The workflow ensures a robust and interpretable model with a comprehensive evaluation process, including residual analysis.

## Features
Polynomial Terms: Creating non-linear relationships through polynomial feature engineering.

Feature Selection: Using forward selection and correlation analysis to retain relevant variables.

Lasso and Ridge Regression: Employing regularization to handle multicollinearity and reduce overfitting.

Cross-Validation: Utilizing repeated cross-validation (repeatedcv) for robust performance metrics.

Residual Analysis: Validating model assumptions with diagnostic tools and visualizations.

## Libraries
This project utilizes the following R libraries:

car: For residual analysis and diagnostic testing.

glmnet: For Lasso and Ridge Regression.

ggplot2: For visualizing data and residuals.

dplyr: For efficient data manipulation.

summarytools: For summarizing and exploring data.

corrplot: For visualizing feature correlations.

tidyverse: For streamlined data workflows.

caret: For feature selection, model training, and cross-validation.
Install the libraries using:

code:
install.packages(c("car", "glmnet", "ggplot2", "dplyr", "summarytools", "corrplot", "tidyverse", "caret"))

## Dataset Details
Features:

- Zip code
- Number of bedrooms and bathrooms
- Living space (square footage)
- City, state, and county
- Zip code population and density
- Address
- Latitude and longitude

Target Variable: Housing prices

Dataset Source: Kaggle

Preprocessing Steps

    1. Converted categorical variables (zip code, city, state, county) to factors.

    2. Removed rows with missing values.
    3. Filled missing values for zip code population, zip code density, and median household income with the value from the nearest neighboring area, assuming similar characteristics.

    4. Dropped the Address column, as it is not relevant to predicting housing prices.


## Workflow
Data Exploration:
- Summarize data using summarytools.
- Visualize correlations with corrplot.
Feature Engineering:
- Add polynomial terms to capture non-linear effects.
- Perform forward selection to refine predictors.
Model Development:
- Train and tune a linear regression model with Ridge and Lasso regularization.
- Implement repeatedcv in the caret package for robust cross-validation.
Evaluation:
- Conduct residual analysis to validate model assumptions.
- Compare performance metrics (e.g., RMSE, R²) across models.
Example Usage

## Load libraries
library(car)

library(glmnet)

library(ggplot2)

library(dplyr)

library(summarytools)

library(corrplot)

library(tidyverse)

library(caret)

## Preprocess the dataset
hd <- read.csv("C:/Maryam/Project/American_Housing_Data_20231209 - American_Housing_Data_20231209.csv")

View(hd)

#### Converting these variables to factors 
hd$Zip.Code <- as.factor(hd$Zip.Code)
hd$City <- as.factor(hd$City)
hd$State <- as.factor(hd$State)
hd$County <- as.factor(hd$County)

summary(hd)

#### Finding missing values
rows_with_na <- apply(hd, 1, function(row) any(is.na(row)))

x <- hd[rows_with_na,]

print(x)

#### Deleting row with missing value
hd <- hd[-27787,]

#### Filling row with a similar value above it
hd$Zip.Code.Population[27786] <- 565

hd$Zip.Code.Density[27786] <- 41.0

hd$Median.Household.Income[is.na(hd$Median.Household.Income)] <- 106290


#### Deleting Useless Columns
cols.dont.want <- c("Address")

hd <- hd[, ! names(hd) %in% cols.dont.want, drop = F]

## Generate polynomial terms
model <- lm(log(Price) ~ poly(Living.Space, 2) + poly(Baths, 2) + poly(Median.Household.Income, 2),  
                 data = hd)
## Train a Lasso model
set.seed(125)

x <- model.matrix(log(Price) ~ poly(Living.Space, 2) + poly(Baths, 2) + poly(Median.Household.Income, 2), data = hd)

y <- hd$Price

fit.lasso = glmnet(x,y)

plot(fit.lasso, xvar="lambda", label=TRUE)

cv.lasso <- cv.glmnet(x,y)

plot(cv.lasso)

coef(cv.lasso)

## Cross Validation
set.seed(125)

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

model <- train(log(Price) ~ poly(Living.Space, 2) + poly(Baths, 2) + poly(Median.Household.Income, 2), data = hd, method = "lm", trControl = train_control)

print(model)

    Results
    Linear Regression

    39980 samples
        3 predictor

    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 3 times)
    Summary of sample sizes: 35982, 35982, 35982, 35982, 35981, 35983, ...
    Resampling results:

    RMSE       Rsquared   MAE
    0.5287774  0.5815955  0.3942567

    Tuning parameter 'intercept' was held constant at a value of TRUE

## Analyze residuals
model <- lm(log(Price) ~ poly(Living.Space, 2) + poly(Baths, 2) + poly(Median.Household.Income, 2), data = hd)

residuals <- resid(model)

#### detecting fit
yhat <- fitted(model)

plot(yhat, residuals, xlab = "Predicted Values", ylab= "residuals")
abline(h = 0, col = "red")

MSE <- mean(residuals^2)

#### detecting normality
hist(freq = TRUE,residuals, main = "Histogram of Residuals",
     xlab = "Residuals", ylab = "Frequency")

abline(h = 0, col = "red")

![image](https://github.com/user-attachments/assets/52758afa-0282-41f6-82b5-1d5c217eb96b)
![image](https://github.com/user-attachments/assets/b389af56-1c04-47d0-84a8-310f6294c846)


# Final Model
model <- lm(formula = log(Price) ~ poly(Living.Space, 2) + poly(Baths, 2) + poly(Median.Household.Income, 2), data = hd)

summary(model)

    Call:
    lm(formula = log(Price) ~ poly(Living.Space, 2) + poly(Baths, 
        2) + poly(Median.Household.Income, 2), data = hd)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -5.3161 -0.3084 -0.0164  0.3114  6.4744 

    Coefficients:
                                        Estimate Std. Error t value Pr(>|t|)    
    (Intercept)                        12.959145   0.002621 4944.74   <2e-16 ***
    poly(Living.Space, 2)1             40.751537   0.861961   47.28   <2e-16 ***
    poly(Living.Space, 2)2            -16.378047   0.605669  -27.04   <2e-16 ***
    poly(Baths, 2)1                    32.315128   0.880065   36.72   <2e-16 ***
    poly(Baths, 2)2                   -17.730463   0.582785  -30.42   <2e-16 ***
    poly(Median.Household.Income, 2)1  78.084163   0.544882  143.31   <2e-16 ***
    poly(Median.Household.Income, 2)2 -18.909630   0.526366  -35.92   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    Residual standard error: 0.524 on 39973 degrees of freedom
    Multiple R-squared:  0.5888,	Adjusted R-squared:  0.5887 
    F-statistic:  9538 on 6 and 39973 DF,  p-value: < 2.2e-16 
