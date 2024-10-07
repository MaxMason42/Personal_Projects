#Question 1
#Load data and transform categorical variables to factors
data <- read.table("C:/Users/Maxim/Downloads/CpsWages.txt", header = TRUE)

data$sex <- factor(data$sex, levels = c(0, 1), labels = c("male", "female"))
data$race <- factor(data$race, levels = c(1, 2, 3), labels = c("other", "Hispanic", "white"))
data$marr <- factor(data$marr, levels = c(0, 1), labels = c("unmarried", "married"))
data$occupation <- factor(data$occupation, levels = c(1, 2, 3, 4, 5, 6), labels = c("management", "sales", "clerical", "service", "professional", "other"))
data$sector <- factor(data$sector, levels = c(0, 1, 2), labels = c("other", "manufacturing", "construction"))
data$south <- factor(data$south, levels = c(0, 1), labels = c("North", "South"))
data$union <- factor(data$union, levels = c(0, 1), labels = c("not a member", "union member"))

#Scatterplot Matrix of the data
pairs(data)

#Checking normality with qqplot and Shapiro test
qqnorm(data$wage)
qqline(data$wage)

shapiro.test(data$wage)

#Log transforming the data and fitting a linear model with all predictors
log_wages <- data

log_wages$wage <- log(log_wages$wage)

log_model <- lm(wage ~ ., data = log_wages)

summary(log_model)


#Adding quadratic terms and finding the best predictors through BIC
library(bestglm)

wage = log_wages$wage

data_quadratic <- log_wages[, -6]
numerical_predictors <- c("age", "education", "experience")
for (predictor in numerical_predictors) {
  data_quadratic[paste0(predictor, "_sq")] <- data_quadratic[[predictor]]^2
}
d = cbind(data_quadratic, wage)

best_model <- bestglm(Xy = d, family = gaussian, IC = "BIC")

#BIC Best Model Plot
plot(0:(dim(d)[2]-1), best_model$Subsets$BIC, type = "b", ylab = "BIC",
     xlab = "Number of Covariates", lwd = 3, pch = 19, main = "BIC")
abline(v=which.min(best_model$Subsets$BIC)-1)

#BIC Best model Coefficients
summary(best_model$BestModel)

#Finding the best coefficients for non-log transformed data and fitting a GLM

wage = data$wage

data_quadratic <- data[, -6]
numerical_predictors <- c("age", "education", "experience")
for (predictor in numerical_predictors) {
  data_quadratic[paste0(predictor, "_sq")] <- data_quadratic[[predictor]]^2
}
d = cbind(data_quadratic, wage)

best_model <- bestglm(Xy = d, family = Gamma(link = "log"), IC = "BIC")

#BIC Best Model Plot
plot(0:(dim(d)[2]-1), best_model$Subsets$BIC, type = "b", ylab = "BIC",
     xlab = "Number of Covariates", lwd = 3, pch = 19, main = "BIC")
abline(v=which.min(best_model$Subsets$BIC)-1)

#BIC Best model Coefficients
summary(best_model$BestModel)



#Question 2

#Load data
Q2dat <- data.frame(X = c(2, 2, 0.667, 0.667, 0.4, 0.4, 0.286, 0.286, 0.222, 0.222, 0.2, 0.2),
                    Y = c(0.0615, 0.0527, 0.0344, 0.0258, 0.0138, 0.0258, 0.0129, 0.0183, 0.0083, 0.0169, 0.0129, 0.0087))

#Transform values, run simple linear regression, and calculate for initial values
Y_star <- 1 / Q2dat$Y
X_star <- 1 / Q2dat$X

model <- lm(Y_star ~ X_star)


g0_1 <- 1 / coef(model)["(Intercept)"]
g0_2 <- coef(model)["X_star"] / coef(model)["(Intercept)"]

cat("Initial estimate g^(0)_1:", g0_1, "\n")
cat("Initial estimate g^(0)_2:", g0_2, "\n")


#Create a non-linear model with the found starting values
nonlin <- nls(Y ~ ((gamma1 * X)/(gamma2 + X)), data = Q2dat, start = list(gamma1 = g0_1, gamma2 = g0_2))

summary(nonlin)


#Plot the scatterplot of the data with the fitted non-linear model
plot(Q2dat$X, Q2dat$Y, pch = 16, xlab = "X", ylab = "Y", main = "Scatterplot with Fitted Curve")
h = seq(0, 2, len=100)
lines(h, ((coef(nonlin)[1]*h)/(coef(nonlin)[2] + h)), col = "red", lwd = 2)


#Plotting residuals vs fitted values
plot(predict(nonlin), resid(nonlin))
abline(h=0)



#Question 4

set.seed(549)

x_val <- rnorm(100, mean = 0, sd = 1)

mean(x_val)


gamma = 0.5
beta1 = 1
n = 100

output_matrix <- matrix(nrow = 100000, ncol = 2)

for(i in 1:100000){
  error_variances <- exp(gamma * x_val[1:n])
  errors <- rnorm(n, mean = 0, sd = sqrt(error_variances))
  y_val <- 10 + beta1 * x_val[1:n] + errors

  model <- lm(y_val ~ x_val[1:n])
  output_matrix[i, ] <- coef(model)
}

output_matrix

mean(output_matrix[,1])
mean(output_matrix[,2])

gamma = 1
beta1 = 2
n = 10

output_matrix <- matrix(nrow = 100000, ncol = 2)

for(i in 1:100000){
  error_variances <- exp(gamma * x_val[1:n])
  errors <- rnorm(n, mean = 0, sd = sqrt(error_variances))
  y_val <- 10 + beta1 * x_val[1:n] + errors

  model <- lm(y_val ~ x_val[1:n])
  output_matrix[i, ] <- coef(model)
}

output_matrix

mean(output_matrix[,1])
mean(output_matrix[,2])
