#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all loaded data sets, functions and so on.

###################
###################
#### Libraries ####
###################
###################

library(easypackages) # enables the libraries function
suppressPackageStartupMessages(
  libraries("dplyr", # for data wrangling
            "ggplot2", # for plotting
            "rsample", # for data splitting
            "earth", # for fitting MARS models
            "caret", # for automating the tuning process
            "vip", # for variable importance
            "pdp", # for variable relationships
            "pls", # for pls and pcr regression
            "kableExtra", # for comlex tables
            "tidyverse",
            "doParallel" # for parallel computing
  ))

###############################
###############################
#### Set Working Directory ####
###############################
###############################

setwd("C:/R Portfolio/Multivariate Adaptive Regression Splines/Data")

bikes <- read.csv("bikes.csv")
str(bikes)
glimpse(bikes)
summary(bikes)

# Convert categorical variables into factors

bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$weather <- as.factor(bikes$weather)

# Convert numeric variables into integers

bikes$temperature <- as.integer(bikes$temperature)
bikes$realfeel <- as.integer(bikes$realfeel)
bikes$windspeed <- as.integer(bikes$windspeed)

levels(bikes$season) <- c("Spring", "Summer", "Autumn", "Winter")
library(plyr)
bikes <- revalue(bikes$season, c("1"="Spring", "2"="Summer", "3" = "Autumn", "4" = "Winter"))

# remove column named date
bikes <- bikes %>% select(-date)

###############################
###############################
# Training and Test Data Sets #
###############################
###############################

set.seed(1234) # changing this alters the make up of the data set, which affects predictive outputs

ind <- sample(2, nrow(bikes), replace = T, prob = c(0.8, 0.2))
train <- bikes[ind == 1, ]
test <- bikes[ind == 2, ]

#############
#############
# Modelling #
#############
#############

mrsl <- earth(rentals ~., train)
summary(mrsl)

plot(mrsl, which = 1)


# create a tuning grid
hyper_grid <- expand.grid(degree = 1:3,
                          nprune = seq(2, 50, length.out = 10) %>%
                            floor())

train2 <- train %>% dplyr::select(-rentals)

# make this example reproducible
set.seed(1)
# fit MARS model using k-fold cross-validation
tuned_mars <- train(
  x = train2,
  y = train$rentals,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid)

# display model with lowest test RMSE
tuned_mars$results %>%
  filter(nprune==tuned_mars$bestTune$nprune, degree == tuned_mars$bestTune$degree) 

# The model that contained the lowest RMSE was the one that had no interction terms and had two terms

# display test RMSE by terms and degree
ggplot(tuned_mars)

# best model
tuned_mars$bestTune

################################################
################################################
# Compare against Other Models for RMSE Metric #
################################################
################################################

#######################
# multiple regression #
#######################

set.seed(123)
cv_model1 <- train(
  rentals ~ ., 
  data = train, 
  method = "lm",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale")
)

##################################
# principal component regression #
##################################

set.seed(123)
cv_model2 <- train(
  rentals ~ ., 
  data = train, 
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  metric = "RMSE",
  preProcess = c("zv", "center", "scale"),
  tuneLength = 20
)

####################################
# partial least squares regression #
####################################

set.seed(123)
cv_model3 <- train(
  rentals ~ ., 
  data = train, 
  method = "pls",
  trControl = trainControl(method = "cv", number = 10),
  metric = "RMSE",
  preProcess = c("zv", "center", "scale"),
  tuneLength = 20
)

##########################
# regularized regression #
##########################

set.seed(123)
cv_model4 <- train(
  rentals ~ ., 
  data = train,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  metric = "RMSE",
  preProcess = c("zv", "center", "scale"),
  tuneLength = 10
)

##############################################
# extract out of sample performance measures #
##############################################

summary(resamples(list(
  Multiple_regression = cv_model1, 
  PCR = cv_model2, 
  PLS = cv_model3,
  Elastic_net = cv_model4,
  MARS = cv_mars
)))$statistics$RMSE %>%
  kableExtra::kable() %>%
  kableExtra::kable_styling(bootstrap_options = c("striped", "hover"))

# By incorporating non-linear relationships and interaction effects, the MARS model provides a substantial improvement over the previous linear models that we have explored.

#############################
#############################
# variable importance plots #
#############################
#############################

######################################
# Generalized Cross-Validation (GCV) #
######################################

# GCV can be regarded as an approximation to leave-one-out cross-validation (CV). Hence, GCV provides an approximately unbiased estimate of the prediction error

p1 <- vip(tuned_mars, num_features = 40, bar = FALSE, value = "gcv") + 
  ggtitle("GCV") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) 

#################################
# Residual Sum of Squares (RSS) #
#################################

p2 <- vip(tuned_mars, num_features = 40, bar = FALSE, value = "rss") + 
  ggtitle("RSS") + 
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) 

gridExtra::grid.arrange(p1, p2, ncol = 2)

# Its important to realize that variable importance will only measure the impact of the prediction error as features are included; however, it does not measure the impact for particular hinge functions created for a given feature. For example, in the above figure, we see that temperature and humidity are the two most influential variables; however, variable importance does not tell us how our model is treating the non-linear patterns for each feature.

coef(tuned_mars$finalModel) 

# To better understand the relationship between these features and rentals, we can create partial dependence plots (PDPs) for each feature individually and also an interaction PDP. 

p1 <- pdp::partial(tuned_mars, pred.var = "temperature", grid.resolution = 10) %>% autoplot() + theme_classic()  + ylab("Rentals")
p2 <- pdp::partial(tuned_mars, pred.var = "windspeed", grid.resolution = 10) %>% autoplot() + theme_classic() + ylab("Rentals")
p3 <- pdp::partial(tuned_mars, pred.var = c("temperature", "windspeed"), chull = TRUE, grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "Rentals", drape = TRUE, colorkey = TRUE, screen = list(z = -20, x = -60))

gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

##########################
##########################
# Avoiding Extrapolation #
##########################
##########################

# In two or more dimensions, plotting the convex hull is more informative; it outlines the predictor space region that the model was trained on. When chull = TRUE, the convex hull of the first two dimensions of zs (i.e., the first two variables supplied to pred.var) is computed; for example, if you set chull = TRUE in the call to partial only the region within the convex hull of the first two variables is plotted. Over interpreting the PDP outside of this region is considered extrapolation and is ill-advised

P4 <- pdp::partial(tuned_mars, pred.var = c("temperature", "windspeed"), plot = T, chull = TRUE, grid.resolution = 10)
P4

# The above figure indicates that dependency of windspeed more than 15 and temperature higher than 70 is extrapotaion and windspeed less than 5 and more than 80 is also extrapolation

######################
######################
# Parallel Computing #
######################
######################

# For three or more variables

library(doParallel) # load the parallel backend
no_cores <- detectCores() # detect core number for cluster
cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend

pdp::partial(tuned_mars, 
        pred.var = c("temperature", "windspeed", "realfeel"), 
        plot = TRUE,
        chull = TRUE, 
        parallel = TRUE, 
        paropts = list(.packages = "earth")) # Figure 6
stopCluster(cl) # good practice

###########################
###########################
# classification Problems #
###########################
###########################

# Indicate the clustering for rentals, based on windspeed and temperature split seperately into the four seasons.

pd <- NULL
for (i in 1:4) {
  tmp <- pdp::partial(tuned_mars, pred.var = c("temperature", "windspeed"),
                 which.class = i, grid.resolution = 101, progress = "text")
  pd <- rbind(pd, cbind(tmp, season = levels(train$season)[i]))
}

ggplot(pd, aes(x = temperature, y = windspeed, z = yhat, fill = yhat)) +
  geom_tile() +
  geom_contour(color = "white", alpha = 0.5) +
  scale_fill_distiller(name = "Rentals", palette = "Spectral") +
  theme_bw() +
  facet_grid(~ season)

###########################
###########################
# Predicted Probabilities #
###########################
###########################

library(randomForest) # for svm function
Season_train <- randomForest(season ~ ., data = train, probability = TRUE)
pred.prob <- function(object, newdata) {
  pred <- predict(object, newdata, probability = TRUE)
  prob.season <- attr(pred, which = "probabilities")[, "Summer"]
  mean(prob.season)
}
warnings()
# PDPs for Petal.Width and Petal.Length on the probability scale
pdp.pw <- pdp::partial(Season_train, pred.var = "temperature", 
                       pred.fun = pred.prob, plot = TRUE)
pdp.pl <- pdp::partial(Season_train, pred.var = "windspeed", pred.fun = pred.prob,
                  plot = TRUE)
pdp.pw.pl <- pdp::partial(Season_train, pred.var = c("temperature", "windspeed"),
                     pred.fun = pred.prob, plot = TRUE)
# Figure 8
grid.arrange(pdp.pw, pdp.pl, pdp.pw.pl, ncol = 3)
