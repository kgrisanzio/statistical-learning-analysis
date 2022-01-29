# Katherine Grisanzio
# Statistical learning analysis
# 10/25/20


##---------Load packages-------

library(poLCA)
library(flexmix)
library(vcd)
require(qgraph)
require(MPsychoR)
require(corrplot)
require(elasticnet) 
library(tidyverse)
library(partykit)       
library(caret)
library(ISLR)           
library(magrittr)              
library(pROC)
library(AppliedPredictiveModeling) 
library(cluster) 
library(factoextra) 
library(NbClust) 
library(mclust) 
library(glmnet) 

##---------Load data-------

# Dataset #1
data <- read.csv("data.csv")
head(data)
nrow(data) # 700 participants

data$gender <- as.factor(data$gender)

# Dataset #2
data_schoolstaff <- read.csv("data_schoolstaff.csv")
head(data_schoolstaff)
nrow(data_schoolstaff) # 86 participants

##---------Clustering-------

# Let's cluster individuals on their four emotion factors (cols 33:36)

# 1) K-Means

# Determine number of clusters to retain using the NbClust package to tell us the majority vote
fitmany_kmeans <- NbClust(data[c(33:36)], min.nc = 2, max.nc = 10, method = "kmeans")
# 2 clusters is the majority vote
fitmany_kmeans$Best.nc
table(fitmany_kmeans$Best.partition)    
fitmany_kmeans$Best.partition # cluster memberships

set.seed(123)
fit_kmeans <- kmeans(data[c(33:36)], centers = 2)
fit_kmeans

fit_kmeans$centers # cluster centers
fit_kmeans$cluster # cluster memberships
table(fit_kmeans$cluster) # cluster sizes
tapply(rowMeans(data[c(33:36)]), fit_kmeans$cluster, mean) # mean ratings for each cluster


# Visualize the clusters with a line plot

# Add cluster number to data file
data$cluster_kmeans <- fit_kmeans$cluster

# Reshape data for plotting
varying <- colnames(data)[c(33:36)]
df_long <- reshape(data, varying = varying, idvar = "PIN", timevar = "component", v.names = "score", times = varying, direction = "long")
df_long <- data.frame(df_long, row.names = NULL)

ggplot(df_long, aes(component, score, group = as.factor(cluster_kmeans), color = as.factor(cluster_kmeans))) +
  stat_summary(fun.y = "mean", geom = "line", size = 1) +
  stat_summary(fun.y = "mean", geom = "point") +
  stat_summary(fun.data=mean_se, geom="errorbar",width = .2, size=.75, color = "black", alpha = .3) +
  theme_bw(base_size = 14) +
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1), legend.position = "top", legend.title=element_blank(),
        legend.text = element_text(size = 26)) 
# It appears we have a "low emotion" cluster and a "high emotion" cluster

# Perform t-test to see whether reports of emotional support differs across the two emotion clusters
boxplot(data$emotional_support ~ fit_kmeans$cluster, xlab = "Cluster", ylab = "Emotional Support")
fit_ttest <- t.test(data$emotional_support ~ fit_kmeans$cluster)
fit_ttest
# p<.05 the clusters do differ on levels of emotional support, with cluster 1 having sig higher levels


# 2) K-mediods

# Try again quickly using k-mediods method, a more robust version of k-means

set.seed(123)
fit_kmediods <- pam(data[c(33:36)], 2)
fit_kmediods
fit_kmediods$medoids # these are the prototype levels of emotion for each cluster


# 3) Hierarchical

# And now with hierarchical clustering using ward's method, which aims to find compact, 
# spherical clusters. The dissimilarities are squared before cluster updating.

# Determine number of clusters
fitmany_ward <- NbClust(data[c(33:36)], min.nc = 2, max.nc = 10, method = "ward.D2")
# Also indicates 2 clusters
fitmany_ward$Best.nc
table(fitmany_ward$Best.partition)    
fitmany_ward$Best.partition

# Plot dendrogram
d <- dist(data[c(33:36)], method = "euclidean")
fit <- hclust(d, method = "ward.D2", members = NULL)
dend <- as.dendrogram(fit) 
plot(dend)
# To the eye (which shouldn't be used for cluster number determination), we can see two or three obvious clusters

cluster_hierarchical <- cutree(fit, 2)
data$cluster_hierarchical <- cluster_hierarchical

# Visualize the clusters

# Reshape data for plotting
varying <- colnames(data)[c(33:36)]
df_long <- reshape(data, varying = varying, idvar = "PIN", timevar = "component", v.names = "score", times = varying, direction = "long")
df_long <- data.frame(df_long, row.names = NULL)

ggplot(df_long, aes(component, score, group = as.factor(cluster_hierarchical), color = as.factor(cluster_hierarchical))) +
  stat_summary(fun.y = "mean", geom = "line", size = 1) +
  stat_summary(fun.y = "mean", geom = "point") +
  stat_summary(fun.data=mean_se, geom="errorbar",width = .2, size=.75, color = "black", alpha = .3) +
  theme_bw(base_size = 14) +
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1), legend.position = "top", legend.title=element_blank(),
        legend.text = element_text(size = 26)) 
# These clusters look very similar to the k-means clusters, in that we have one "high" emotion and one "low" emotion. 


##---------Latent Class Analysis-------

# We'll conduct a LCA on our categorical data (ordinal, likert responses): columns 5-32

formula <- cbind(Response.PedRepAng13, Response.PedRepAng14, Response.PedRepAng16, Response.PedRepAng17, Response.PedRepAng18, 
                 Response.PedRepAnx42, Response.PedRepAnx43, Response.PedRepAnx44_Anxiety51, Response.PedRepAnx46, Response.PedRepAnx48_Anxiety57,
                 Response.PedRepAnx50, Response.PedRepAnx51, Response.PedRepDep36_Depression48, Response.PedRepDep38_Depression36, 
                 Response.PedRepDep41, bisbas1, bisbas2, bisbas3, bisbas4, bisbas5, bisbas6, bisbas7, X95, X98, X101, X176, X197, X271) ~ 1
formula

# Vary the cluster number from 1 to 5 and pull out the BIC

data[,20:26] <- data[,20:26] + 1 # these variables start with 0 so we need to add 1, because in poLCA the lowest category needs to start with 1

set.seed(123)
bicvec <- rep(NA, 4)
names(bicvec) <- paste0("k=", 2:5)
for (i in 2:5) {
  bicvec[i-1] <- poLCA(formula, data[5:32], nclass = i, graphs = FALSE, nrep = 10)$bic
}

bicvec
# The best model is the one with 3 classes (lowest BIC)

# Re-fit the model 10 times because the EM-algorithm is sensitive to starting values
fitlca <- poLCA(formula, data, nclass = 3, graphs = TRUE, nrep = 10)  # picks the best one

# Relative cluster sizes 
round(fitlca$P, 2)  

# Cluster-specific item ("solving") probabilities 
fitlca$probs

# Posterior probabilities for each participant
postmat <- round(fitlca$posterior, 2)
rownames(postmat) <- rownames(data)
colnames(postmat) <- paste0("C", 1:3)
postmat

# External evaluation with gender (not of theoretical interest, just to try)
table(data[,3], fitlca$predclass) # cross-classifiying cluster memberships with gender
mosaicplot(fitlca$predclass ~ data[,3], xlab = "Cluster", ylab = "Gender", main = "Party and Gender")
# We see that cluster 1 is slightly more males (gender=1) and cluster 3 is slightly more females (gender=2),
# but overall not too much difference.


##---------Latent Class Analysis with Mixed Input Scale Levels-------

# To demonstrate the method with mixed input scales, we'll use 3 categorical variables and 1 metric variable 
set.seed(123)
flex <- stepFlexmix(~ 1, data = data, k = 1:4, nrep = 3,
                       model = list(FLXMRmultinom(Response.PedRepAng13 ~ .), # categorical
                                    FLXMRmultinom(Response.PedRepAnx42 ~ .), # categorical
                                    FLXMRmultinom(Response.PedRepDep36_Depression48 ~ .), # categorical
                                    FLXMRmultinom(bisbas1 ~ .), # categorical
                                    FLXMRglm(emotional_support ~ ., family = "gaussian"))) # metric

flex2 <- getModel(flex, "BIC") # extract best model
summary(flex2) # 3-cluster solution
# This summary shows the prior probability for each component, the number of observations assigned to 
# the corresponding cluster (N = 426, 89, and 185), the number of observations with a posterior probability 
# larger than eps, and the ratio of the latter two numbers. This ratio indicates how separated the cluster is 
# from the others -- we can see that cluster 1 is more separated than the other two.

parameters(flex2)
# We see, for example, that people in cluster 3 hqve higher levels of emotional support than the other two clusters


# Visualization
cluster <- as.factor(flex2@cluster) # hard cluster membership
A <- mosaic(Response.PedRepAng13 ~ cluster, data = data, return_grob = TRUE)
B <- mosaic(Response.PedRepAnx42 ~ cluster, data = data, return_grob = TRUE)
C <- mosaic(Response.PedRepDep36_Depression48 ~ cluster, data = data, return_grob = TRUE)
D <- mosaic(bisbas1 ~ cluster, data = data, return_grob = TRUE)
mplot(A, B, C, D) # mosaic plots for each item
# These items are all on a 1-5 likert scale, with 5 being higher levels of an emotion (e.g. Response.PedRepAng13
# is a question about anger). So, we can see that each cluster is reporting different levels of emotion, with 
# Cluster 2 showing higher levels (more 4's and 5's), with cluster 3 showing the lowest levels of emotion
# (more 1's).


##---------Latent Profile Analysis-------

# Rename some variables for easier interpretation/visualization
names(data_schoolstaff)[2] <- "Effectiveness_in_job"
names(data_schoolstaff)[3] <- "Satisfied_with_job"
names(data_schoolstaff)[4] <- "Connected_to_adults"
names(data_schoolstaff)[5] <- "Feel_respected"
names(data_schoolstaff)[6] <- "Feel_like_belong"
names(data_schoolstaff)[7] <- "Trusted_to_teach"
names(data_schoolstaff)[8] <- "Positive_attitudes_of_colleagues"
names(data_schoolstaff)[9] <- "Own_positive_attitudes"
names(data_schoolstaff)[10] <- "Positve_working_environment"
names(data_schoolstaff)[11] <- "Anxious_at_work_COVID"
names(data_schoolstaff)[12] <- "Confident_doing_job_COVID"
names(data_schoolstaff)[13] <- "Colleagues_understand_me_as_person"
names(data_schoolstaff)[14] <- "Feel_you_matter_to_others"

set.seed(1)
data_lpa <- data_schoolstaff[c(2:24)]

# Plot Bayesian Information Criteria for all models with profiles ranging from 1 to 9
BIC <- mclustBIC(data_lpa)
plot(BIC)
summary(BIC) # shows the top three models based on BIC

# We can also compare values of the Integrated Completed Likelikood (ICL) criterion. ICL isn’t much different 
# from BIC, except that it adds a penalty on solutions with greater entropy or classification uncertainty.
ICL <- mclustICL(data_lpa)
plot(ICL)
summary(ICL) # shows the top three models based on ICL

# Perform the Bootstrap Likelihood Ratio Test (BLRT) which compares model fit between k-1 and k cluster models. 
# In other words, it looks to see if an increase in profiles increases fit. Based on simulations by Nylund, Asparouhov, 
# and Muthén (2007) BIC and BLRT are the best indicators for how many profiles there are. 
mclustBootstrapLRT(data_lpa, modelName = "EII")
# Suggests that 5 clusters may be optimal number 

# Because the BIC and ICL suggested 4 clusters and we want to keep this as simple as possible, we'll
# proceed with 4 clusters

# Run LPA with EII, which means spherical, equal volume clusters
mod1 <- Mclust(data_lpa, modelNames = "EII", G = 4, x = BIC) 
summary(mod1)

# Add cluster number to data file
data_schoolstaff$cluster <- mod1$classification

# Transform data to long for visualization
varying <- colnames(data_schoolstaff)[2:24]
df_schoolstaff_long <- reshape(data_schoolstaff, varying = varying, idvar = "id", timevar = "component", v.names = "score", times = varying, direction = "long")
df_schoolstaff_long <- data.frame(df_schoolstaff_long, row.names = NULL)

df_schoolstaff_long$component <- factor(df_schoolstaff_long$component, levels = c("Anxious_at_work_COVID", "Exhausted", "Frustrated",  "Stressed.Out", "Worried", "Effectiveness_in_job", "Satisfied_with_job", "Own_positive_attitudes", "Positve_working_environment", "Confident_doing_job_COVID", "Engaged", "Excited", "Happy",  "Positive_attitudes_of_colleagues", "Hopeful", "Connected_to_adults", "Feel_respected", "Feel_like_belong", "Colleagues_understand_me_as_person", "Feel_you_matter_to_others", "Trusted_to_teach"))
df_schoolstaff_long <- df_schoolstaff_long[which(!df_schoolstaff_long$component == "Safe"),]

# Plot 
ggplot(df_schoolstaff_long, aes(component, score, group = as.factor(cluster), color = as.factor(cluster))) +
  stat_summary(fun.y = "mean", geom = "line", size = 2) +
  stat_summary(fun.y = "mean", geom = "point") +
  stat_summary(fun.data=mean_se, geom="errorbar",width = .2, size=.75, color = "black", alpha = .3) +
  theme_bw(base_size = 14) +
  labs(x = "", y = "Average Score of Cluster") +
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
        plot.margin = margin(10, 10, 10, 50), legend.position = "none") +
  scale_x_discrete(labels=c("Anxious_at_work_COVID" = "Anxious at work (COVID)", "Stressed.Out" = "Stressed Out",
                            "Effectiveness_in_job" = "Feel effective in job", "Satisfied_with_job" = "Satisfied with job", 
                            "Own_positive_attitudes" = "Own positive attitudes", "Positve_working_environment" = "Positive working environment",
                            "Confident_doing_job_COVID" = "Confident doing job (COVID)", "Positive_attitudes_of_colleagues" = "Positive attitudes of colleagues", 
                            "Connected_to_adults" = "Connected to adults at SMS", "Feel_respected" = "Feel respected", "Feel_like_belong" = "Feel like belong",
                            "Colleagues_understand_me_as_person" = "Colleagues understand me as a person", "Feel_you_matter_to_others" = "Feel you matter to others",
                            "Trusted_to_teach" = "Trusted to teach")) +
  scale_color_manual(values=c("#F8766D", "#00BA38", "#C77CFF", "#619CFF"))

# This LPA is really interesting and revealing. The items on the x-axis are organized such that negative emotions
# are on the left, professional well-being items are in the middle, and feelings of interpersonal connection are 
# on the right. One cluster (blue) is high on negative emotions, and low on professional well-being and 
# interpersonal connection. Another cluster (red) is high on negative emotions and low on professional well-being, but 
# is high on interpersonal connection. A third cluster (green) is sort of the opposite (and doing really well!) - they're
# low on negative emotions, high on professional well-being, and high on interpersonal connection. A final cluster (purple)
# is average on everything and sits right in the middle. 

##---------Mixture Regression Models-------

# Fit mixture regression models with emotional support as DV and sadness as predictor

# Fit k=1 solution for model comparison
fit0 <- flexmix(emotional_support ~ ML4_sadness, data = data, k = 1)

# Fit a linear regression
set.seed(123)
fit1 <- flexmix(emotional_support ~ ML4_sadness, data = data, k = 2)
fit1
summary(fit1)  
BIC(fit0, fit1) # 2-cluster solution fits better than 1 cluster
fit1@cluster # cluster assignments
parameters(fit1) # regression parameters

# Plot scatterplot with colors for cluster membership
xpred <- seq(-2,4.5, by = 0.05)
pred1 <- predict(fit1, newdat = data.frame(ML4_sadness = xpred))$Comp.1
# predictions for cluster 1
pred2 <- predict(fit1, newdat = data.frame(ML4_sadness = xpred))$Comp.2
# predictions for cluster 2

dev.new()
with(data, plot(ML4_sadness, emotional_support, xlab = "sadness", ylab = "emotional support", main = "Scatterplot", col = fit1@cluster))
legend("bottomright", pch = 1, col = 1:2, legend = c("Cluster 1", "Cluster 2"))
lines(xpred, pred1, col = 1, lwd = 2)
lines(xpred, pred2, col = 2, lwd = 2)
# We can see that cluster 2 has lower levels of emotional support overall, and might have a slightly stronger
# relationship between sadness and emotional support (steeper slope than cluster 1)


# Inference on these parameters
fit2 <- modeltools::refit(fit1)
summary(fit2)
# Sadness is a significant predictor of emotional support in both clusters, 
# with lower p-value for cluster 2


##---------Lasso-------

# Run a GLM with lasso, with all categorical variables as predictors and perceived hostility as DV

# Separate the predictors from the response
predictors <- data[c(5:32)]
head(predictors)
predictors <- as.matrix(predictors) # convert to matrix for glmnet
class(predictors)

response <- as.matrix(data["perceived_hostility"]) # convert to matrix for glmnet

# Standardize variables
predictors1 <- apply(predictors, 2, scale)
head(predictors1)
response1 <- scale(response)

# Fit an ordinary regression first
fit_lm <- glmnet(predictors1, response1, alpha = 0, lambda = 0) # same as a regular lm fit
coef(fit_lm)


# Shrinkage effect for different values of lambda
lambda_grid <- seq(2, 10e-4, length = 100) # lambda vector
lambda_grid
fit_lasso <- glmnet(predictors1, response1, lambda = lambda_grid)
plot(fit_lasso, "lambda", label = TRUE, lwd = 2, col = 1:8)


# Cross-validation to find the best lambda using cv.glmnet
set.seed(123)
fit_glmnetcv <- cv.glmnet(predictors1, response1, nfolds = 10, lambda = lambda_grid, type = "mse") # 10-fold cross-validation 
fit_glmnetcv 
plot(fit_glmnetcv)
lambda_best <- fit_glmnetcv$lambda.min
lambda_best # best lambda
log(lambda_best) # (first vertical line in plot)
lambda_best2 <- fit_glmnetcv$lambda.1se   
lambda_best2 # largest value of lambda such that error is within 1 standard error of the minimum
log(lambda_best2) # (second vertical line in plot)

# Fit the model with best lambda
fit_lasso1 <- glmnet(predictors1, response1, alpha = 1, lambda = lambda_best) # (alpha = 1 to indicate lasso)
round(coef(fit_lasso1), 3) 
# several variables eliminated

# Fit the model with 1se lambda
fit_lasso2 <- glmnet(predictors1, response1, alpha = 1, lambda = lambda_best2)
round(coef(fit_lasso2), 3) 
# even more variables eliminated, only 5 retained


##---------Graphical Lasso-------

# Try a graphical lasso on the 4 emotion factors (cols 32:36) and the 5 social functioning factors (cols 37:41)

# Visualize correlation structure
dev.new()
cormat <- cor(data[33:40])
corrplot(cormat, method = "circle")
dev.off()

# For comparison - regular correlation network (Bonferroni corrected)
dev.new()
qgraph(cormat, layout = "spring", threshold = "sig", sampleSize = nrow(data),
       graph = "cor", labels = colnames(data[33:40]), 
       title = "Emotions Correlation Network")

# For comparison - partial correlation network (significance threshold)
dev.new()
qgraph(cormat, layout = "spring", threshold = "sig", sampleSize = nrow(data),
       graph = "pcor", labels = colnames(data[33:40]), 
       title = "Emotions Partial Correlation Network")

# Graphical LASSO
dev.new()
qgraph(cormat, layout = "spring", sampleSize = nrow(data),
       graph = "glasso", labels = colnames(data[33:40]), 
       title = "Emotions Graphical LASSO")

# We see things that make a lot of sense in this graphical lasso network - friendship and loneliness
# are strongly negatively correlated, and feelings of different negative emotions (like anger, worry
# and sadness) are highly positively correlated.



##---------Sparse PCA-------

# Conduct a sparse PCA on the emotion items (cols 5:32)

# For comparison - standard PCA
pcaEmotion <- prcomp(data[5:32], scale = TRUE)
pcaEmotion
summary(pcaEmotion) 
# Explains 42.5% of the variance with 3 components, 48% with 4 components, 52.2% with 5 components
screeplot(pcaEmotion, type = "lines", main = "Emotion Scree Plot")
abline(h = 1, col = "gray", lty = 2)
# Seems like 4 components would be a good number to choose

loadings <- pcaEmotion$rotation[,1:4]
loadings
plot(loadings[,1:2], pch = 20, main = "Emotion Loadings Plot") # 2D loadings plot
text(loadings[,1:2], rownames(loadings), cex = 0.8, pos = 3)

# Sparse PCA

# Specify penalization with penalty
spcaEmotion <- spca(scale(data[5:32]), K = 4, sparse = "penalty", para = c(.5, .5, .5, .5)) 
spcaEmotion

# Specify penalization with number of loadings per component that should be different from 0
spcaEmotion <- spca(scale(data[5:32]), K = 4, sparse = "varnum", para = c(8, 8, 8, 8)) # We'll say, keep 8 for each component
spcaEmotion
sum(spcaEmotion$pev)*100
# Explains 38.9% of the variance, which is about 6% less than the standard PCA run above


# Question - How do we figure out the optimal penalty parameters to use and which spare loadings to retain?


##---------Conditional Inference Trees-------

# Create conditional inference tree using categorical variables as nodes
  # Predictors: age, gender, and categorical emotion variables (cols 5:32)
  # Response variable: perceived rejection (col 41)

# Convert categorical variables to ordered factors
data %<>% mutate_if(is.integer, ~ as.factor(ordered(.x)))
str(data)

# Fit classification tree with all predictors
perc_rej_tree <- ctree(perceived_rejection ~ ., data = data[c(2, 3, 5:32, 41)])
perc_rej_tree
plot(perc_rej_tree)

# Predicted probabilities for each observation
pred_perc_rej <- predict(perc_rej_tree, type = "prob")   
head(pred_perc_rej)

# Predicted response for each observation
pred_perc_rej1 <- predict(perc_rej_tree, type = "response")  
pred_perc_rej1


# Create another conditional inference tree using continuous variables as nodes
  # Predictors: emotion factors (cols 33:36)
  # Response variable: perceived rejection (col 41)

perc_rej_tree1 <- ctree(perceived_rejection ~ ., data = data[c(33:36, 41)])
perc_rej_tree1
plot(perc_rej_tree1)

# Predicted probabilities for each observation
pred_perc_rej <- predict(perc_rej_tree, type = "prob")   
head(pred_perc_rej)

# Predicted response for each observation
pred_perc_rej1 <- predict(perc_rej_tree, type = "response")  
pred_perc_rej1


# Do the same with another dependent variable - loneliness
  # Predictors: emotion factors (cols 33:36)
  # Response variable: loneliness (col 40)

loneliness_tree1 <- ctree(loneliness ~ ., data = data[c(33:36, 40)])
loneliness_tree1
plot(loneliness_tree1)

# Predicted probabilities for each observation
pred_loneliness <- predict(loneliness_tree1, type = "prob")   
head(pred_loneliness)

# Predicted response for each observation
pred_loneliness1 <- predict(loneliness_tree1, type = "response")  
pred_loneliness1

# We'll interpret this last tree with loneliness as the response variable, since it 
# is simplest. We can see that sadness are the higher nodes of the tree, doing the first
# splits. On the left most box, we have the people with lower levels of sadness (<= .6 score) and 
# low levels of evaluative anticipation (<= .44 score), and these people have the lowest 
# levels of loneliness too (boxplot is down near -1). In constrast, for the right most box,
# we have people with higher levels of sadness (> 1.875 score), and they have higher levels of 
# loneliness (boxplot around 2). 


# Conditional Inference Trees with CV stabilization

# Use the same predictors and response as loneliness we just ran
predictors <- data[c(33:36)]
response <- data$loneliness

set.seed(123)
caret_emotion <- caret::train(x = predictors, y = response, method = "ctree", 
                          trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3))
caret_emotion

# Set the tuning parameter to p = 0.05 (mincriterion = 0.95)
myGrid <- data.frame(mincriterion = 0.95)
myGrid
caret_emotion2 <- caret::train(x = predictors, y = response, method = "ctree", tuneGrid = myGrid, 
                           trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3))
caret_emotion2

dev.new()
plot(caret_emotion2$finalModel, main = "Emotion ctree through caret")   
# Looks identical to the first loneliness tree


##---------Regression Trees-------

# Create a regression tree with emotion factors (cols 33:36) as predictors and 
# friendship (col 38) as response variable

# Split into training and test data (80-20 split)
set.seed(123)
# Create partition index
train_ind <- createDataPartition(y = data$friendship, p = 0.8, list = FALSE) %>% as.vector() 
data_train <- data[train_ind,] # training set
dim(data_train)
data_test <- data[-train_ind,] # test set
dim(data_test)

friendship_tree <- ctree(friendship ~ ., data = data[c(33:36, 38)]) 
friendship_tree
plot(friendship_tree)

# Predictions on test data
preds <- predict(friendship_tree, newdata = data_test)   
# Calculate RMSE
(data_test$friendship - preds)^2 %>% sum() %>% sqrt()

# This tree isn't complicated, but let's simplify it just to try by setting p threshold to .0001
friendship_tree2 <- ctree(friendship ~ ., data = data[c(33:36, 38)], control = ctree_control(mincriterion = 0.9999))
plot(friendship_tree2)
# Predictions on test data
preds2 <- predict(friendship_tree2, newdata = data_test)   
(data_test$friendship - preds2)^2 %>% sum() %>% sqrt()  
# The RMSE increased, which is to be expected


##---------Random Forests-------

# Generate a random forest, which uses a subset of m predictors randomly chosen at each split

tc <- trainControl("oob") # OOB evaluation
mgrid <- expand.grid(mtry = seq(10, 100, 10))
mgrid

set.seed(123)
rf <- train(loneliness ~ ., data = data[c(5:32, 40)], method = 'cforest', trControl = tc, tuneGrid = mgrid)
rf
varImp(rf) # shows variable importance     

##---------Model-Based Recursive Partitioning-------

# We want to run some linear models with covariates, so instead of including complex interactions 
# in the model we'll use a tree, which will help us understand how these variables influence the
# relationship between the predictors and the response

# Linear model tree

lmt_emotion <- lmtree(emotional_support ~ ML1_worry + ML2_anger + ML3_eval_antic + ML4_sadness|age + gender, data = data)
# predictors: ML1_worry, ML2_anger, ML3_eval_antic, ML4_sadness; covariates: age, gender
plot(lmt_emotion)
summary(lmt_credit)
# Here we can see the scatterplots showing the relationship between the 4 emotion predictors and the response
# (emotional support) for each terminal node. It's hard to interpret this exact graph, and I can't quite tell
# the difference between the scatterplots by eye, but this could be a really useful method in the future to parse
# the effect the covariates have.



