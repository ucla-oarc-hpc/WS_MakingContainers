#From https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html#examples
library(h2o)
h2o.init(nthreads=1)

# Import the insurance dataset into H2O:
insurance <- h2o.importFile("https://s3.amazonaws.com/h2o-public-test-data/smalldata/glm_test/insurance.csv")

# Set the factors:
offset = log(insurance$Holders)
insurance$Holders <- as.factor(insurance$Holders)
insurance$Age <- as.factor(insurance$Age)
insurance$Group <- as.factor(insurance$Group)
insurance$District <- as.factor(insurance$District)


# Build and train the model:
dl <- h2o.deeplearning(x = 1:3,
                       y = "Claims",
                       distribution = "tweedie",
                       hidden = c(1),
                       epochs = 1000,
                       train_samples_per_iteration = -1,
                       reproducible = TRUE,
                       activation = "Tanh",
                       single_node_mode = FALSE,
                       balance_classes = FALSE,
                       force_load_balance = FALSE,
                       seed = 23123,
                       tweedie_power = 1.5,
                       score_training_samples = 0,
                       score_validation_samples = 0,
                       training_frame = insurance,
                       stopping_rounds = 0)

# Eval performance:
perf <- h2o.performance(dl)
perf

# Generate predictions on a test set (if necessary):
pred <- h2o.predict(dl, newdata = insurance)
pred