# Exercises for Chapter 1

### 1. How would you define machine learning?

Machine learning is the study and practice of enabling computers to make decisions and predictions without the need to define strict, programmatic rules. This often requires large amounts of data and resources to produce an effective system.

### 2. Can you name four types of applications where it shines?

1. Handwriting recognition.
2. Spam filtering.
3. Product recommendation systems.
4. Credit card fraud detection.

### 3. What is a labeled training set?

A labeled training set is a set of data which includes some value that the model you are creating is intended to predict.

### 4. What are the two most common supervised tasks?

1. Classification.
2. Regression (prediction).

### 5. Can you name four common unsupervised tasks?

1. Clustering.
2. Anomaly detection.
3. Novelty detection.
4. Association rule learning.

### 6. What type of algorithm would you use to allow a robot to walk in various unknown terrains?

I would use a reinforcement learning algorithm.

### 7. What type of algorithm would you use to segment your customers into multiple groups?

I would use a clustering algorithm.

### 8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?

I would frame the spam detection problem as a supervised learning problem since it predicts (with some confidence interval) or classifies emails as spam/ham and the training process uses email data labeled in the same way to produce the model.

### 9. What is an online learning system?

An online learning system is one in which the system is trained incrementally, rather than in one large batch before the model is deployed for its duty.

### 10. What is out-of-core learning?

Out-of-core learning is a process in which training data is split into smaller chunks so datasets which are not able to be stored in the computers memory can be used to train a model. This probably requires particular kinds of models which support this kind of training.

### 11. What type of algorithm relies on a similarity measure to make predictions?

An instance-based algorithm (as opposed to a model-based algorithm).

### 12. What is the difference between a model parameter and a model hyperparameter?

A model parameter is a value derived from the training set which is used when the model does its duty. A hyperparameter controls how model parameters evolve or valid values they may attain and are used during the training of the model.

### 13. What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

A model-based algorithm's training process searches for parameters for some mathematical model the produce good predictions given new inputs. A common strategy for this is to optimize the proposed model based on a utility/cost function by maximizing or minimizing it. Predictions are made by evaluating the mathematical model with the learned parameters on new inputs.

### 14. Can you name four of the main challenges in machine learning?

1. Insufficient quantity of training data.
2. Poor quality training data.
3. Non-representative training data.
4. Data with irrelevant features.

### 15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

Your model is likely over-fitting. Three possible solutions are:

1. Simplify the model, perhaps choosing one with fewer parameters or by removing irrelevant features or by using dimensional reduction.
2. Use more training data.
3. Reduce noise in the training data by removing outliers and fixing errors.

### 16. What is a test set, and why would you want to use it?

A test set is a portion of your training data which is set aside for evaluating your model after it's trained to assess how well it performs.

### 17. What is the purpose of a validation set?

The purpose of a validation set is to evaluate your model while engaging in hyperparameter tuning. Several models are trained and evaluated against the validation set. The best performing model is selected and trained with the validation set included and finally evaluated against the test set.

### 18. What is the train-dev set, when do you need it, and how do you use it?

The train-dev set is yet another portion of the training data which is helpful for highlighting issues you may face while training your model. After training a particular model, it can be evaluated against the train-dev set and if it performs poorly this likely indicates over-fitting. If it performs well on the tain-dev set, but performs poorly on the validation set then data mismatch may be at play (i.e. the data in the training set isn't similar enough to the data expected in the field). If these both perform well, then the model can be evaluated against the test set.

### 19. What can go wrong if you tune hyperparameters using the test set?

You will likely create an over-fitted model which may perform poorly on new inputs since the model and hyperparameters are optimized to that particular set.
