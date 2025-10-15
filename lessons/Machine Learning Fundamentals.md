# Machine Learning Fundamentals

**Author**: AMMS

**Repository**: [BI, ML and AI Class Repository](https://github.com/anguihero/Data_BI_AA_AI_Lessons)

**Update Date**: 2025/10/14

This document presents a concise guide on the fundamental concepts of machine learning, including supervised and unsupervised methods, main algorithms, evaluation metrics, and common challenges. The content is designed to serve as a quick reference for students and professionals interested in understanding the basic principles of machine learning.

# Introduction to Machine Learning Methods

Machine Learning is a field of artificial intelligence that focuses on developing computer programs that can access data and use it to learn by themselves. It is fundamentally divided into two major categories based on the type of training data: Supervised and Unsupervised.

The use of machine learning methods is necessary when problems are too complex to be solved through explicit rules or when data contains patterns that are not evident at first glance. These methods allow for task automation, improved decision-making, and discovery of valuable information from large volumes of data.

![alt text](https://scikit-learn.org/1.3/_static/ml_map.png)

**Examples of Supervised Learning applications:**
1. **Medical diagnosis:** Classifying medical images to detect diseases such as cancer or pneumonia.
2. **Fraud detection:** Identifying suspicious banking transactions in real-time.
3. **Voice recognition:** Converting audio to text in virtual assistants like Siri or Alexa.

**Examples of Unsupervised Learning applications:**
1. **Customer segmentation:** Grouping users according to their purchasing habits for personalized marketing campaigns.
2. **Topic analysis in texts:** Automatically discovering recurring themes in large document collections.
3. **Anomaly detection:** Identifying unusual patterns in industrial sensors for predictive maintenance.

These examples show how machine learning can add value across different sectors, helping to solve complex problems and optimize processes.

---

## 1. Supervised Learning

Supervised learning uses a dataset where inputs are paired with their desired outputs (or **"labels"**). It's like teaching a child by showing them an object (the input) and telling them what it is (the output or label). The goal of the model is to learn the function that maps the input to the output.

### 1.2 Classification

Classification models are fundamental in machine learning because they allow elements to be assigned to specific categories based on their characteristics. These models analyze input data and, using algorithms such as decision trees, logistic regression, or neural networks, determine which class each example belongs to. For instance, in a spam detection system, the model classifies emails as "spam" or "not spam" based on their content and other attributes.

The main characteristic of classification models is that the output is a **discrete category**, meaning the result belongs to a limited set of predefined classes. This differentiates them from regression models, where the output is a continuous value. Classification models are used in applications such as image recognition, medical diagnosis, and sentiment analysis, where it is essential to identify which group each analyzed data point belongs to.

![alt text](https://www.themachinelearners.com/wp-content/uploads/2021/01/1_aE8XLyApqvaQA9B7MWjjlA.png)

1. **Fraud detection in transactions:** Classifying financial transactions as fraudulent or legitimate based on spending patterns and user behavior.

2. **Medical diagnosis:** Classifying radiological images to determine if there is a presence of diseases such as pneumonia, tumors, or fractures.

3. **Sentiment analysis:** Classifying product reviews or social media comments as positive, negative, or neutral.

4. **Facial recognition:** Identifying and classifying people in images for security systems or identity verification.

5. **Content filtering:** Classifying web content or social media posts as appropriate or inappropriate for different audiences.

### 1.2 Regression

Regression models are fundamental in predictive analysis because they allow for estimating quantitative relationships between variables. Unlike classification models, which assign discrete categories, regression models generate a **continuous** output, making them ideal for tasks such as price prediction, demand estimation, or trend analysis. These models can be simple, like linear regression, where a linear relationship between variables is assumed, or more complex, such as polynomial regression, Ridge, Lasso, or even neural network-based regression, which capture non-linear relationships and multiple interactions.

The **continuous** output of a regression model means that the predicted value can take any number within a range, allowing for greater precision in contexts where results are not limited to discrete classes. For example, when predicting temperature, monthly income, or waiting time, a model that can adjust to subtle variations is required. Additionally, regression models allow for evaluating the impact of each independent variable on the dependent variable, making them powerful tools for interpretation and informed decision-making in scientific, economic, and operational environments.

![alt text](https://pub.mdpi-res.com/ijerph/ijerph-15-02907/article_deploy/html/images/ijerph-15-02907-g001.png?1570846772)

Regression models are fundamental in many practical applications where we need to predict continuous numerical values.

Some notable examples include:

1. **Real estate price prediction:** Estimating a property's value based on characteristics such as location, size, number of rooms, age, and nearby amenities.

![alt text](https://drek4537l1klr.cloudfront.net/serrano/v-4/Figures/image028.png)

2. **Sales forecasting:** Predicting a company's future revenues considering historical data, seasonality, market trends, and economic variables.

3. **Energy consumption estimation:** Calculating buildings' energy demand based on factors such as climate, insulation, occupancy, and installed systems.

4. **Crop yield prediction:** Estimating agricultural production according to variables such as precipitation, temperature, soil type, and fertilizers used.

5. **Environmental impact assessment:** Modeling the relationship between pollutant emissions and factors such as industrial activity, population, and control measures to predict future pollution levels.

These examples demonstrate how regression allows for informed decision-making in areas as diverse as finance, urban planning, sustainability, and agriculture, providing accurate numerical estimates based on data.

### 1.3 Algorithms

|Algorithm|Main Task Type|Brief Description|Key Hyperparameters|
|-----------------:|----------------:|-----------------:|---------------|
|Linear Regression|Regression|Models the relationship between a dependent variable (output) and one or more independent variables (inputs) by fitting the best straight line to the data.|No learning hyperparameters. Parameters are directly calculated (least squares). Learning rate is often only considered if Gradient Descent is used.|
|Logistic Regression|Classification|Uses the logistic function to estimate the probability that an instance belongs to a class. Despite its name, it is a binary classification model (two classes).|1. C (or λ): Inverse of regularization strength. Smaller values specify stronger regularization. 2. Penalty: Type of regularization applied (L1 or L2). 3. Solver: Algorithm to use in optimization (e.g., liblinear, saga, lbfgs).|
|Decision Trees|Classification / Regression|Create a model that predicts the value of a target variable (output) by dividing the training dataset into subsets based on the values of the features (inputs), forming a tree-like structure.|1. max_depth: Maximum depth of the tree. 2. min_samples_split: Minimum number of samples required to split an internal node. 3. criterion: Function to measure the quality of a split (e.g., gini or entropy for classification).|
|Random Forest|Classification / Regression|An ensemble method that builds multiple decision trees and combines their predictions to improve accuracy and avoid overfitting.|1. n_estimators: Number of trees in the forest. 2. max_features: Number of features to consider for the best split at each node. 3. max_depth: Maximum depth of each individual tree.|
|Support Vector Machine (SVM)|Classification / Regression|Finds an optimal hyperplane that separates classes in a high-dimensional space, maximizing the margin between the hyperplane and the closest data points (support vectors).|1. C: Regularization parameter. Penalizes classification errors. 2. kernel: Type of kernel function (linear, poly, rbf (Radial Basis Function), sigmoid). 3. gamma (γ): Parameter of the rbf kernel (defines how much influence a single training example has).|
|K-Nearest Neighbors (KNN)|Classification / Regression|A non-parametric algorithm that classifies a new point based on the majority class of its K nearest neighbors in the feature space.|1. n_neighbors (K): The number of neighbors to consider. 2. weights: Weighting function used in prediction (uniform or distance). 3. metric: Distance metric to use (euclidean, manhattan, minkowski).|
|Boosting|Classification / Regression|Creates a strong model from a sequence of weak models, correcting the errors of the previous model.|1. n_estimators (Number of weak models). 2. learning_rate (Weight of each model). 3. max_depth (If using weak trees).|
|XGBoost (eXtreme Gradient Boosting)|Classification / Regression|A highly efficient and popular boosting algorithm that iteratively improves a set of weak models (typically trees) to form a strong predictive model.|1. n_estimators: Number of boosting rounds (number of trees). 2. learning_rate (η): Learning rate or step size at each iteration. 3. max_depth: Maximum depth of each tree. 4. gamma (γ): Minimum loss reduction required for an additional split. 5. subsample: Fraction of random samples to use for training each tree.|

### 1.4 Validation Metrics

Validation metrics are fundamental tools for **evaluating model performance** in machine learning. They allow quantifying how well a model performs classification or regression tasks, helping to **compare different algorithms** and **adjust their parameters** to obtain better results.

![alt text](https://db0dce98.rocketcdn.me/es/files/2024/08/Schema-model_evaluation-42-42.png)

In classification, metrics such as accuracy, precision, recall, and F1-score help understand if the model correctly identifies classes, which is crucial in applications such as fraud detection (where minimizing false positives is important) or medical diagnosis (where reducing false negatives is vital).

![alt text](https://almablog-media.s3.ap-south-1.amazonaws.com/image_14_4f4fc2cf7d.png)

In regression, metrics such as mean squared error (MSE), mean absolute error (MAE), and the coefficient of determination (R²) measure the difference between predicted and actual values. For example, when predicting a house price, a low MSE indicates that the model makes estimates close to real values.

![alt text](https://miro.medium.com/1*5fnmYVHLTC8mGxybHm4XkA.png)

Selecting the appropriate metric depends on the problem and the impact of errors. Therefore, understanding and correctly applying these metrics is essential to develop robust and useful models in practice.

|Metric|Usage Type|How It's Calculated|Strengths and Weaknesses|
|-------|-----------|---------------|------------------------|
|Accuracy|Classification|Proportion of correct predictions over the total number of samples.|Easy to interpret, but can be misleading in imbalanced datasets.|
|Precision|Classification|TP / (TP + FP), where TP = true positives, FP = false positives.|Useful when the cost of false positives is high; may ignore false negatives.|
|Recall (Sensitivity)|Classification|TP / (TP + FN), where FN = false negatives.|Important when the cost of false negatives is high; may ignore false positives.|
|F1-Score|Classification|Harmonic mean between precision and recall: 2 * (Precision * Recall) / (Precision + Recall).|Balances precision and recall; useful in imbalanced data.|
|AUC-ROC|Classification|Area under the ROC curve, which compares the true positive rate vs. false positive rate.|Evaluates performance across all thresholds; robust in imbalanced data.|
|MSE (Mean Squared Error)|Regression|Average of the squares of the differences between actual and predicted values.|Heavily penalizes large errors; sensitive to outliers.|
|MAE (Mean Absolute Error)|Regression|Average of the absolute values of the differences between actual and predicted values.|Less sensitive to outliers than MSE; easy to interpret.|
|R² (Coefficient of Determination)|Regression|Proportion of variance explained by the model relative to the total variance.|Indicates overall fit; can be negative if the model is worse than the mean.|

---

## 2. Unsupervised Learning

Unsupervised learning works with **unlabeled** data. The model must find patterns, structures, or hidden relationships within the data by itself.

The main uses are:

![alt text](https://scikit-learn.org/0.18/_images/sphx_glr_plot_cluster_comparison_001.png)

* **Clustering**: Grouping similar data.
* **Association**: Discovering rules that describe relationships (example: market basket analysis).
* **Dimensionality Reduction**: Simplifying data.

### 2.1 Algorithms

|Algorithm|Main Task Type|Brief Description|Key Hyperparameters|
|-----------------:|----------------:|-----------------:|---------------|
|K-Means|Clustering|Divides data into K groups or clusters, where K is a predefined number. Assigns each point to the nearest centroid (central point).|1. n_clusters (K): The number of clusters to form. 2. init: Method for initializing centroids (k-means++ or random). 3. max_iter: Maximum number of iterations for the algorithm.|
|Hierarchical Clustering|Clustering|Builds a hierarchy of clusters. Can be agglomerative (starting with individual points and grouping them) or divisive (starting with a large cluster and dividing it).|1. n_clusters: The number of clusters to stop the process. 2. linkage: Connection criterion between sets of observations (ward, average, complete). 3. affinity (or metric): Distance metric used to calculate distances (euclidean, manhattan).|
|Density-Based Clustering|Clustering|Identifies clusters based on the density of data points. Good for finding clusters of arbitrary shapes and robust to noise (outliers).|1. eps (ϵ): The maximum distance between two samples to be considered neighbors. 2. min_samples: The number of samples (or points) in a neighborhood for a point to be considered a core point. 3. metric: Distance metric to use (euclidean, manhattan).|

### 2.2 Validation Metrics

|Metric|Usage Type|How It's Calculated|Strengths and Weaknesses|
|-------|-----------|---------------|------------------------|
|Silhouette Score|Clustering|Average of the difference between intra-cluster distance and distance to the nearest cluster.|Evaluates the separation and cohesion of clusters; requires knowing the clusters.|
|Davies-Bouldin Index|Clustering|Average of the ratio between intra-cluster dispersion and inter-cluster distance.|Lower value indicates better clustering; sensitive to clusters of different sizes.|
|Calinski-Harabasz Index|Clustering|Ratio between inter-cluster dispersion and intra-cluster dispersion.|Higher value indicates better clustering; favors compact and well-separated clusters.|
|Homogeneity|Clustering|Measures if each cluster contains only members of a single class.|Useful if true labels are known; not applicable in pure clustering.|
|Completeness|Clustering|Measures if all members of a class are in the same cluster.|Complements homogeneity; useful with true labels.|
|Explained Variance|Dimensionality Reduction|Proportion of total variance captured by the selected components.|Indicates how much information is preserved; doesn't measure interpretability.|

---

## 3. Machine Learning Challenges

### 3.1 Underfitting and Overfitting

![alt text](https://media.licdn.com/dms/image/v2/D4E22AQFLsYgMYO-H7Q/feedshare-shrink_800/B4EZkk_dW5KYAg-/0/1757262241175?e=2147483647&v=beta&t=FSDUbc3ZebNr4pO1yqRu6awd1VHCny45aOzxyFwMcoY)

**Underfitting** occurs when a model is too simple to capture relevant patterns in the data, resulting in poor performance on both training and test sets. **Overfitting** happens when the model is too complex and learns specific details or noise in the training set, losing generalization capability and showing high performance on training but low on testing. Neither of these scenarios is desirable, as they prevent the model from being useful on new data.

### 3.2 Class Imbalance

![alt text](https://cdn.sanity.io/images/31qskqlc/production/a680f0dd5ab72cd0dfb06effd8cdbfa0858ac6a8-850x647.webp?fit=max&auto=format)

**Class imbalance** appears when some classes are represented by many more samples than others. This can lead to the model ignoring minority classes, affecting accuracy and utility in critical applications (e.g., fraud detection or rare disease identification).

### 3.3 Data Quality and Quantity

The **quality** of data (presence of errors, missing values, noise) and insufficient **quantity** of data can limit the model's ability to learn useful patterns. Poor or scarce data often leads to unreliable models and misleading results.

### 3.4 Hyperparameter Tuning

**Hyperparameter tuning** involves exploring different configurations to find the combination that optimizes model performance. It's a way to explore solutions and avoid both underfitting and overfitting.

#### 3.4.1 Hyperparameter Tuning Algorithms

| Hyperparameter Tuning Method | How It Works | Strengths | Weaknesses |
|------------------------------------|---------------|------------|-------------|
| Grid Search | Explores all possible combinations of hyperparameters in a defined grid. | Exhaustive; guarantees finding the best combination in the defined space. | Computationally expensive; doesn't scale well with many hyperparameters. |
| Random Search | Selects random combinations of hyperparameters within defined ranges. | More efficient than grid search; can find good combinations quickly. | May miss the best combination; results depend on chance. |
| Bayesian Optimization | Models the objective function and selects hyperparameters based on previous results to maximize performance. | Efficient; requires fewer evaluations; learns from previous iterations. | More complex to implement; depends on the quality of the probabilistic model. |
| Hyperband | Uses sampling techniques and early stopping to efficiently allocate resources among configurations. | Fast; saves resources; good for large search spaces. | May prematurely discard good configurations; requires tuning of its own parameters. |
| Optuna | Automatic optimization algorithm that dynamically adjusts the search space and uses advanced techniques such as pruning. | Flexible; efficient; easy to integrate with modern frameworks. | May require advanced configuration; results depend on search space definition. |

### 3.5 Sampling and Validation Strategies

![alt text](https://towardsdatascience.com/wp-content/uploads/2021/02/1l3NEnB5bThd0uqxVe5Mbqg.jpeg)

To address these challenges, several strategies are employed:

| Method | Description | When to Use | Strengths | Weaknesses |
|----------|----------|----------|----------|----------|
| Data Partitioning | Separate the dataset into training and test data. | Always, to evaluate the model's generalization capability. | Easy to implement; quick evaluation. | May depend on the chosen partition; doesn't use all data for training. |
| Cross-Validation | Divides data into several partitions and alternates training/validation on each one. | When robust evaluation is needed and data is limited. | Reduces evaluation variance; uses all data for training and validation. | More computationally expensive; can be slow with large datasets. |
| Subsampling and Oversampling | Decrease or increase class samples to balance the dataset. | When there is class imbalance in the dataset. | Improves class balance; easy to apply. | May remove useful information (subsampling) or cause overfitting (oversampling). |
| SMOTE (Synthetic Minority Over-sampling) | Generates synthetic examples of the minority class. | When the minority class is very small and traditional oversampling is not sufficient. | Improves balance without duplicating data; reduces overfitting. | May generate unrealistic examples; requires careful application. |

These techniques help build more robust and reliable models, mitigating common problems in machine learning.
