### ML Practical Tip # 1 ###

One of the great qualities of random forests in sklearn is that it allows measuring the relative importance of each feature by looking at how much the tree nodes that use that feature reduce the impurity on a weighted average (across all trees in the forest). The weight of each node is the number of training samples associated with it. 

For example, the code below trains a Random Forest Classifier on the iris dataset and outputs each feature’s importance. It seems that the most important features are the petal length (44%) and width (42%), while sepal length and width are rather unimportant in comparison (11% and 2%, respectively).


![pika-2022-08-15T11_06_41 062Z](https://user-images.githubusercontent.com/72076328/188969870-2e62cc31-309d-4810-8627-088f7cf28000.png)

------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Practical Tip # 2 ###
If you are training a deep neural network and the loss is unstable (getting large changes from update to update), the loss goes to Nan during training or the model is not learning from the data this is a sign of exploding gradients. You can make sure by checking the weights if they are very large go to NaN values during training this will sign that there is an exploding gradient.

A simple but effective solution to this problem is 𝗴𝗿𝗮𝗱𝗶𝗲𝗻𝘁 𝗰𝗹𝗶𝗽𝗽𝗶𝗻𝗴, in which you clip the gradient if they pass a certain threshold.

In the Keras deep learning library, you can use gradient clipping by setting the 𝗰𝗹𝗶𝗽𝗻𝗼𝗿𝗺 or 𝗰𝗹𝗶𝗽𝘃𝗮𝗹𝘂𝗲 arguments on your optimizer before training.

Good default values are clipnorm = 1.0 and clipvalue = 0.5.


![gradclip](https://user-images.githubusercontent.com/72076328/188970232-120aed43-28e7-475a-afdf-c4edfab7a8aa.png)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 3 ### 

You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make. Also, this could be used as a backup if the new model fails for some reason. Likewise, you should keep a backup of the different versions of the datasets so that you can go back to it if the new version got corrupted for any reason.You can easily save Scikit-Learn models by using Python’s pickle module or using the joblib library, which is more efficient at serializing large NumPy arrays as shown in the code below. Also, you can save Keras's deep learning models in a similar way.

![pika-2022-08-17T11_19_16 604Z](https://user-images.githubusercontent.com/72076328/188971012-e4cbd3d2-89cd-4a5d-9073-ba6f65a02ae9.png)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 4 ###

One of the practical tips that might be a little bit strange is to split the data and leave out the test even before exploration and cleaning. The reason for that is that our brains are amazing pattern detection system, which means that it is highly prone to overfitting.

So if you look at the test set, you may stumble upon some seemingly interesting pattern in the test data that leads you to select a particular kind of Machine Learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic and you will launch a system that will not perform as well as expected. This is called **𝐝𝐚𝐭𝐚 𝐬𝐧𝐨𝐨𝐩𝐢𝐧𝐠 𝐛𝐢𝐚𝐬**

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Practical Tip # 5 ###
One of the main reasons bagging and pasting are famous, besides their good results, is that they can scale very well in an easy way. since the predictor can be trained in parallel as they are independent, you can train on multiple CPU cores or on multiple servers. The same can also be done in prediction. 

Scikit learn offers a  simple API for bagging and pasting with the  BaggingClassifier class. The parameter n_jobs defines the number of cores that can be used in both training and prediction. The code below n_jobs is -1, meaning use all the available cores.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Tip 6 ###
A big challenge with online learning* is that if bad data is fed to the system, the system performance will gradually decline. if it is a live system, your clients will notice. Bad data could come from a malfunctioning sensor on a robot or someone spamming a search engine to try to rank high in search results. To reduce the risk, you need to monitor your system closely and promptly switch learning off if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data for example using an anomaly detection algorithm. 

*Online learning: You train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Practical Tip # # 7 ###

If you would like to increase the speed of PCA you can set the svd_solver hyperparameter to "randomized" in Scikit-Learn. This will make it use a stochastic algorithm called Randomized PCA that quickly finds an approximation of the first d principal components. Its computational complexity is O(m × d2) + O(d3),
instead of O(m × n2) + O(n3) for the full SVD approach, so it is dramatically faster than full SVD when d is much smaller than n:

```
from sklearn.decomposition import PCA

𝐏𝐂𝐀(𝐧_𝐜𝐨𝐦𝐩𝐨𝐧𝐞𝐧𝐭𝐬=𝟏𝟎𝟎, 𝐬𝐯𝐝_𝐬𝐨𝐥𝐯𝐞𝐫="𝐫𝐚𝐧𝐝𝐨𝐦𝐢𝐳𝐞𝐝")

```

By default, svd_solver is actually set to "auto": Scikit-Learn automatically uses the randomized PCA algorithm if m or n is greater than 500 and d is less than 80% of m
or n, or else it uses the full SVD approach. If you want to force Scikit-Learn to use full SVD, you can set the svd_solver hyperparameter to "full".

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Practical Tip # 8 ###
If you would like to apply PCA to a large dataset you will probably face a memory error since the implementations of PCA requires the whole training set to fit in memory in order for the algorithm to run.

Fortunately, 𝗜𝗻𝗰𝗿𝗲𝗺𝗲𝗻𝘁𝗮𝗹 𝗣𝗖𝗔 (𝗜𝗣𝗖𝗔) algorithms have been developed to solve this problem: you can split the training set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is useful for large training sets, and also to apply PCA online.

The following code splits the training dataset into 100 mini-batches (using NumPy’s array_split() function) and feeds them to Scikit-Learn’s IncrementalPCA class to reduce the dimensionality of the training dataset down to 100 dimensions. Note that you must call the partial_fit() method with each mini-batch rather than the fit() method with the whole training set:
```
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=100)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
    
X_reduced = inc_pca.transform(X_train)

```
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 9 ###

If you would like to use a very slow dimension reduction method especially if your data is nonlinear such as Locally Linear Embedding (LLE) you can chain two dimension reduction algorithms to decrease the computational time of the LLE and still have almost the same result.

You can first apply PCA to your data to quickly get rid of a large number of useless dimensions and then use LLE. This will yield a similar performance as just using LLE but in a fraction of its time.

-------------------------------------------------------------------------
### ML Practical Tip # 10 ###

When splitting the dataset into training/ validations and test datasets it is important to avoid sampling bias especially if your dataset is imbalanced or not large enough. One way to avoid this is using stratified sampling: the population is divided into homogeneous subgroups called strata, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population.

You can apply this through the 𝐬𝐭𝐫𝐚𝐭𝐢𝐟𝐲 parameter in the 𝐬𝐤𝐥𝐞𝐚𝐫𝐧.𝐦𝐨𝐝𝐞𝐥_𝐬𝐞𝐥𝐞𝐜𝐭𝐢𝐨𝐧.𝐭𝐫𝐚𝐢𝐧_𝐭𝐞𝐬𝐭_𝐬𝐩𝐥𝐢𝐭 method as the following:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 11 ###

If you would like to visualize data with high dimensions a good method is 𝘁-𝗗𝗶𝘀𝘁𝗿𝗶𝗯𝘂𝘁𝗲𝗱 𝗦𝘁𝗼𝗰𝗵𝗮𝘀𝘁𝗶𝗰 𝗡𝗲𝗶𝗴𝗵𝗯𝗼𝗿 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴 (𝘁-𝗗𝗦𝗡𝗘).

𝘁-𝗗𝗦𝗡𝗘 reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. 𝘁-𝗗𝗦𝗡𝗘 uses a heavy-tailed Student-t distribution to compute the similarity between two points in the low-dimensional space rather than a Gaussian distribution, which helps to address the crowding and optimization problems.

For example, applying t-DSNE to the MNSIT dataset will results in 9 clusters as shown in the figure below:
![1665215052130](https://user-images.githubusercontent.com/72076328/194696370-00435ffd-69d5-4989-8e34-9218b36c0e4e.jpg)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 12 ###

Although creating a test set is theoretically quite simple: you can just pick some instances randomly of the dataset, and set them aside. There are some tips to make it more efficient one of them is Consistent Splitting.

Consistent splitting of the same instants as train and test will prevent the model from seeing the whole data set which is something we want to avoid to make sure that the models are reliable.

This can be done by using random.seed() when splitting or removing the data from the beginning and put it into a separate file.

However, both of these solutions will break the next time you fetch an updated dataset. A common better, and more reliable solution is to use each instance’s identifier to decide whether or not it should go in the test set (assuming instances have a unique and immutable identifier).

For example, you could compute a hash of each instance’s identifier and put that instance in the test set if the hash is lower or equal to 20% of the maximum hash value. This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset.

If your data is imbalanced and you will oversample or undersample certain classes in your data to overcome this problem. Do this to the training data only after splitting.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 13 ###


Do not oversample or undersample the test or validation dataset as they will make it nonrepresentative to the real world and will give unrealistic results.  

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

### ML Practical Tip # 14 ###


If you use SVM this tip can decrease time and computational complexity.

Since 𝐋𝐢𝐧𝐞𝐚𝐫𝐒𝐕𝐂 class is based on the 𝐥𝐢𝐛𝐥𝐢𝐧𝐞𝐚𝐫 library, which implements an optimized algorithm for linear SVMs. It does not support the kernel trick, but it 𝐬𝐜𝐚𝐥𝐞𝐬 𝐚𝐥𝐦𝐨𝐬𝐭 𝐥𝐢𝐧𝐞𝐚𝐫𝐥𝐲 with the number of training instances and the number of features: its training time complexity is roughly O(m × n).

The algorithm takes longer if you require very high precision. This is controlled by the tolerance hyperparameter ϵ (called 𝐭𝐨𝐥 in 𝐒𝐜𝐢𝐤𝐢𝐭-𝐋𝐞𝐚𝐫𝐧). In most classification tasks, the default tolerance is fine. This will be a perfect choice if you have a large dataset but are linearly separable.

On the other hand, the 𝐒𝐕𝐂 class is based on the 𝐥𝐢𝐛𝐬𝐯𝐦 library, which implements an algorithm that supports the kernel trick. The training time complexity is usually between O(m2 × n) and O(m3 × n).
Unfortunately, this means that it gets very slow when the number of training instances gets large (e.g., hundreds of thousands of instances).

This algorithm is perfect for 𝐜𝐨𝐦𝐩𝐥𝐞𝐱 𝐛𝐮𝐭 𝐬𝐦𝐚𝐥𝐥 𝐨𝐫 𝐦𝐞𝐝𝐢𝐮𝐦 𝐭𝐫𝐚𝐢𝐧𝐢𝐧𝐠 𝐬𝐞𝐭𝐬. However, it scales well with the number of features, especially with sparse features (i.e., when each instance has few nonzero features). In this case, the algorithm scales roughly with the average number of nonzero features per instance.


