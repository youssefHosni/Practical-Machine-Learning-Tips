### Tip 1 ###

One of the great qualities of random forests in sklearn is that it allows measuring the relative importance of each feature by looking at how much the tree nodes that use that feature reduce the impurity on a weighted average (across all trees in the forest). The weight of each node is the number of training samples associated with it. 

For example, the code below trains a Random Forest Classifier on the iris dataset and outputs each feature’s importance. It seems that the most important features are the petal length (44%) and width (42%), while sepal length and width are rather unimportant in comparison (11% and 2%, respectively).


![pika-2022-08-15T11_06_41 062Z](https://user-images.githubusercontent.com/72076328/188969870-2e62cc31-309d-4810-8627-088f7cf28000.png)

------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Tip 2 ###
If you are training a deep neural network and the loss is unstable (getting large changes from update to update), the loss goes to Nan during training or the model is not learning from the data this is a sign of exploding gradients. You can make sure by checking the weights if they are very large go to NaN values during training this will sign that there is an exploding gradient.

A simple but effective solution to this problem is 𝗴𝗿𝗮𝗱𝗶𝗲𝗻𝘁 𝗰𝗹𝗶𝗽𝗽𝗶𝗻𝗴, in which you clip the gradient if they pass a certain threshold.

In the Keras deep learning library, you can use gradient clipping by setting the 𝗰𝗹𝗶𝗽𝗻𝗼𝗿𝗺 or 𝗰𝗹𝗶𝗽𝘃𝗮𝗹𝘂𝗲 arguments on your optimizer before training.

Good default values are clipnorm = 1.0 and clipvalue = 0.5.


![gradclip](https://user-images.githubusercontent.com/72076328/188970232-120aed43-28e7-475a-afdf-c4edfab7a8aa.png)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Tip 3 ### 

You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make. Also, this could be used as a backup if the new model fails for some reason. Likewise, you should keep a backup of the different versions of the datasets so that you can go back to it if the new version got corrupted for any reason.You can easily save Scikit-Learn models by using Python’s pickle module or using the joblib library, which is more efficient at serializing large NumPy arrays as shown in the code below. Also, you can save Keras's deep learning models in a similar way.

![pika-2022-08-17T11_19_16 604Z](https://user-images.githubusercontent.com/72076328/188971012-e4cbd3d2-89cd-4a5d-9073-ba6f65a02ae9.png)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Tip 4 ###

One of the practical tips that might be a little bit strange is to split the data and leave out the test even before exploration and cleaning. The reason for that is that our brains are amazing pattern detection system, which means that it is highly prone to overfitting.

So if you look at the test set, you may stumble upon some seemingly interesting pattern in the test data that leads you to select a particular kind of Machine Learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic and you will launch a system that will not perform as well as expected. This is called **𝐝𝐚𝐭𝐚 𝐬𝐧𝐨𝐨𝐩𝐢𝐧𝐠 𝐛𝐢𝐚𝐬**

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Tip 5 ###
One of the main reasons bagging and pasting are famous, besides their good results, is that they can scale very well in an easy way. since the predictor can be trained in parallel as they are independent, you can train on multiple CPU cores or on multiple servers. The same can also be done in prediction. 

Scikit learn offers a  simple API for bagging and pasting with the  BaggingClassifier class. The parameter n_jobs defines the number of cores that can be used in both training and prediction. The code below n_jobs is -1, meaning use all the available cores.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Tip 6 ###
A big challenge with online learning* is that if bad data is fed to the system, the system performance will gradually decline. if it is a live system, your clients will notice. Bad data could come from a malfunctioning sensor on a robot or someone spamming a search engine to try to rank high in search results. To reduce the risk, you need to monitor your system closely and promptly switch learning off if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data for example using an anomaly detection algorithm. 

*Online learning: You train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Tips # 7 ###

If you would like to increase the speed of PCA you can set the svd_solver hyperparameter to "randomized" in Scikit-Learn. This will make it use a stochastic algorithm called Randomized PCA that quickly finds an approximation of the first d principal components. Its computational complexity is O(m × d2) + O(d3),
instead of O(m × n2) + O(n3) for the full SVD approach, so it is dramatically faster than full SVD when d is much smaller than n:

```
from sklearn.decomposition import PCA

𝐏𝐂𝐀(𝐧_𝐜𝐨𝐦𝐩𝐨𝐧𝐞𝐧𝐭𝐬=𝟏𝟎𝟎, 𝐬𝐯𝐝_𝐬𝐨𝐥𝐯𝐞𝐫="𝐫𝐚𝐧𝐝𝐨𝐦𝐢𝐳𝐞𝐝")

```

By default, svd_solver is actually set to "auto": Scikit-Learn automatically uses the randomized PCA algorithm if m or n is greater than 500 and d is less than 80% of m
or n, or else it uses the full SVD approach. If you want to force Scikit-Learn to use full SVD, you can set the svd_solver hyperparameter to "full".

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ML Tips # 8 ###
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

