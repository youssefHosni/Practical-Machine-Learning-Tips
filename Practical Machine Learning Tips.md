

### Tip 7 ###

If you would like to increase the speed of PCA you can set the svd_solver hyperparameter to "randomized" in Scikit-Learn. This will make it use a stochastic algorithm called Randomized PCA that quickly finds an approximation of the first d principal components. Its computational complexity is O(m × d2) + O(d3),
instead of O(m × n2) + O(n3) for the full SVD approach, so it is dramatically faster than full SVD when d is much smaller than n:

```
from sklearn.decomposition import PCA
𝐏𝐂𝐀(𝐧_𝐜𝐨𝐦𝐩𝐨𝐧𝐞𝐧𝐭𝐬=𝟏𝟎𝟎, 𝐬𝐯𝐝_𝐬𝐨𝐥𝐯𝐞𝐫="𝐫𝐚𝐧𝐝𝐨𝐦𝐢𝐳𝐞𝐝")

```

By default, svd_solver is actually set to "auto": Scikit-Learn automatically uses the randomized PCA algorithm if m or n is greater than 500 and d is less than 80% of m
or n, or else it uses the full SVD approach. If you want to force Scikit-Learn to use full SVD, you can set the svd_solver hyperparameter to "full".
