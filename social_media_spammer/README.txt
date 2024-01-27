FinalProject.py Readme

This code contains functionalities to preprocess the dataset, perform PCA analysis, and visualize data-related information using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, Flask, and Dash.

Installation: Install the dataset called 'follower_followee_Adjust', put it to the same path as the FinalProject.py and FinalProjectApp.py.
------------------------------------------------------------
Instructions: The FinalProject.py is for all static plots and Report and FinalProjectApp.py. is for dashboard.
------------------------------------------------------------
Warning: Larger datasets may result in slower operation

------------------------------------------------------------
Pre-processing Dataset:
- Reads the 'follower_followee_Adjust.csv' file.
- Cleans the dataset by dropping NaN values and displays the shape and statistical summary of the cleaned dataset.
-----------------------------------------------------------
PCA Analysis:
- Standardizes the dataset and performs PCA to reduce dimensionality.
- Evaluates explained variance ratios to determine the number of features to retain.
- Generates a line plot displaying the cumulative explained variance.
- Computes singular values and condition numbers for raw and transformed datasets.
-----------------------------------------------------------
- Describes statistical summary of columns in the dataset.
- Conducts normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, and Normal tests) on 'followings', 'fans', 'repost_num', 'Spam_num', and 'comment_num'.
- Applies Box-Cox transformation to handle non-Gaussian distributions in 'followings', 'fans', 'Spam_num', 'repost_num', and 'comment_num'.
- Determines and removes outliers using the Interquartile Range (IQR) method based on the 'Spam_num' column.

----------------------------------------------------------
Functionalities:
1. Line Plots:
   - Examines the linear relationship between 'followings', 'fans', 'other_post_num', 'repost_num', 'comment_num', and 'Spam_num'.
   - Illustrates the trends between user levels and numerical properties.

2. Bar Plots:
   - Displays bar plots of 'followings', 'fans', 'other_post_num', 'Spam_num', 'repost_num', and 'comment_num' categorized by 'gender'.
   - Provides bar plots for the top user levels.

3. Count Plots:
   - Presents count plots for 'isVIP', 'gender', 'level', and 'first_or_last'.

4. Pie Charts:
   - Exhibits pie charts showing the distribution of users by 'gender' and 'isVIP'.
   - Shows Spam_num distribution by 'gender' and 'isVIP'.

5. Pair Plot:
   - Displays pairwise relationships among different features categorized by 'gender'.

6. Histogram and KDE Plots:
   - Examines histograms with KDE for 'level', 'followings', and 'Spam_num'.
   - Provides histograms and KDEs divided by 'gender', 'isVIP', and 'first_or_last'.

7. QQ Plot:
   - Shows Quantile-Quantile plots for 'level', 'fans', 'followings', and 'Spam_num'.

8. Kernel Density Estimate (KDE) Plot:
   - Illustrates KDE plots for various features divided by 'gender', 'isVIP', and 'first_or_last'.
   - Adjusted KDE plots for better visualization of 'repost_num', 'comment_num', 'fans', 'level', and 'Spam_num'.

9. Two-dimensional KDE:
   - Displays contour plots demonstrating the relationship between 'level' and 'gender'.

10. Regression Plot:
   - Shows regression plots of 'followings' vs 'fans' categorized by 'gender'.
   - Regression plots of 'Spam_num' against 'fans', 'followings', 'repost_num', and 'comment_num' by 'gender' and 'isVIP'.

11. Multivariate Boxen Plot:
   - Displays boxen plots of 'isVIP' vs 'level', 'gender' vs 'isVIP', and 'gender' vs 'level'.

12. Area Plot:
   - Illustrates an area plot representing various features from the dataset.

13. Violin Plot:
   - Depicts violin plots showing relationships between 'gender' and 'level', 'gender' and 'isVIP', and 'isVIP' and 'level'.

14. Joint Plot with KDE and Scatter Representation:
    - Joint plots displaying the relationship between 'level' and 'Spam_num', 'followings' and 'Spam_num', 'repost_num' and 'Spam_num', 'comment_num' and 'Spam_num'.

15. Rug Plot:
    - Rug plots displaying data distribution for 'level' vs 'Spam_num', 'followings' vs 'Spam_num', 'fans' vs 'Spam_num', 'repost_num' vs 'Spam_num', and 'comment_num' vs 'Spam_num' divided by 'gender' and 'isVIP'.

16. Contour Plot:
    - Generates 3D and contour plots visualizing the relationship between 'followings' and 'fans' regarding 'level'.

17. Cluster Map:
    - Cluster maps showing clustered relationships between features within the dataset using the 'follower_f_5000' subset.
18.QQ PLot
19. Hexbin Plot
20. Strip Plot
21. Swarm Plot
22. Tables
-----------------------------------------------------------
Notice:

These visualizations offer further insights into the distribution, relationships, and normality of features in the MicroBlog dataset. Please refer to the code comments and outputs for detailed interpretations.



