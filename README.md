# Semesterthesis
In this folder the necessary code to replicate the results of my semesterthesis can be
found. This document will give a run down on how to use them to achieve this. First
make sure that the code is in the same directory as the data you are trying to cluster and
analyze. 
  * "clustering.py" takes the sleep and survey data and outputs "pearson_feature_survey_corr.csv", "feature-feature_matrix.csv", 'top10_feature_feature_heatmap.png', 'top10_feature_survey_heatmap.png', "cluster_survey_selected.csv" and "umap_clusters_survey.png". In order, these are: a csv for the correlation between the features and the survey questions, a csv for the correlation between the features, pngs of the heatmaps of these correlations, a csv with the final cluster results and finally a png with the UMAP visualization. In order to run this code make sure that
the data is correctly named in the file. As of now it should be slpwk survey.csv on line
57 and mhs survey stress.csv on line 187.
  * "chi_square.py" takes the demographics and cluster data and prints a table containing the results of the chi square test and Cramer's V of the age and gender with the cluster results
  * "cor_workout.py" takes the cluster and workout data and outputs "active_weeks_cluster_stats.csv" and "all_weeks_cluster_stats.csv". These are csv containing the mean and median of workouts metrics like amount of workouts or type of workout for each cluster. Active weeks only considers weeks, in which users have reported at least one workout. All weeks considers all user weeks.
  * "plot_clusterstress.py" takes the cluster and survey data and outputs 'boxplot_{metric}_by_cluster.png' and 'stats_by_cluster_mean_median_CI95.csv'. The pngs are boxplots for each metric by cluster and the csv contains the mean and median for the metrics for each cluster.

    
Same as for the clustering code, make out the names of the input
data and replace with your data. All files can be run on with python3 yourfile.py expect
for cor workout.py which needs the following cmd: python3 cor workout.py clusterdata.csv
workoutdata.csv -o results. If there are any uncertainties do not hesitate to reach out to
jonas64.ly@gmail.com.
