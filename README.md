# Semesterthesis
In this folder the necessary code to replicate the results of my semesterthesis can be
found. This document will give a run down on how to use them to achieve this. First
make sure that the code is in the same directory as the data you are trying to cluster and
analyze. The first code to run is clustering.py. There it will create the features and select
the ones, which correlate to the survey the most. In order to run this code make sure that
the data is correctly named in the file. As of now it should be slpwk survey.csv on line
57 and mhs survey stress.csv on line 187. The code is documented so adding or replacing
features should be manageable. It should output a range of csv and png files. This can
be adjusted as wanted by commenting out the blocks, in which these outputs are created.
Included in this folder are the feature feature correlation, feature survey correlation and
the results of the clustering algorithm. Addtionally to the clustering code, there are python
files for performing the chi-square on the results of the clustering and code for analyzing
the created clusters. Same as for the clustering code, make out the names of the input
data and replace with your data. All files can be run on with python3 yourfile.py expect
for cor workout.py which needs the following cmd: python3 cor workout.py clusterdata.csv
workoutdata.csv -o results. If there are any uncertainties do not hesitate to reach out to
jonas64.ly@gmail.com.
