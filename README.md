1. We first import the data and set up lists for the different tasks
2. We then impute missing values the following way: priority descending - next available value > previous available value > global mean
3. The features we implemented are: mean, std dev., first and last observation and the number of missing values. The idea behind this was that we interpret the given measurements for the vitals as a time series and we only are interested in these things to make our prediction. We also have the number of missing measurements feature since we believe that having no measurement in the first 12 hours greatly correlates with no tests being ordered in the future as well.
4. For subtask 1&2 we decided to use a random forest classifier for each of the tests + sepsis
5. For task 3 we used a Ridge model and chose the optimal alpha using kfold crossvalidation
6. We combine all our predictions into a DataFrame and export it in the correct format
