# TensorFlow-Clustering
How to make Clusters with TensorFlow


Clustering
Now that we've covered regression and classification it's time to talk about clustering data!

Clustering is a Machine Learning technique that involves the grouping of data points. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. (https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

Unfortunalty there are issues with the current version of TensorFlow and the implementation for KMeans. This means we cannot use KMeans without writing the algorithm from scratch. We aren't quite at that level yet, so we'll just explain the basics of clustering for now.

Basic Algorithm for K-Means.
Step 1: Randomly pick K points to place K centroids
Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
Step 4: Reassign every point once again to the closest centroid.
Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.
Please refer to the video for an explanation of KMeans clustering.


K-centroids are where clusters exist
k = 3
placement is close to clusters but completely random

for every data point find Euclidean distance or manhatten distance to all the centroids
data point is linked to closest centroid
data point 1 is data point 1 because it is closest to centroid 1 out of 3
every data point applies this
after this centroids are moved to middle of data points also known as Center of Mass that has same numbers

Repeat process and reassign all points to new closest centroid
keep repeating process until none of the data pont's centroid numbers are chaging

When cetroids are in middle of data point's numbers as much as possible it is now called Cluster

for new data points a prediction is made to which cluster it is a part of by plotting data point and finding which cluster centroid it is closest to
this can be done for any new data point
output is label number of cluster

number of clusters has to be defined inside a variable K
Some algorithms could determine best amount of clusters for specific data









Hidden Markov Models


Deals with probability distributions
Predict wheather on any given day with probability of different events occurring.
Hidden Markov Model allows future predictions of weather given probabilities discovered.
Sometimes user might have huge data set and calulate probability of things occurring based on data set.

Hot days and cold days are hidden stats because they aren't accessed or looked at while interacting with model.
What is looked at is observations.
an observable probability is weather being between 5-15 degrees celsius with an average temp of 11 degrees.
is previous usage of data what was used was 100 or 1,000s of entries of row or data points for model to train, Hidden Markov Models does not use this.
All that is needed is constant values for probability, transition distribution and oversevation distribution.

States, Observations and Transitions
States like Warm cold, High low, red or blue
could have only 1 state
They're called hidden because they aren't directly observed

Observation
fact that during hot day 100% true tim is happy although what could be observed is 80% of time Tim is happy and 20% Tim is sad.
These are what are known as different observations of each state and probabilities of each state occurrring

Outcome 
Means there's no probaility because there's 100% chance of event occurring.

Transitions
Each state has probability to find likelihood of transitioning to different state.
Example hot day percentage chance that next day would be a cold day and vice versa.
So there's a probability of transitioning into different state and for each state could transition into every other state or defined set of states given certain probaility

Example
Hot day - has 20% of transitioning to cold day and an 80% chance of transitioning to another hot day
Cold day - 30% chance of transitioning to hot day and a 70% chance of transitioning to another cold day

Ovbservation
Each day has a list of Observations called States
S1 = Hot day (name doesn't matter)
S2 = Cold day

Transition probability is known but now need Observations proabbility or distribution for this.
Hot day - Obervationn is temp could be between 15 and 25 degrees Celsius with average temp of 20

Observation:
Mean: (average) = 20
Distribution: Min: 15, Max: 25 (standard deviation)

Standard Deviation
Mean is middle point or most common event that could occur
There's a probability of hitting different temps when moving to left and right of value.
On deviation curve somewhere on left is 15 and on right 25
these numbers are where the end of curve is located
Model figures out things to do with this information

Cold day:
Distribution - Mean: 5, Min: -5, Max: 15
This is dealing with standard deviation 

Straight percentage observations can also be used for exmaple:
20% chance Tim is happy or 80% cahnce Tim is sad.
Probabilities that could be had as obseervation probabilities in model.

Point of Hidden Markov Model is to predict future events based on past events.
So once proability distribution is known and want to predict weather for next week, model could be used to do so.




"The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional) probability distribution []. Transitions among the states are governed by a set of probabilities called transition probabilities." (http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html)

A hidden markov model works with probabilities to predict future events or states. In this section we will learn how to create a hidden markov model that can predict the weather.

This section is based on the following TensorFlow tutorial. https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel

Data
Let's start by discussing the type of data we use when we work with a hidden markov model.

In the previous sections we worked with large datasets of 100's of different entries. For a markov model we are only interested in probability distributions that have to do with states.

We can find these probabilities from large datasets or may already have these values. We'll run through an example in a second that should clear some things up, but let's discuss the components of a markov model.

States: In each markov model we have a finite set of states. These states could be something like "warm" and "cold" or "high" and "low" or even "red", "green" and "blue". These states are "hidden" within the model, which means we do not direcly observe them.

Observations: Each state has a particular outcome or observation associated with it based on a probability distribution. An example of this is the following: On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.

Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state. An example is the following: a cold day has a 30% chance of being followed by a hot day and a 70% chance of being follwed by another cold day.

To create a hidden markov model we need.





Centeroid stands for where cluster is defined

States
Observation Distribution
Transition Distribution
For our purpose we will assume we already have this information available as we attempt to predict the weather on a given day.


%tensorflow_version 2.x  # this line is not required unless you are in a notebook

!pip install tensorflow_probability==0.8.0rc0 --user --upgrade

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf


Weather Model
Taken direclty from the TensorFlow documentation (https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel).

We will model a simple weather system and try to predict the temperature on each day given the following information.

Cold days are encoded by a 0 and hot days are encoded by a 1.
The first day in our sequence has an 80% chance of being cold.
A cold day has a 30% chance of being followed by a hot day.
A hot day has a 20% chance of being followed by a cold day.
On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
If you're unfamiliar with standard deviation it can be put simply as the range of expected values.

In this example, on a hot day the average temperature is 15 and ranges from 5 to 25.

To model this in TensorFlow we will do the following.


tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())



Conclusion
So that's it for the core learning algorithms in TensorFlow. Hopefully you've learned about a few interesting tools that are easy to use! To practice I'd encourage you to try out some of these algorithms on different datasets.

Sources
Chen, James. “Line Of Best Fit.” Investopedia, Investopedia, 29 Jan. 2020, www.investopedia.com/terms/l/line-of-best-fit.asp.
“Tf.feature_column.categorical_column_with_vocabulary_list.” TensorFlow, www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list?version=stable.
“Build a Linear Model with Estimators  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/estimator/linear.
Staff, EasyBib. “The Free Automatic Bibliography Composer.” EasyBib, Chegg, 1 Jan. 2020, www.easybib.com/project/style/mla8?id=1582473656_5e52a1b8c84d52.80301186.
Seif, George. “The 5 Clustering Algorithms Data Scientists Need to Know.” Medium, Towards Data Science, 14 Sept. 2019, https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68.
Definition of Hidden Markov Model, http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html.
“Tfp.distributions.HiddenMarkovModel  :   TensorFlow Probability.” TensorFlow, www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel.



%tensorflow_version 2.x  # this line is not required unless you are in a notebook

!pip install tensorflow_probability==0.8.0rc0 --user --upgrade #restart runtime after this command and continue on to imports

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time that deals with probability
import tensorflow as tf #make sure version above is compatible with tensorFlow

tfd = tfp.distributions  # making a shortcut for later on, tensorFlow probability distribution model
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above, probability of 80% and 20%. First day in sequence has 80% chance of being cold and 20% chance after
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], #2 states so 2 probabilities of landing on each state at beginning of sequence, transition probability. 70% chance of cold day again and 30% chance of hot day
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above. when doing standard deviation loc stands for average or mean - 0 on hot day and 15 on cold day. Standard deviation on cold day is 5 so range from -5 to 5 degrees and on hot day is 10 so range from 5 to 25 degrees and average temp is 15.

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7) #how many days to predict so number of times to step through probability cycle and run model.


mean = model.mean() #calculates probability from model. model.mean is partially defined Tensor. Tensors are partially defined computations. to get value from this use lines below.

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  #create new session in TensorFlow. use to run session in new version of TensorFlow. sess keyword doesn't matter
  print(mean.numpy()) #run this part of graph to print out value. mean.numpy get value from mean = model.mean().
Run:
[12.       11.1      10.83     10.748999 10.724699 10.71741  10.715222]
Starts at 12 degrees and following temperatures are for next days
No training so calculations are same every run.


Changing values

%tensorflow_version 2.x  # this line is not required unless you are in a notebook
!pip install tensorflow_probability==0.8.0rc0 --user --upgrade

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.5, 0.5])  # Refer to point 2 above, 50% and 50% winds up making higher temp prediction. switching values above changes temperature lower
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)


More days this goes on less accurate it becomes like predicting wheather year in advance
