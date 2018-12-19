%
% Interaction Primitives (ProMP-based)
%
% This code is a basic implementation of the Interaction Probabilistic
% Movement Primitives (ProMPs).
%
% This code implements only the part that makes the inference of the
% trajectory mean and variance. It does not implement the part that computes the
% feedback gains of the controller. It is assumed that the robot can track
% the mean of the trajectory distribution with its internal feedback
% controller.
%
% The data used in this example represents a total of 4 degrees-of-freedom
% (DoFs) where time is always normalized (extension of this code is to also
% scale time as needed).
% The first two trajectories represent X-Y coordinates of the human writing
% the letter "a". The third and fourth trajectories represent the X-Y
% coordinates of the robot writing the letter "b".
%
% There are a total of 50 examples of the writing pair "a-b".
% The member function "Weights.least_square_on_weights" computes the joint
% distribution of ProMP weights over the 50 examples. This becomes our 
% prior model of the interaction.
% 
% The "obs" structure contains the data that represents the measured states
% of the human. We use this data to condition the prior distribution.
% This computation is done recursively (one data each time). I believe in
% the Interaction Primitive paper (ICRA2014) they did this in batch.
% Recursion is better to do things on-line, using the real robot.
%
% The blue circles on the human plots represent the conditioned points,
% that is, the measured states from the human (for example, from a motion
% capture system). The code will pause after each conditioning so that you
% can look at the plots. Note that at this state the velocity is not being
% used in any sense, but it comes for free from the ProMP as the weights
% are the same for the position, with the only difference that the Gaussian
% features are differentiated.
%
%
% 23.04.2014 v00: initial code written. Guilherme
% 25.12.2014 v01: header update. Guilherme



clear; close all; clc; dbstop if error;
addpath('func/');
addpath('func_plot/');
addpath('Data/');       % data set created previously. Just load it.


% get traj distrib
dt       = 0.01;
[data]   = loadData(dt);

% plot individual joints
[data, h] = plot_nStates(data);

% put data in the ClassSystem-like format
data_ias = format_ias_data(data);

%[h] = plot2DwrapA(h, data);
 
% create the ProMP (ProMP) here
ProMP = regression_pmp(data_ias, 30, 'GaussianFeatures');

if 0 % check if trajectory was properly encoded by ProMP
    check_ProMP_encoding(ProMP, data);
end


% create a parameter structure that is useful for several functions
param.nTraj = data(1).nSize; % Trajectory size
%param.nFull = length(full.q_mean); % The full size of the correlated joints
param.nTotalJoints = 4;


%% Setup the measured data

% list here the joint positions  that are going to be observed. In other
% wods, these could be the human joints. The remaining joints are the ones
% estimated to control the robot.
param.observedJointPos = [1 2];

% list here the joint velocities that are going to be observed. Keep empty
% if no velocity is observerd, which is probably what is going to happen in practice
param.observedJointVel = [ ];    
                                 
% data set used as test data
dataSet = 1;      

% in KF context this says how much noise there is in the observation
stdev = 0.005; 

% create observation object here
obs = Observation(dataSet , stdev, param);            
              
% define the points in the data set to be measured, and update the object
observed_data_index = [ 5  25   50  80 ];              
obs.measuredIndexes(observed_data_index, data);
            
% plot test data
for j=1:4
    figure(h(j).q); grid off;
    plot(data(j).q(obs.set, :), SBLACK(2));  
    figure(h(j).qdot); grid off;
    plot(data(j).qdot(obs.set, :), SBLACK(2));  
end



[kf1] = mainloop(h, ProMP, param, obs);





 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

