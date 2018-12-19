# Probabilistic-Robotics
Bayesian filtering algorithms applied on robotic perceptual tasks based on selected chapters from Thrun et al. 2005.
These algorithms were implemented and evaluated with Matlab for educational purposes.  
This repo includes software for

- EKF Localization
- EKF SLAM
- Monte Carlo Localization

In all three cases, the observer is a planar robot for which the state space is (x,y,theta). 
It moves according to translational and angular velocity motion commands.
The environment consists of unambiguous landmarks located at (x,y) coordinates.
The robot collects observations of landmarks as (range, bearing) measurements.
Both the motion, and measurement processes are subject to additive Gaussian perturbations.
Nonetheless, these software packages demonstrate how Bayesian filters produce *optimal* estimates of robot state under increasing amounts of uncertainty.

## References
Thrun S, Burgard W, Fox D. (2005). [*Probabilistic Robotics*](http://www.probabilistic-robotics.org/), Intelligent Robotics and Autonomous Agents. MIT Press.
