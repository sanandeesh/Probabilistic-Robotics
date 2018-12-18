# Planar Robot Simultaneous Localization & Mapping with the Extended Kalman Filter

## Getting Started
These two files demonstrate the application of the Extended Kalman Filter to SLAM of a planar robot.
Briefly, the planar robot state-space is (x,y,theta) in an environment with landmark features existing at (x,y) points. 
It experiences random perturbations both in the motion it undergoes, and in the measurements it collects.
Whereas during localization the environment landmark coordinates were known a priori, in this case, neither position of the robot nor the environment landmarks were given.
*Optimal* estimation combines a priori known motion/measurement models and error parameters to minimize the expected squared error between the true state and estimate of it.
For a rigorous description of this algorithm as well as aggregate simulation results, please read the document, *EKFSLAM_ssk93.pdf*.
Next, download this repository to any directory on you machine. 

### Prerequisites

This software was developed on Matlab 2016, and requires the *Statistics and Machine Learning Toolbox*.

### Installing
No additonal installation procedures are required.

## Running the tests

Open Matlab. 
Navigate the *Current Folder* pane to the root directory of this repository. 
Run 'mainEKFSLAM.m'.
This entry point function will initialize the robot and environment state parameters and will orchestrate the discrete time simulation.
On each simulation iteration, the underlying 'EKFSLAM.m' function in invoked to produce the latest posterior density of state.

### Example Output
Shown below is an example snapshot of the random dynamic simulation which unfolds.
The left hand column describes the applied simulation parameters. 
These include the applied motion commands, and the error parameters associated with motion commands and measurements.
The large white square depicts the motion of the planar robot as well as the associated observer-estimate of state (i.e. *Localization*).
The true and estimated positions of the landmarks are shown on the top left (i.e. *Mapping*).
The covariance matrix of the SLAM posterior density is depicted on the botton left.
These include elements for robot state as well as landmark coordiantes.


## References
Thrun S, Burgard W, Fox D. (2005). [*Probabilistic Robotics*](http://www.probabilistic-robotics.org/), Intelligent Robotics and Autonomous Agents. MIT Press.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
