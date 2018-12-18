# Planar Robot Localization with the Extended Kalman Filter

## Getting Started

Download this repository to any directory on you machine. 

### Prerequisites

This software was developed on Matlab 2016, and requires the *Statistics and Machine Learning Toolbox*.

### Installing
No additonal installation procedures are required.

## Running the tests

Open Matlab. 
Navigate the *Current Folder* pane to the root directory of this repository. 
Run 'mainEKFLocalization.m'.
This entry point function will initialize the robot and environment state parameters and will orchestrate the discrete time simulation.
On each simulation iteration, the underlying 'EKFLocalization.m' function in invoked to produce the latest posterior density of state.

### Example Output
Shown below is an example snapshot of the random dynamic simulation which unfolds.
The left hand column describes the applied simulation parameters. 
These include the applied motion commands, and the error parameters associated with motion commands and measurements.
The large white square depicts the motion of the planar robot as well as the associated observer-estimate of state.
The equations governing motion and measurement processes are shown on the top left.
The Gaussian posterior density of the planar robot position (the third angle dimension is not shown) is depicted on the botton left.
![](./Figures/EKFLocalizationSample.png)

## References
Thrun S, Burgard W, Fox D. (2005). [*Probabilistic Robotics*](http://www.probabilistic-robotics.org/), Intelligent Robotics and Autonomous Agents. MIT Press.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


