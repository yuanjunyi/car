**PID Controller Project**

#### 1. The effect of the P, I, D component of the PID algorithm

The P component is proportional to the cross track error. This is the main factor that decides the steering angle in the project.

As the car keeps moving, when it receives an updated steering angle it has already moved to a new position. This problem is resolved by introducing the D component, which can compensate the steering angle computed by the P component by considering the trend of the cross track error.

The I component is used to reduce the bias from the car itself. E.g. the car cannot execute perfectly the requested steering angle. This error will accumulate over time and the I component will address it.

#### 2. How the final hyperparameters were chosen

In this project, I manually tuned the coefficient Kp associated to the P component. I increased Kp until the car is steered enough when it passes the curve.

As Kp is increased, the car starts to oscillate. So I increased the coefficient Kd until the car is stable.

The Ki coefficient was not very useful in the project. I guess there is no bias in the simulator.
