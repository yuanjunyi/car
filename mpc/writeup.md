**Model Predictive Control Project**

#### 1. The model

I use a kinematic model in the project. It has 6 states, including the position x and y, the orientation psi, the velocity v, the distance of vehicle from trajectory (cross-track error), and the difference of vehicle orientation and trajectory orientation (epsi). It has 2 actuators, including steering angle delta and acceleration a.

The update equations are:
x1 = x0 + v0 * cos(psi0) * dt
y1 = y0 + v0 * sin(psi0) * dt
psi1 = psi0 + v0 / Lf * delta * dt, where Lf is the distance between the front of vehicle and the center of gravity
v1 = v0 + a * dt
cte1 = f(x0) - y0 + v0 * sin(epsi0) * dt
epsi1 = psi0 - arctan(f'(x0)) + v0 / Lf * delta * dt

#### 2. Timestep Length and Elapsed Duration (N & dt)

N * dt decides the prediction horizon, which is the duration over which future predictions are made. N * dt should be large enough so that the model could predict a meaningful trajectory. Meanwhile, N * dt should also not be too large because the environment will change too much so that it won't make sense to predict any further into the future.

#### 3. Polynomial Fitting and MPC Preprocessing

The waypoints and vehicle position (x, y) received from the simulator are in map coordinate. As it's easier to calculate the cte in car coordinate, I converted the waypoints to car coordinate (see line #80 to #86 in main.cpp). As a result, vehicle position and orientation also need to be converted to the car coordinate before being passed to the MPC controller. The vehicle position in car coordinate is (0, 0) and the orientation is 0.

I fit a 3rd order polynomial to the converted waypoints and passed it to the MPC controller.

In addition, the velocity received from the simulator is in mph. I converted it to m/s.

#### 4. Model Predictive Control with Latency

There is a latency between the moment an actuator command is sent and the moment this command is effective. During this latency, the car has already moved to a new position and the command computed based on the previous position is less accurate. In order to overcome it, I compute the new position after the latency and pass it to the MPC as the initial state (see line #114 to #116).

I also filtered the waypoints which are behind the new position so that they are not used in the MPC controller.

After this preprocessing, MPC will predict based on the new position. When the actuator command becomes effective, it matches with the car position.