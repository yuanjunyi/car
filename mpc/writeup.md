**Model Predictive Control Project**

#### 1. The model

I use a kinematic model in the project. It has 6 states, including the position x and y, the orientation psi, the velocity v, the distance of vehicle from trajectory (cte), and the difference of vehicle orientation and trajectory orientation (epsi). It has 2 actuators, including steering angle delta and acceleration a.

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

The waypoints and vehicle position (x, y) received from the simulator are in map coordinate. As it's easier to calculate the cte in car coordinate, I converted the waypoints to car coordinate. As a result, vehiclue postion becomes (0, 0) and vehicle orientation becomes 0.

I fit a 3rd order polynomial to the converted waypoints and passed it to the MPC procedure.

In addition, the velocity received from the simulator is in mph. I converted it to m/s.

#### 4. Model Predictive Control with Latency
