#include "kalman_filter.h"
#include <iostream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in,
                        const MatrixXd &P_in,
                        const MatrixXd &F_in,
                        const MatrixXd &H_in,
                        const MatrixXd &R_laser_in,
                        const MatrixXd &R_rader_in,
                        const MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_laser_ = R_laser_in;
  R_rader_ = R_rader_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  const VectorXd y = z - H_ * x_;
  const MatrixXd S = H_ * P_ * H_.transpose() + R_laser_;
  const MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  const double px = x_(0);
  const double py = x_(1);
  const double vx = x_(2);
  const double vy = x_(3);

  VectorXd hx(3);
  const double d = sqrt(px * px + py * py);
  
  double bearing = atan2(py, px);
  while (bearing < -M_PI)
    bearing += 2.0 * M_PI;
  while (bearing > M_PI)
    bearing -= 2.0 * M_PI;

  hx << d, bearing, (px * vx + py * vy) / d;

  const VectorXd y = z - hx;
  const MatrixXd Hj = CalculateJacobian(x_);
  const MatrixXd S = Hj * P_ * Hj.transpose() + R_rader_;
  const MatrixXd K = P_ * Hj.transpose() * S.inverse();

  x_ = x_ + K * y;
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * Hj) * P_;
}

MatrixXd KalmanFilter::CalculateJacobian(const VectorXd & x_state)
{
  MatrixXd Hj(3, 4);
  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);

  const double c1 = px*px + py*py;
  const double c2 = sqrt(c1);
  const double c3 = c1 * c2;
  if (c1 < 0.0001) {
    cout << "Error in KalmanFilter::CalculateJacobian(): px and py are both zero" << endl;
    return Hj;
  }

  Hj << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
