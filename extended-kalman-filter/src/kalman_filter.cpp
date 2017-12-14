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

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  const VectorXd y = z - H_ * x_;
  const MatrixXd S = H_ * P_ * H_.transpose() + R_;
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
  hx << d, atan2(py, px), (px * vx + py * vy) / d;

  const VectorXd y = z - hx;
  const MatrixXd Hj = CalculateJacobian(x_);
  const MatrixXd S = Hj * P_ * Hj.transpose() + R_;
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

  if (abs(px) < 0.001 && abs(py) < 0.001) {
    cout << "Error in KalmanFilter::CalculateJacobian(): px and py are both zero" << endl;
    return Hj;
  }

  const double d = px * px + py * py;
  Hj << px / sqrt(d), py / sqrt(d), 0, 0,
        -py / d, px / d, 0, 0,
        py * (vx * py - vy * px) / pow(d, 1.5), px * (vy * px - vx * py) / pow(d, 1.5), px / d, py / d;

  return Hj;
}
