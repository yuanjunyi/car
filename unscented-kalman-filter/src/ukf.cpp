#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    x_.fill(0);
    P_.fill(0);

    if (meas_package.sensor_type_ == LASER) {
      x_[0] = meas_package.raw_measurements_[0];
      x_[1] = meas_package.raw_measurements_[1];
    } else {
      x_[0] = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      x_[1] = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
    }

    P_(0, 0) = 1;
    P_(1, 1) = 1;
    P_(2, 2) = 1000;
    P_(3, 3) = 1000;
    P_(4, 4) = 1000;

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
  } else {

  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  x_aug.fill(0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_. n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; ++i) {
      Xsig_aug.col(i+1)       = x_aug + std::sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
      Xsig_aug.col(i+1+n_aug) = x_aug - std::sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    const double px = Xsig_aug(0, i);
    const double py = Xsig_aug(1, i);
    const double v = Xsig_aug(2, i);
    const double yaw = Xsig_aug(3, i);
    const double yawd = Xsig_aug(4, i);
    const double nu_a = Xsig_aug(5, i);
    const double nu_yawdd = Xsig_aug(6, i);

    double px_p = px;
    double py_p = py;
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    if (abs(yawd) > 0.001) {
      px_p += v / yawd * (sin(yaw+yawd*delta_t)-sin(yaw));
      py_p += v / yawd * (cos(yaw)-cos(yaw+yawd*delta_t));
    } else {
      px_p += v * delta_t * cos(yaw);
      py_p += v * delta_t * sin(yaw);
    }

    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}