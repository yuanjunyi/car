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

  // if this is true, NIS will be computed and displayed to stdout
  compute_nis_ = false;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.45;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;
  
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

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
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

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
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
    const double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      UpdateRadar(meas_package);
    }
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
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.fill(0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
      Xsig_aug.col(i+1)        = x_aug + std::sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - std::sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
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

  x_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngleToPi(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
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
  const int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S(0, 0) += std_laspx_ * std_laspx_;
  S(1, 1) += std_laspy_ * std_laspy_;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngleToPi(x_diff(3));
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  if (compute_nis_)
  {
    const double NIS = z_diff.transpose() * S.inverse() * z_diff;
    cout << NIS << endl;
  }
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
  const int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);
    const double yawd = Xsig_pred_(4, i);

    const double r = sqrt(px*px+py*py);
    const double phi = atan2(py, px);
    const double rd = (px*cos(yaw)*v+py*sin(yaw)*v) / r;

    Zsig(0, i) = r;
    Zsig(1, i) = phi;
    Zsig(2, i) = rd;
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngleToPi(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S(0, 0) += std_radr_ * std_radr_;
  S(1, 1) += std_radphi_ * std_radphi_;
  S(2, 2) += std_radrd_ * std_radrd_;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngleToPi(x_diff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngleToPi(z_diff(1));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = NormalizeAngleToPi(z_diff(1));
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  if (compute_nis_)
  {
    const double NIS = z_diff.transpose() * S.inverse() * z_diff;
    cout << NIS << endl;
  }
}

double UKF::NormalizeAngleToPi(double angle) {
  while (angle < -M_PI)
    angle += 2 * M_PI;
  while (angle > M_PI)
    angle -= 2 * M_PI;
  return angle;
}

