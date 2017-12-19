#include "FusionEKF.h"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

FusionEKF::FusionEKF() : is_initialized_(false), previous_timestamp_(0) {
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  ekf_.Init(VectorXd(4),     // state vector x
            MatrixXd(4, 4),  // state covariance matrix P
            MatrixXd(4, 4),  // transition matrix F
            H_laser_,        // measurement matrix H
            R_laser_,        // measurement covariance R
            R_radar_,        // measurement covariance R
            MatrixXd(4, 4)); // process covariance matrix Q
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!is_initialized_) {
    InitializeWithFirstMeasurement(measurement_pack);
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  const double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the state transition matrix F according to the new elapsed time.
  // Time is measured in seconds.
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt, 
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Update the process noise covariance matrix.
  // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  const double noise_ax = 9;
  const double noise_ay = 9;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt3 * dt;
  ekf_.Q_ << dt4/4*noise_ax, 0, dt3/2*noise_ax, 0,
             0, dt4/4*noise_ay, 0, dt3/2*noise_ay,
             dt3/2*noise_ax, 0, dt2*noise_ax, 0,
             0, dt3/2*noise_ay, 0, dt2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::InitializeWithFirstMeasurement(const MeasurementPackage &measurement_pack) {
  double px = 0, py = 0;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Convert radar from polar to cartesian coordinates and initialize state.
    const double range = measurement_pack.raw_measurements_[0];
    const double bearing = measurement_pack.raw_measurements_[1];
    px = range * cos(bearing);
    py = range * sin(bearing);
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
  }
  ekf_.x_ << px, py, 0, 0;
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
  is_initialized_ = true;
  previous_timestamp_ = measurement_pack.timestamp_;
}