#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "Eigen/Dense"
#include "measurement_package.h"
#include "kalman_filter.h"

class FusionEKF {
public:
  FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

private:
  void InitializeWithFirstMeasurement(const MeasurementPackage &measurement_pack);

  bool is_initialized_;
  long long previous_timestamp_;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
};

#endif /* FusionEKF_H_ */
