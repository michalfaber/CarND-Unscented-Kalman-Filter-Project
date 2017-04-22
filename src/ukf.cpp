#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
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
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

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

  // Set state dimension
  n_x_ = 5;

  // Set augmented dimension
  n_aug_ = 7;

  // Define spreading parameter
  lambda_ = 3 - n_aug_;

  // Create vector for weights
  weights_ = VectorXd(2*n_aug_+1);

  // Set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  // Create predicted sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Create augmented mean vector
  Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // Create augmented mean vector
  X_aug = VectorXd(n_aug_);

  // Create augmented state covariance
  P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create matrices for radar measurements
  n_z_radar = 3;
  Zsig_radar = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
  Z_pred_radar = VectorXd(n_z_radar);
  S_radar = MatrixXd(n_z_radar, n_z_radar);
  R_radar = MatrixXd(n_z_radar,n_z_radar);
  Tc_radar = MatrixXd(n_x_, n_z_radar);
  R_radar << std_radr_*std_radr_, 0, 0,
      0, std_radphi_*std_radphi_, 0,
      0, 0,std_radrd_*std_radrd_;

  // Create matrices for lidar measurements
  n_z_laser = 2;
  Zsig_laser = MatrixXd(n_z_laser, 2 * n_aug_ + 1);
  Z_pred_laser = VectorXd(n_z_laser);
  S_laser = MatrixXd(n_z_laser, n_z_laser);
  R_laser = MatrixXd(n_z_laser,n_z_laser);
  Tc_laser = MatrixXd(n_x_, n_z_laser);
  R_laser << std_laspx_*std_laspx_, 0,
      0, std_laspy_*std_laspy_;

  // Not initialized yet
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {

    // initialize state convariance matrix

    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0.5, 0,
        0, 0, 0, 0, 0.5;

    float px = 0;
    float py = 0;
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      px = ro * cos(phi);
      py = ro * sin(phi);
    }
    else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_(0);
      py = meas_package.raw_measurements_(1);
    }

    x_ << px, py, 0, 0, 0;

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // elapsed time
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  // 1. PREDICTION

  Prediction(dt);

  // 2. UPDATE

  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

double UKF::SNormalizeAngle(double phi) {
  return atan2(sin(phi), cos(phi));
}

void UKF::GenerateSigmaPoints() {

  // Calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  // Set first column of sigma point matrix
  Xsig.col(0)  = x_;

  // Set remaining sigma points
  for (int i = 0; i < n_x_; i++) {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }
}

void UKF::AugmentedSigmaPoints() {

  // Create augmented mean state
  X_aug.head(5) = x_;
  X_aug(5) = 0;
  X_aug(6) = 0;

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.col(0)  = X_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug.col(i+1)       = X_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = X_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

void UKF::SigmaPointPrediction(double delta_t) {
  VectorXd x = VectorXd(5);
  VectorXd x_aug = VectorXd(7);
  VectorXd F = VectorXd(5);
  VectorXd Q = VectorXd(5);

  double px;
  double py;
  double v;
  double yaw;
  double yaw_acc;
  double var_acc;
  double noise_acc;
  double noise_yaw;
  double r1;
  double r2;
  double r4;

  for (int i = 0; i < Xsig_aug.cols(); i++) {
    px = Xsig_aug(0, i);
    py = Xsig_aug(1, i);
    v = Xsig_aug(2, i);
    yaw = Xsig_aug(3, i);
    yaw_acc = Xsig_aug(4, i);
    noise_acc = Xsig_aug(5, i);
    noise_yaw = Xsig_aug(6, i);

    x << px, py, v, yaw, yaw_acc;

    yaw = SNormalizeAngle(yaw);

    // Avoid division by zero

    if (fabs(yaw_acc) < 0.000001) {
      r1 = v*cos(yaw)*delta_t;
      r2 = v*sin(yaw)*delta_t;
      r4 = 0;
    } else {
      r1 = (v/yaw_acc) * (sin(yaw + yaw_acc*delta_t) - sin(yaw));
      r2 = (v/yaw_acc) * (cos(yaw) - cos(yaw + yaw_acc*delta_t));
      r4 = yaw_acc*delta_t;
    }

    F << r1, r2, 0, r4, 0;

    Q << 0.5 * delta_t*delta_t * cos(yaw)* noise_acc,
        0.5 * delta_t*delta_t * sin(yaw) * noise_acc,
        delta_t * noise_acc,
        0.5 * delta_t*delta_t * noise_yaw,
        delta_t * noise_yaw;

    Xsig_pred_.col(i) = x + F + Q;
  }
}

void UKF::PredictMeanAndCovariance() {

  // Predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  // Predicted state covariance matrix
  VectorXd x_diff;
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // State difference
    x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    x_diff(3) = SNormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::PredictRadarMeasurement() {

  // Transform sigma points into measurement space
  double p_x;
  double p_y;
  double v;
  double yaw;
  double v1;
  double v2;

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // Extract values for better readibility
    p_x = Xsig_pred_(0,i);
    p_y = Xsig_pred_(1,i);
    v = Xsig_pred_(2,i);
    yaw = Xsig_pred_(3,i);

    v1 = cos(yaw)*v;
    v2 = sin(yaw)*v;

    // Measurement model
    if (fabs(p_x) < 0.000001 && fabs(p_y) < 0.000001) {
      Zsig_radar(0,i) = 0;
      Zsig_radar(1,i) = 0;
      Zsig_radar(2,i) = 0;
    } else {
      Zsig_radar(0,i) = sqrt(p_x*p_x + p_y*p_y);
      Zsig_radar(1,i) = atan2(p_y, p_x);
      Zsig_radar(2,i) = (p_x*v1 + p_y*v2)/sqrt(p_x*p_x + p_y*p_y);
    }
  }

  // Mean predicted measurement
  Z_pred_radar.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    Z_pred_radar = Z_pred_radar + weights_(i) * Zsig_radar.col(i);
  }

  // Measurement covariance matrix S
  VectorXd z_diff;
  S_radar.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    z_diff = Zsig_radar.col(i) - Z_pred_radar;

    // Angle normalization
    z_diff(1) = SNormalizeAngle(z_diff(1));

    S_radar = S_radar + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S_radar = S_radar + R_radar;
}

void UKF::UpdateRadarState(MeasurementPackage meas_package) {

  // Calculate cross correlation matrix
  VectorXd z_diff;
  VectorXd x_diff;
  Tc_radar.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    // Residual
    z_diff = Zsig_radar.col(i) - Z_pred_radar;
    // Angle normalization
    z_diff(1) = SNormalizeAngle(z_diff(1));

    // State difference
    x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    x_diff(3) = SNormalizeAngle(x_diff(3));

    Tc_radar = Tc_radar + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc_radar * S_radar.inverse();

  // Update state mean and covariance matrix
  z_diff = meas_package.raw_measurements_ - Z_pred_radar;

  // calculate NIS
  NIS_radar_ = z_diff.transpose() * S_radar.inverse() * z_diff;

  // Angle normalization
  z_diff(1) = SNormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;

  P_ = P_ - K * S_radar * K.transpose();
}

void UKF::PredictLidarMeasurement() {
  // Transform sigma points into measurement space
  double px;
  double py;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Extract values for better readibility
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);

    Zsig_laser(0,i) = px;
    Zsig_laser(1,i) = py;
  }

  // Mean predicted measurement
  Z_pred_laser.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    Z_pred_laser = Z_pred_laser + weights_(i) * Zsig_laser.col(i);
  }

  // Measurement covariance matrix S
  VectorXd z_diff;
  S_laser.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // Residual
    z_diff = Zsig_laser.col(i) - Z_pred_laser;

    S_laser = S_laser + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S_laser = S_laser + R_laser;
}

void UKF::UpdateLidarState(MeasurementPackage meas_package) {

  // Calculate cross correlation matrix
  VectorXd z_diff;
  VectorXd x_diff;
  Tc_laser.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    // Residual
    z_diff = Zsig_laser.col(i) - Z_pred_laser;

    // State difference
    x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    x_diff(3) = SNormalizeAngle(x_diff(3));

    Tc_laser = Tc_laser + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc_laser * S_laser.inverse();

  // Update state mean and covariance matrix
  z_diff = meas_package.raw_measurements_ - Z_pred_laser;

  // calculate NIS
  NIS_laser_ = z_diff.transpose() * S_laser.inverse() * z_diff;

  x_ = x_ + K * z_diff;

  P_ = P_ - K * S_laser * K.transpose();
}

void UKF::PrintVec(VectorXd *v) {
  for (int i = 0; i < v->size(); i++) {
    cout << v->row(i) << " | ";
  }
  cout << endl << "-----" << endl;;
}

void UKF::PrintMatrix(MatrixXd *m) {
  for (int i = 0; i < m->rows(); i++) {
    VectorXd e = m->row(i);
    for (int j = 0; j < e.size(); j++) {
      cout << e(j) << " | ";
    }
    cout << endl;
  }
  cout << "-----" << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // 1. GENERATE SIGMA POINTS

  GenerateSigmaPoints();

  // 2. GENERATE AUGMENTED SIGMA POINTS - add noise vector to the state vector

  AugmentedSigmaPoints();

  // 3. PREDICT SIGMA POINTS

  SigmaPointPrediction(delta_t);

  // 4. CALCULATE STATE MEAN AND STATE COVARIANCE MATRIX

  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // 1. CALCULATE MEAN PREDICTED MEASUREMENT AND PREDICTED MEASUREMENT COVARIANCE S

  PredictLidarMeasurement();

  // 2. UPDATE STATE AND STATE COVARIANCE

  UpdateLidarState(meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // 1. CALCULATE MEAN PREDICTED MEASUREMENT AND PREDICTED MEASUREMENT COVARIANCE S

  PredictRadarMeasurement();

  // 2. UPDATE STATE AND STATE COVARIANCE

  UpdateRadarState(meas_package);
}
