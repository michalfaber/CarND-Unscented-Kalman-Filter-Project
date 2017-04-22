#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Check the validity of the inputs
  if (estimations.size() < 0.0001) {
    cout << "CalculateRMSE () Estimations vector is empty" << endl;
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    cout << "CalculateRMSE () Estimations and GroundTruth vectors differ in size" << endl;
    return rmse;
  }

  VectorXd residual;

  // Accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    residual = estimations[i]-ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse/estimations.size();

  // Calculate the squared root
  rmse = rmse.array().sqrt();

  // Return the result
  return rmse;
}
