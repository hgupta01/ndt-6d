#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <string>
#include <vector>
#include <iterator>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>

// Eigen
#include "eigen3/Eigen/Dense"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Ceres
#include "ceres/ceres.h"
#include "ceres/jet.h"

#include <ndt_base_map.h>

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPC;

class NDTCostFunction : public ceres::FirstOrderFunction {
private:
  std::unique_ptr<const ceres::CostFunction> cost_function_;

public:
  NDTCostFunction(const ceres::CostFunction *cost_function)
      : cost_function_(cost_function) {}

  bool Evaluate(const double *parameters, double *cost,
                double *gradient) const {
    if (!gradient)
      return cost_function_->Evaluate(&parameters, cost, NULL);
    return cost_function_->Evaluate(&parameters, cost, &gradient);
  }

  int NumParameters() const {
    return cost_function_->parameter_block_sizes().front();
  }
};

void matrix2Array(const Eigen::Matrix4d transform, double *array);

void loadMap(const double *map, const size_t n_cells,
             Eigen::MatrixXd &points,
             Eigen::MatrixXd &points6d,
             std::vector<Eigen::Matrix3d> &covs);

void setInitGuess(const double *init_guess, Eigen::Affine3d &affine);

Eigen::Affine3d parameterArrayToAffine(
      const double* parameters);

void affineToParameterArray(
      const Eigen::Affine3d& transformation, double* parameters_out);

void RGBDtoPCL(cv::Mat color_image, cv::Mat depth_image, ColorPC::Ptr pointcloud);
void PCFilter(ColorPC::Ptr pointcloud, ColorPC::Ptr cloud_filtered);

ndt::NDTBaseMap RGBD2NDTMap(std::string cimg, std::string dimg);

#endif