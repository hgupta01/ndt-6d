// NDT Cost Function
#include <utils.hpp>
#include <ndt_d2d_function.h>
#include "kdtree_flann.h"

class NDTRegistration6D
{
protected:
  Eigen::MatrixXd fixed_pointmatrix_;
  Eigen::MatrixXd movable_pointmatrix_;
  std::vector<Eigen::Matrix3d> fixed_cov_;
  std::vector<Eigen::Matrix3d> movable_cov_;
  Eigen::MatrixXd fixed_pointmatrix6D_;
  Eigen::MatrixXd movable_pointmatrix6D_;

  double loss_value_ = 0.2, trim_ratio_ = 0.4;

  Eigen::Affine3d initial_guess_;
  Eigen::Affine3d result_;
  ceres::GradientProblem* problem_;
  ceres::GradientProblemSolver::Summary summary_;
  ceres::GradientProblemSolver::Options options_;

  knn::KDTreeFlannd kdtree;

public:
  NDTRegistration6D(const double *fixed_map,
                    const size_t n_cells_fixed,
                    const double *movable_map,
                    const size_t n_cells_movable);

  ~NDTRegistration6D();

  double NDTD2DRegistration(double* transformation);
};