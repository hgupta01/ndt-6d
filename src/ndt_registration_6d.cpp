#include <ndt_registration_6d.h>

NDTRegistration6D::NDTRegistration6D(Eigen::MatrixXd fixed_pointmatrix,
                                     Eigen::MatrixXd movable_pointmatrix,
                                     std::vector<Eigen::Matrix3d> fixed_cov,
                                     std::vector<Eigen::Matrix3d> movable_cov,
                                     Eigen::MatrixXd fixed_pointmatrix6D,
                                     Eigen::MatrixXd movable_pointmatrix6D) {
  
  fixed_pointmatrix_ = fixed_pointmatrix;
  movable_pointmatrix_ = movable_pointmatrix;
  fixed_cov_ = fixed_cov;
  movable_cov_ = movable_cov;
  fixed_pointmatrix6D_ = fixed_pointmatrix6D;
  movable_pointmatrix6D_ = movable_pointmatrix6D;


  this->initial_guess_ = Eigen::Affine3d::Identity();
  this->result_ = Eigen::Affine3d::Identity();
  this->options_.line_search_direction_type = ceres::LineSearchDirectionType::BFGS;
  this->problem_ = nullptr;

  // See the FLANN documentation for expalantion of the parameters.
  kdtree.setData(this->fixed_pointmatrix6D_, true);
  kdtree.setIndexParams(flann::KDTreeSingleIndexParams(10));
  kdtree.setChecks(32);
  kdtree.setSorted(true);
  kdtree.setMaxDistance(1.0);
  kdtree.setThreads(8);
  kdtree.build();
}

NDTRegistration6D::~NDTRegistration6D() {
  if (problem_ != nullptr)
    delete problem_;
}


double NDTRegistration6D::NDTD2DRegistration(Eigen::Matrix4d& transformation) {
  int num_itr = 0;
  double final_cost;
  
  while (num_itr<50){
    Eigen::Matrix3d rotation = result_.matrix().block(0,0,3,3);
    Eigen::Vector3d translation = result_.matrix().block(0,3,3,1);
    Eigen::Matrix<double, -1, -1> transformed_mean = (rotation * movable_pointmatrix_).colwise() + translation;
    std::vector<Eigen::Matrix3d> transformed_cov;
    transformed_cov.resize(movable_cov_.size());

    for (unsigned int i=0; i<movable_cov_.size(); ++i){
      transformed_cov[i] = rotation.transpose() * movable_cov_[i] * rotation;
      movable_pointmatrix6D_.col(i).head(3) = transformed_mean.col(i);
    }

    Eigen::MatrixXi indices;
    Eigen::MatrixXd distances;
    kdtree.query(movable_pointmatrix6D_, 1, indices, distances);

    // std::cout << "---NDT D2D Registration---" << "\n";
    NDTD2D6DCostFunctor *d2dcost (new NDTD2D6DCostFunctor(fixed_pointmatrix_, transformed_mean, fixed_cov_,
                                      transformed_cov, indices, trim_ratio_, loss_value_));
    ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<NDTD2D6DCostFunctor, 1, 6>(d2dcost);
    ceres::FirstOrderFunction *cost = new NDTCostFunction(cost_function);

    this->problem_ = new ceres::GradientProblem(cost);

    double parameters[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double init_cost = d2dcost->getCost(parameters);
    // std::cout << "init cost: " << init_cost << "\n";
    affineToParameterArray(initial_guess_, parameters);

    ceres::Solve(options_, *problem_, parameters, &summary_);

    Eigen::Affine3d temp = parameterArrayToAffine(parameters);
    final_cost =  d2dcost->getCost(parameters);
    result_ = temp*result_;
    if (std::norm(final_cost-init_cost)<1e-3) break;
    num_itr++;
  }

  transformation = result_.matrix();
  return final_cost;

}

// extern "C" 
// {
//   NDTRegistration6D *NDTRegistration_map(const double *fixed_map,
//                                          const size_t n_cells_fixed,
//                                          const double *movable_map,
//                                          const size_t n_cells_movable) {
//     return new NDTRegistration6D(fixed_map, n_cells_fixed, movable_map, n_cells_movable);
//   }


//   double NDTRegistration_d2d(NDTRegistration6D *reg, double *transformation) {
//     return reg->NDTD2DRegistration(transformation);
//   }
// }
