
#ifndef NDT_COST_FUCNTION
#define NDT_COST_FUCNTION
// STL
#include <iostream>
#include <algorithm>
#include <typeinfo>

// Eigen
#include "eigen3/Eigen/Dense"
#include <eigen3/Eigen/Geometry>

// Ceres
#include "ceres/ceres.h"
#include "ceres/jet.h"

// OpenMP
#include <omp.h>


struct NDTD2D6DCostFunctor
{
protected:
  Eigen::MatrixXi indices_;
  Eigen::MatrixXd fixed_pointmatrix_;
  Eigen::MatrixXd movable_pointmatrix_;
  std::vector<Eigen::Matrix3d> fixed_covs_;
  std::vector<Eigen::Matrix3d> movable_covs_;

  int fixed_cols_;
  int movable_cols_;
  int gaussians_after_trim_;
  double trim_ratio_;
  double loss_value_;

  public:   
  NDTD2D6DCostFunctor(const Eigen::MatrixXd &fixed_pointmatrix,
                      const Eigen::MatrixXd &movable_pointmatrix,
                      const std::vector<Eigen::Matrix3d> &fixed_cov,
                      const std::vector<Eigen::Matrix3d> &movable_cov,
                      const Eigen::MatrixXi &indices,
                      double trim_ratio, 
                      double loss_value)
  {
    this->fixed_pointmatrix_ = fixed_pointmatrix;
    this->fixed_cols_ = fixed_pointmatrix_.cols();

    this->movable_pointmatrix_ = movable_pointmatrix;
    this->movable_cols_ = movable_pointmatrix_.cols();

    this->indices_ = indices;

    fixed_covs_ = fixed_cov;
    movable_covs_ = movable_cov;

    this->trim_ratio_ = trim_ratio;
    this->loss_value_ = loss_value;
    this->gaussians_after_trim_ = movable_cols_ * (1.0 - trim_ratio_);
  }


  template <typename T>
  bool operator()(const T *parameters, T *residuals) const
  {
    //Transform movable cloud
    Eigen::Matrix<T, 3, 3> rotation_matrix = getRotationMatrix(parameters);
    Eigen::Matrix<T, 3, 1> translation_vector = getTranslationVector(parameters);

    Eigen::Matrix<T, -1, -1> transformed_meanmatrix =
        (rotation_matrix * (movable_pointmatrix_).cast<T>()).colwise() + translation_vector;

    //Calculate distance matrix
    Eigen::Matrix<T, 1, -1> distance_matrix = Eigen::Matrix<T, 1, -1>::Constant(movable_cols_, T(0));
    Eigen::Matrix<T, -1, -1> fixed_meanmatrix = fixed_pointmatrix_.cast<T>();

#pragma omp parallel for num_threads(100) shared(distance_matrix)
    for(unsigned int i = 0; i < movable_cols_; ++i)
    {
      if (indices_(i) == -1) continue;

      Eigen::Matrix<T, 3, 3> movable_cov = rotation_matrix.transpose() * movable_covs_[i].cast<T>() * rotation_matrix;
      Eigen::Matrix<T, 3, 3> fixed_cov = fixed_covs_[indices_(i)].cast<T>();
        
      Eigen::Matrix<T, 3, 1> mu = transformed_meanmatrix.col(i) - fixed_pointmatrix_.col(indices_(i));
      distance_matrix(i) = (mu.transpose() * ( movable_cov + fixed_cov).inverse() * mu);
    }

    //Calculate cost for each movable cluster
    Eigen::Matrix<T, 1, -1> obj_movable = distance_matrix;
    
    //Sum cost for each movable cluster
    if (trim_ratio_ == 0.0)
    {
      if (loss_value_ == 0.0)
      {
        //Sum costs
        residuals[0] = obj_movable.sum();
      }
      else
      {
        //Apply lossfunction
        const T a_squared(loss_value_ * loss_value_);
        T cost = T(0.0);
        for (int col = 0; col < obj_movable.cols(); col++)
        {
          const T value = 1.0 - obj_movable(col) / a_squared;
          const T value_sq = value * value;
          const T use = T(obj_movable(col) <= a_squared);
          cost += a_squared / 3.0 * (1.0 - value_sq * value * use);
        }
        residuals[0] = cost;
      }
    }
    else
    {
      //Apply trim ratio
      std::sort(obj_movable.data(), obj_movable.data() + obj_movable.size());
      residuals[0] = obj_movable.leftCols(gaussians_after_trim_).sum();
    }

    return true;
  }

  template <typename T>
  Eigen::Matrix<T,3,3> getRotationMatrix(const T* parameters) const
  {
    Eigen::Matrix<T,3,3> rotation_matrix;
    const T s_1 = ceres::sin(parameters[2]);
    const T c_1 = ceres::cos(parameters[2]);
    const T s_2 = ceres::sin(parameters[1]);
    const T c_2 = ceres::cos(parameters[1]);
    const T s_3 = ceres::sin(parameters[0]);
    const T c_3 = ceres::cos(parameters[0]);
    rotation_matrix << 
      c_1*c_2, c_1*s_2*s_3-s_1*c_3, c_1*s_2*c_3+s_1*s_3,
      s_1*c_2, s_1*s_2*s_3+c_1*c_3, s_1*s_2*c_3-c_1*s_3,
      -s_2, c_2*s_3, c_2*c_3;
    return rotation_matrix;
  }

  template <typename T>
  Eigen::Matrix<T,3,1> getTranslationVector(const T* parameters) const
  {
    return Eigen::Matrix<T,3,1>(parameters[3], parameters[4], parameters[5]);
  }


  ceres::Jet<double, 6> getGradient(const ceres::Jet<double, 6>* parameters) const
  {
    ceres::Jet<double, 6> gradient;
    this->operator()(parameters, &gradient);
    return gradient;
  }

  double getCost(const double* parameters) const
  {
    double cost = .0;
    this->operator()(parameters, &cost);
    return cost;
  }
};


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


// void setInitGuess(const double *init_guess, Eigen::Affine3d &affine) {
//   affine.matrix() << init_guess[0], init_guess[1], init_guess[2], init_guess[3],
//       init_guess[4], init_guess[5], init_guess[6], init_guess[7], init_guess[8],
//       init_guess[9], init_guess[10], init_guess[11], init_guess[12],
//       init_guess[13], init_guess[14], init_guess[15];
// };

Eigen::Affine3d parameterArrayToAffine(const double *parameters) {
  Eigen::Affine3d result;
  result.setIdentity();
  result = Eigen::Translation<double, 3>(parameters[3], parameters[4],
                                         parameters[5]) *
           Eigen::AngleAxis<double>(parameters[0], Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxis<double>(parameters[1], Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxis<double>(parameters[2], Eigen::Vector3d::UnitZ());

  return result;
};

void affineToParameterArray(const Eigen::Affine3d &transformation,
                            double *parameters_out) {
  Eigen::AngleAxisd rotation(transformation.rotation());
  Eigen::Vector3d translation = transformation.translation();
  parameters_out[0] = rotation.axis()[0] * rotation.angle();
  parameters_out[1] = rotation.axis()[1] * rotation.angle();
  parameters_out[2] = rotation.axis()[2] * rotation.angle();
  parameters_out[3] = translation[0];
  parameters_out[4] = translation[1];
  parameters_out[5] = translation[2];
};

#endif