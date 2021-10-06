// STL
#include <vector>
#include <random>
#include <memory>

// Eigen
#include "eigen3/Eigen/Dense"

namespace ndt
{
  class NDTBase3dCell
  {
    protected:
    Eigen::Matrix3d cov_; // point cov
    Eigen::Matrix<double, 6, 1> mean_; // point+color

    bool has_gaussian_;

    const unsigned char minN_ = 15;
    
    public:
    std::vector<Eigen::Matrix<double, 6, 1>> points;

    NDTBase3dCell(){
      this->cov_ = Eigen::Matrix3d::Zero();
      this->mean_ = Eigen::Matrix<double, 6, 1>::Zero();
      this->has_gaussian_ = false;
    };

    ~NDTBase3dCell(){};

    void addPoint(const Eigen::Matrix<double, 6, 1>& point){
      this->points.push_back(point);
    };


    void calculateMean(){
      for (size_t i = 0; i < points.size(); i++)
        mean_ += points[i];
      mean_ /= double(points.size());
      mean_.tail(3) *= 0.1; 
    };

    bool calculateGaussian(){
      if (points.size() < minN_)
      {
        points.clear();
        return false;
      }

      this->calculateMean();
      cov_ = this->computePointCovariance(mean_.head(3));

      bool invertible;
      Eigen::Matrix3d inv_cov;
      cov_.computeInverseWithCheck(inv_cov, invertible);
      if (invertible){
        has_gaussian_ = true;
        return true;
      }
      // if (cov_.determinant()>1e-20){
      //   has_gaussian_ = true;
      //   return true;
      // }

      points.clear();
      return false;
    };

    Eigen::Matrix3d computePointCovariance(const Eigen::Vector3d &mean){
      Eigen::Matrix3d cov_sum = Eigen::Matrix3d::Zero();
      for (size_t i = 0; i < points.size(); i++)
      {
        Eigen::Vector3d mean_diff = points[i].head(3) - mean;
        cov_sum += mean_diff * mean_diff.transpose();
      }
      return (1.0 / (double(points.size() - 1))) * cov_sum;
    };

    bool hasGaussian() const {return has_gaussian_;};
    
    const Eigen::Matrix3d& getCovariance() const{return cov_;};

    const Eigen::Matrix<double, 6, 1>& getMean() const{return mean_;};

  };
}
