// STL
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

// Boost
#include <boost/functional/hash.hpp>

// Eigen
#include "eigen3/Eigen/Dense"

// NDT
#include "ndt_map/hash_grid_3d.h"
#include "ndt_map/ndt_base_3d_cell.h"

namespace ndt
{
  class NDTBaseMap
  {
    protected:
    HashGrid3d<NDTBase3dCell> map_;

    public:
    NDTBaseMap(double grid_size)
    : map_(grid_size)
    {
    };

    ~NDTBaseMap(){
      this->map_.clear();
    };

    void addPoint(const Eigen::Matrix<double, 6, 1>& point){
      map_.getCell(point.head(3)).addPoint(point);
    };

    void addPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud){
      for(unsigned int i = 0; i < pointcloud->size(); i++)
      {
        Eigen::Matrix<double, 6, 1> point;
        point.head(3) = pointcloud->at(i).getVector3fMap().cast<double>();
        point.tail(3) = pointcloud->at(i).getRGBVector3i().cast<double>();
        addPoint(point);
      }
    };

    void calculateGaussians(){
      for(auto map_iter = map_.begin(); map_iter != map_.end(); map_iter++) 
        map_iter->second.calculateGaussian();
    };

    Eigen::MatrixXd getPointMatrix() {
      Eigen::MatrixXd mean_matrix(6,this->getNumberOfGaussians());
      unsigned int index = 0;
      for(auto map_iter = map_.begin(); map_iter != map_.end(); map_iter++)
      {
        if (map_iter->second.hasGaussian()){
          mean_matrix.col(index) = map_iter->second.getMean();
          index++;
        }
      }
      return mean_matrix;
    };

    std::vector<Eigen::Matrix3d> getPointCovariances(){
      std::vector<Eigen::Matrix3d> covs;
      covs.resize(this->getNumberOfGaussians());
      unsigned int index = 0;
      for(auto map_iter = map_.begin(); map_iter != map_.end(); map_iter++)
      {
        if (map_iter->second.hasGaussian()){
          covs[index] = map_iter->second.getCovariance();
          index++;
        }
      }
      return covs;
    }
    
    
    // Getters and setters
    unsigned int getNumberOfGaussians() const{
      unsigned int nb_gaussians = 0;
      for (auto map_iter = map_.begin(); map_iter != map_.end(); map_iter++)
      {
        if (map_iter->second.hasGaussian())
        {
          nb_gaussians++;
        }
      }
      return nb_gaussians;
    };
  };
}