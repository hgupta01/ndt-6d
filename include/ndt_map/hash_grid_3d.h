// STL
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

// Boost
#include <boost/functional/hash.hpp>

// Eigen
#include "eigen3/Eigen/Dense"

namespace ndt
{
  struct EigenVector3dHash
  {
    std::size_t operator()(const Eigen::Vector3i &vector) const noexcept
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, vector[0]);
      boost::hash_combine(seed, vector[1]);
      boost::hash_combine(seed, vector[2]);
      return seed;
    }
  };

  template <class T>
  class HashGrid3d : public std::unordered_map<Eigen::Vector3i, T, EigenVector3dHash>
  {
    public:
    HashGrid3d(double grid_size = 1.0) : grid_size_(grid_size) {}

    T &getCell(const Eigen::Vector3d &point)
    {
      Eigen::Vector3i key_vector = getKeyVector(point);
      auto map_iter = this->find(key_vector);
      if (map_iter == this->end())
      {
        T new_cell;
        auto ret = this->insert(std::make_pair(key_vector, new_cell));
        map_iter = ret.first;
      }
      return map_iter->second;
    };


    protected:
    Eigen::Vector3i getKeyVector(const Eigen::Vector3d &point) const
    {
      return Eigen::Vector3i(std::floor(point[0] / grid_size_), std::floor(point[1] / grid_size_), std::floor(point[2] / grid_size_));
    };

    double grid_size_;
  };

} // namespace ndt
