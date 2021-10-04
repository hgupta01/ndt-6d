#include <utils.hpp>

void matrix2Array(const Eigen::Matrix4d transform, double *array) {
  array[0] = transform(0, 0);
  array[1] = transform(0, 1);
  array[2] = transform(0, 2);
  array[3] = transform(0, 3);

  array[4] = transform(1, 0);
  array[5] = transform(1, 1);
  array[6] = transform(1, 2);
  array[7] = transform(1, 3);

  array[8] = transform(2, 0);
  array[9] = transform(2, 1);
  array[10] = transform(2, 2);
  array[11] = transform(2, 3);

  array[12] = transform(3, 0);
  array[13] = transform(3, 1);
  array[14] = transform(3, 2);
  array[15] = transform(3, 3);
}

void loadMap(const double *map, const size_t n_cells,
             Eigen::MatrixXd &points,
             Eigen::MatrixXd &points6d,
             std::vector<Eigen::Matrix3d> &covs){

  points.resize(3, n_cells);
  points6d.resize(6, n_cells);
  covs.resize(n_cells);

  size_t index = 0;

  for (unsigned int i=0; i<n_cells; ++i){
    points.col(i) << map[index], map[index + 1], map[index + 2];
    points6d.col(i) << map[index], map[index + 1], map[index + 2], map[index + 3], map[index + 4], map[index + 5];
    Eigen::Matrix3d cov;
    cov <<  map[index + 6], map[index + 7], map[index + 8], 
            map[index + 9], map[index + 10], map[index + 11],
            map[index + 12], map[index + 13], map[index + 14];
    covs[i] = cov;
    index += 15;
  }
}


void setInitGuess(const double *init_guess, Eigen::Affine3d &affine) {
  affine.matrix() << init_guess[0], init_guess[1], init_guess[2], init_guess[3],
      init_guess[4], init_guess[5], init_guess[6], init_guess[7], init_guess[8],
      init_guess[9], init_guess[10], init_guess[11], init_guess[12],
      init_guess[13], init_guess[14], init_guess[15];
}

Eigen::Affine3d parameterArrayToAffine(const double *parameters) {
  Eigen::Affine3d result;
  result.setIdentity();
  result = Eigen::Translation<double, 3>(parameters[3], parameters[4],
                                         parameters[5]) *
           Eigen::AngleAxis<double>(parameters[0], Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxis<double>(parameters[1], Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxis<double>(parameters[2], Eigen::Vector3d::UnitZ());

  return result;
}

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
}

