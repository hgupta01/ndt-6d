//ndt map
#include <utils.hpp>
#include <ndt_registration_6d.h>


int main(){
    ndt::NDTBaseMap srcmap = RGBD2NDTMap("/home/onepiece/25.jpg",
                                         "/home/onepiece/25.png");

    ndt::NDTBaseMap trgmap = RGBD2NDTMap("/home/onepiece/20.jpg",
                                         "/home/onepiece/20.png");

    std::cout << "Nsrc:" << srcmap.getNumberOfGaussians() << " " << "Ntrg:" << trgmap.getNumberOfGaussians() <<std::endl;

    NDTRegistration6D reg(trgmap.getPointMatrix().block(0,0,3,trgmap.getNumberOfGaussians()),
                          srcmap.getPointMatrix().block(0,0,3,srcmap.getNumberOfGaussians()),
                          trgmap.getPointCovariances(),
                          srcmap.getPointCovariances(),
                          trgmap.getPointMatrix(),
                          srcmap.getPointMatrix()
                          );

    Eigen::Matrix4d transformation;
    double cost = reg.NDTD2DRegistration(transformation);
    std::cout << transformation <<std::endl;

    return 0;
}