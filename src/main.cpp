//ndt map
#include <utils/pchandler.h>
#include <ndt_registration/ndt_registration_6d.h>

int main(){
    double grid_size = 0.15;

    ColorPC::Ptr srcpc(new ColorPC), trgpc(new ColorPC);
    ndt::NDTBaseMap srcmap = RGBD2NDTMap("/home/onepiece/25.jpg", "/home/onepiece/25.png", 
                                         srcpc, grid_size);

    ndt::NDTBaseMap trgmap = RGBD2NDTMap("/home/onepiece/20.jpg", "/home/onepiece/20.png",
                                         trgpc, grid_size);

    // visualizeSinglePC(trgpc);
    // visualizeSinglePC(srcpc);
    visualizePCPairs(srcpc, trgpc, Eigen::Matrix4f::Identity());


    NDTRegistration6D reg(trgmap.getPointMatrix().block(0,0,3,trgmap.getNumberOfGaussians()),
                          srcmap.getPointMatrix().block(0,0,3,srcmap.getNumberOfGaussians()),
                          trgmap.getPointCovariances(),
                          srcmap.getPointCovariances(),
                          trgmap.getPointMatrix(),
                          srcmap.getPointMatrix()
                          );

    Eigen::Matrix4d transform;
    double cost = reg.NDTD2DRegistration(transform);
    visualizePCPairs(srcpc, trgpc, transform.cast<float>());
    std::cout << transform <<std::endl;

    return 0;
}