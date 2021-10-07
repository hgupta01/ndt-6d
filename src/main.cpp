//ndt map
#include <yaml-cpp/yaml.h>
#include <utils/pchandler.h>
#include <ndt_registration/ndt_registration_6d.h>

int main(){

    YAML::Node config = YAML::LoadFile("../config.yaml");
    const std::string src_color = config["src_color"].as<std::string>();
    const std::string src_depth = config["src_depth"].as<std::string>();
    const std::string trg_color = config["trg_color"].as<std::string>();
    const std::string trg_depth = config["trg_depth"].as<std::string>();
    const double grid_size = config["grid_size"].as<double>();;


    ColorPC::Ptr srcpc(new ColorPC), trgpc(new ColorPC);
    ndt::NDTBaseMap srcmap = RGBD2NDTMap(src_color, src_depth, srcpc, grid_size);
    ndt::NDTBaseMap trgmap = RGBD2NDTMap(trg_color, trg_depth, trgpc, grid_size);

    NDTRegistration6D reg(trgmap.getPointMatrix().block(0,0,3,trgmap.getNumberOfGaussians()),
                          srcmap.getPointMatrix().block(0,0,3,srcmap.getNumberOfGaussians()),
                          trgmap.getPointCovariances(),
                          srcmap.getPointCovariances(),
                          trgmap.getPointMatrix(),
                          srcmap.getPointMatrix()
                          );

    Eigen::Matrix4d transform;
    double cost = reg.NDTD2DRegistration(transform);
    std::cout << transform <<std::endl;

    visualizePCPairs(srcpc, trgpc, Eigen::Matrix4f::Identity(), "before registration");
    visualizePCPairs(srcpc, trgpc, transform.cast<float>(), "after registration");
    

    return 0;
}