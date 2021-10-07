#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <ndt_map/ndt_base_map.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPC;


void RGBDtoPCL(cv::Mat color_image, cv::Mat depth_image, ColorPC::Ptr pointcloud)
{
    // ColorPC::Ptr pointcloud(new ColorPC);
    int image_width = depth_image.cols;
    int image_height = depth_image.rows;

    float fx = 924.616;
    float fy = 924.584;
    float cx = 651.648;
    float cy = 355.379;

    float factor = 800;

    depth_image.convertTo(depth_image, CV_32F); // convert the image data to float type 

    if (!depth_image.data) {
        std::cerr << "No depth data!!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    pointcloud->width = image_width; //Dimensions must be initialized to use 2-D indexing 
    pointcloud->height = image_height;
    pointcloud->resize(pointcloud->width*pointcloud->height);

#pragma omp parallel for
    for (int v = 0; v < image_height; v++)
    {
        for (int u = 0; u < image_width; u++)
        {
            float Z = depth_image.at<float>(v, u) / factor;

            PointC p;
            p.z = Z;
            p.x = (u - cx) * Z / fx;
            p.y = (v - cy) * Z / fy;

            cv::Vec3b & color = color_image.at<cv::Vec3b>(v,u);
            p.r = (int)color[2];
            p.g = (int)color[1];
            p.b = (int)color[0];
            pointcloud->points[v*image_width+u] = p;
        }
    }
}

void PCFilter(ColorPC::Ptr pointcloud, ColorPC::Ptr cloud_filtered){
    ColorPC::Ptr temp (new ColorPC);
    ColorPC::Ptr temp2 (new ColorPC);

    // distance based filter
    pcl::PassThrough<PointC> passthrough;
    passthrough.setInputCloud(pointcloud);
    passthrough.setFilterFieldName("z");
    passthrough.setFilterLimits(0.0, 3.0);
    passthrough.filter(*temp);

    // Create the filtering object
    pcl::VoxelGrid<PointC> voxelfilter;
    voxelfilter.setInputCloud (temp);
    voxelfilter.setLeafSize (0.005, 0.005, 0.005);
    voxelfilter.filter (*temp2);

    // Radius outlier removal
    pcl::RadiusOutlierRemoval<PointC> outrem;
    outrem.setInputCloud(temp2);
    outrem.setRadiusSearch(0.01);
    outrem.setMinNeighborsInRadius (10);
    outrem.filter (*cloud_filtered);
}

ndt::NDTBaseMap RGBD2NDTMap(std::string cimg, std::string dimg, ColorPC::Ptr pc_, double grid_size=0.15){
    // Read images
    cv::Mat color_image = cv::imread(cimg);
    cv::Mat depth_image = cv::imread(dimg, -1);

    cv::Mat lab_image;
    cv::cvtColor( color_image, lab_image, cv::COLOR_BGR2Lab);
    
    // create Point Cloud
    ColorPC::Ptr labpc (new ColorPC), rgbpc (new ColorPC);
    RGBDtoPCL(lab_image, depth_image, labpc);
    RGBDtoPCL(color_image, depth_image, rgbpc);
    // std::cout << "original points: " << pc->size() << std::endl;

    // Filter Point Cloud
    ColorPC::Ptr pcfiltered (new ColorPC);
    PCFilter(labpc, pcfiltered);
    PCFilter(rgbpc, pc_);
    // std::cout << "filtered points: " << pcfiltered->size() << std::endl;


    ndt::NDTBaseMap ndtmap(grid_size);
    ndtmap.addPointCloud(pcfiltered);
    ndtmap.calculateGaussians();   
    return ndtmap; 
}

void visualizeSinglePC(ColorPC::Ptr pc){
    // Visualize Point Cloud
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    pcl::visualization::PointCloudColorHandlerRGBField<PointC> rgb(pc);
    viewer->addPointCloud<PointC> (pc, rgb, "pc");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "pc");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped())
        viewer->spinOnce(100);
}

void visualizePCPairs(ColorPC::Ptr srcpc, ColorPC::Ptr trgpc, Eigen::Matrix4f transform){

    // Executing the transformation
    ColorPC::Ptr transformed_cloud (new ColorPC);
    pcl::transformPointCloud (*srcpc, *transformed_cloud, transform);

    // Visualize Point Cloud
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    pcl::visualization::PointCloudColorHandlerRGBField<PointC> rgb(srcpc);
    viewer->addPointCloud<PointC> (transformed_cloud, rgb, "src");
    viewer->addPointCloud<PointC> (trgpc, rgb, "trg");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "src");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "trg");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped())
        viewer->spinOnce(100);
}