/*********************************************************************
 *
 *  Copyright (c) 2014, Jeannette Bohg - MPI for Intelligent System
 *  (jbohg@tuebingen.mpg.de)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Jeannette Bohg nor the names of MPI
 *     may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
/* Header file that sets up the simulator and triggers the simulation 
 * of the kinect measurements and stores the results under a given directory.
 */
#ifndef SIMULATE_H
#define SIMULATE_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#endif 

#include <string.h>

#include <render_kinect/kinectSimulator.h>

static unsigned countf = 0;

namespace render_kinect {

class Simulate {
    public:

        Simulate(CameraInfo &cam_info, std::string out_path, 
                std::string object_name, std::string dot_path) 
            : out_path_(out_path), 
            width_(cam_info.width),
            height_(cam_info.height)
    {
        // allocate memory for depth image
        depth_im_ = cv::Mat(height_, width_, CV_32FC1);
        scaled_im_ = cv::Mat(height_, width_, CV_32FC1);

        object_model_ = new KinectSimulator(cam_info, object_name, dot_path);

        transform_ = Eigen::Affine3d::Identity();

        // extract file name from file path
        obj_name_ = object_name.substr(object_name.find_last_of("//") + 1);
        obj_name_ = obj_name_.substr(0, obj_name_.rfind("."));
    }

        ~Simulate() {
            delete object_model_;
        }

        void simulateMeasurement(const Eigen::Affine3d &new_tf, bool store_depth, 
                bool store_label, bool store_pcd, bool ply) {
            countf++;

            // update old transform
            transform_ = new_tf;

            // simulate measurement of object and store in image, point cloud and labeled image
            cv::Mat p_result;
            object_model_->intersect(transform_, point_cloud_, depth_im_, labels_);
            std::cout << "simulate success" << std::endl;

            // in case object is not in view, don't store any data
            // However, if background is used, there will be points in the point cloud
            // although they don't belong to the arm
            int n_vis = 2000;
            if (point_cloud_.rows<n_vis) {
                std::cout << "Object not in view. num of points: " << point_cloud_.rows << std::endl;
                return;
            }

            // store on disk
            if (store_depth) {
                std::stringstream lD;
                lD << out_path_ << "depth_orig" << std::setw(3) << std::setfill('0')
                    << countf << ".png";
                convertScaleAbs(depth_im_, scaled_im_, 255.0f);
                std::cout << "saving in " << lD.str().c_str() << std::endl;
                cv::imwrite(lD.str().c_str(), scaled_im_);
            }

            // store on disk
            if (store_label) {
                std::stringstream lD;
                lD << out_path_ << "labels" << std::setw(3) << std::setfill('0')
                    << countf << ".png";
                std::cout << "saving in " << lD.str().c_str() << std::endl;
                cv::imwrite(lD.str().c_str(), labels_);
            }

            //convert point cloud to pcl/pcd format
            if (store_pcd) {

#ifdef HAVE_PCL
                std::stringstream lD;
                lD << out_path_ << obj_name_ << std::setw(3)
                    << std::setfill('0') << countf;
                lD << ((ply) ? ".ply" : ".pcd");

                pcl::PointCloud<pcl::PointXYZ> cloud;
                // Fill in the cloud data
                // cloud.width = point_cloud_.rows;
                // cloud.height = 1;
                cloud.width = width_;
                cloud.height = height_;
                cloud.is_dense = false;
                cloud.points.resize(cloud.width * cloud.height);

                for (size_t r = 0; r < height_; r++)
                {
                    for (size_t c = 0; c < width_; c++)
                    {
                        int i = (r * width_) + c;
                        const float* point = point_cloud_.ptr<float>(i);
                        if (point[0] != -99)
                        {
                            cloud(c, r).x = point[0];
                            cloud(c, r).y = point[1];
                            cloud(c, r).z = point[2];
                            // blue graspblock
                            /* cloud(c, r).r = 10;
                            cloud(c, r).g = 10;
                            cloud(c, r).b = 200; */
                        } else if (!ply) {
                            const float nan = std::numeric_limits<double>::quiet_NaN();
                            cloud(c, r).x = nan; 
                            cloud(c, r).y = nan;
                            cloud(c, r).z = nan;
                        }
                    }
                }


                std::cout << "saving in " << lD.str() << std::endl;
                if (ply) {
                    if (pcl::io::savePLYFileASCII(lD.str(), cloud) != 0)
                        std::cout << "Couldn't store point cloud at " << lD.str() << std::endl;
                } else {
                    if (pcl::io::savePCDFileBinary(lD.str(), cloud) != 0)
                        std::cout << "Couldn't store point cloud at " << lD.str() << std::endl;
                }

#else
                std::cout << "Couldn't store point cloud since PCL is not installed." << std::endl;
#endif
            }
        }

        KinectSimulator *object_model_;
        cv::Mat depth_im_, scaled_im_, point_cloud_, labels_;
        std::string out_path_;
        std::string obj_name_;
        Eigen::Affine3d transform_; 
        int width_;
        int height_;
};

} //namespace render_kinect
#endif // SIMULATE_H
