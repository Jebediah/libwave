#include "../include/kitti_utility_methods.hpp"

namespace wave {

namespace plot = matplotlibcpp;

using f64_seconds = std::chrono::duration<double>;

void LoadSensorParameters(const std::string &path, const std::string &filename, wave::RangeSensorParams &senparams) {
    wave::ConfigParser parser;
    parser.addParam("rings", &(senparams.rings));
    parser.addParam("sigma_spherical", &(senparams.sigma_spherical));
    parser.addParam("elevation_angles", &(senparams.elevation_angles));

    parser.load(path + filename);
}

void LoadParameters(const std::string &path, const std::string &filename, wave::LaserOdomParams &params) {
    wave::ConfigParser parser;
    parser.addParam("Qc", &(params.Qc));
    parser.addParam("num_trajectory_states", &(params.num_trajectory_states));
    parser.addParam("n_window", &(params.n_window));
    parser.addParam("opt_iters", &(params.opt_iters));
    parser.addParam("max_inner_iters", &(params.max_inner_iters));
    parser.addParam("diff_tol", &(params.diff_tol));
    parser.addParam("solver_threads", &(params.solver_threads));
    parser.addParam("robust_param", &(params.robust_param));
    parser.addParam("max_correspondence_dist", &(params.max_correspondence_dist));
    parser.addParam("max_planar_residual_val", &(params.max_planar_residual_val));
    parser.addParam("max_linear_residual_val", &(params.max_linear_residual_val));
    parser.addParam("min_residuals", &(params.min_residuals));
    parser.addParam("n_ring", &(params.n_ring));

    LoadSensorParameters(path, "sensor_model.yaml", params.sensor_params);

    parser.addParam("elevation_tol", &(params.elevation_tol));
    parser.addParam("max_planar_dist_threshold", &(params.max_planar_dist_threshold));
    parser.addParam("max_planar_ang_threshold", &(params.max_planar_ang_threshold));
    parser.addParam("max_linear_dist_threshold", &(params.max_linear_dist_threshold));
    parser.addParam("max_linear_ang_threshold", &(params.max_linear_ang_threshold));
    parser.addParam("ang_scaling_param", &(params.ang_scaling_param));
    parser.addParam("min_new_points", &(params.min_new_points));

    parser.addParam("only_extract_features", &(params.only_extract_features));
    parser.addParam("use_weighting", &(params.use_weighting));
    parser.addParam("print_opt_sum", &(params.print_opt_sum));

    parser.load(path + filename);

    params.inv_Qc = params.Qc.inverse();
}

void loadFeatureParams(const std::string &path, const std::string &filename, wave::FeatureExtractorParams &params) {
    wave::ConfigParser parser;

    parser.addParam("variance_window", &(params.variance_window));
    parser.addParam("variance_limit_rng", &(params.variance_limit_rng));
    parser.addParam("variance_limit_int", &(params.variance_limit_int));
    parser.addParam("angular_bins", &(params.angular_bins));
    parser.addParam("min_intensity", &(params.min_intensity));
    parser.addParam("max_intensity", &(params.max_intensity));
    parser.addParam("occlusion_tol", &(params.occlusion_tol));
    parser.addParam("occlusion_tol_2", &(params.occlusion_tol_2));
    parser.addParam("occlusion_filter_length", &(params.occlusion_filter_length));
    parser.addParam("parallel_tol", &(params.parallel_tol));
    parser.addParam("edge_tol", &(params.edge_tol));
    parser.addParam("flat_tol", &(params.flat_tol));
    parser.addParam("int_edge_tol", &(params.int_edge_tol));
    parser.addParam("int_flat_tol", &(params.int_flat_tol));
    parser.addParam("n_edge", &(params.n_edge));
    parser.addParam("n_flat", &(params.n_flat));
    parser.addParam("n_int_edge", &(params.n_int_edge));
    parser.addParam("knn", &(params.knn));
    parser.addParam("key_radius", &(params.key_radius));

    parser.load(path + filename);
}

void setupFeatureParameters(wave::FeatureExtractorParams &param) {
    std::vector<wave::Criteria> edge_high, edge_low, flat, edge_int_high, edge_int_low;
    edge_high.emplace_back(
            wave::Criteria{wave::Signal::RANGE, wave::Kernel::LOAM, wave::SelectionPolicy::HIGH_POS, &(param.edge_tol)});

    edge_low.emplace_back(
            wave::Criteria{wave::Signal::RANGE, wave::Kernel::LOAM, wave::SelectionPolicy::HIGH_NEG, &(param.edge_tol)});

    flat.emplace_back(
            wave::Criteria{wave::Signal::RANGE, wave::Kernel::LOAM, wave::SelectionPolicy::NEAR_ZERO, &(param.flat_tol)});
    flat.emplace_back(wave::Criteria{
            wave::Signal::RANGE, wave::Kernel::VAR, wave::SelectionPolicy::NEAR_ZERO, &(param.variance_limit_rng)});

    edge_int_high.emplace_back(wave::Criteria{
            wave::Signal::INTENSITY, wave::Kernel::FOG, wave::SelectionPolicy::HIGH_POS, &(param.int_edge_tol)});
    edge_int_high.emplace_back(
            wave::Criteria{wave::Signal::RANGE, wave::Kernel::LOAM, wave::SelectionPolicy::NEAR_ZERO, &(param.int_flat_tol)});
    edge_int_high.emplace_back(wave::Criteria{
            wave::Signal::RANGE, wave::Kernel::VAR, wave::SelectionPolicy::NEAR_ZERO, &(param.variance_limit_int)});

    edge_int_low.emplace_back(wave::Criteria{
            wave::Signal::INTENSITY, wave::Kernel::FOG, wave::SelectionPolicy::HIGH_NEG, &(param.int_edge_tol)});
    edge_int_low.emplace_back(
            wave::Criteria{wave::Signal::RANGE, wave::Kernel::LOAM, wave::SelectionPolicy::NEAR_ZERO, &(param.int_flat_tol)});
    edge_int_low.emplace_back(wave::Criteria{
            wave::Signal::RANGE, wave::Kernel::VAR, wave::SelectionPolicy::NEAR_ZERO, &(param.variance_limit_int)});

    param.feature_definitions.clear();
    param.feature_definitions.emplace_back(wave::FeatureDefinition{edge_high, &(param.n_edge)});
    param.feature_definitions.emplace_back(wave::FeatureDefinition{edge_low, &(param.n_edge)});
    param.feature_definitions.emplace_back(wave::FeatureDefinition{flat, &(param.n_flat)});
    param.feature_definitions.emplace_back(wave::FeatureDefinition{edge_int_high, &(param.n_int_edge)});
    param.feature_definitions.emplace_back(wave::FeatureDefinition{edge_int_low, &(param.n_int_edge)});
}

void loadBinnerParams(const std::string &path, const std::string &filename, wave::IcosahedronBinnerParams &params) {
    wave::ConfigParser parser;

    parser.addParam("range_divisions", &(params.range_divisions));
    parser.addParam("azimuth_divisions", &(params.azimuth_divisions));
    parser.addParam("xy_directions", &(params.xy_directions));
    parser.addParam("z_cutoff", &(params.z_cutoff));
    parser.addParam("z_limit", &(params.z_limit));
    parser.addParam("xy_limit", &(params.xy_limit));

    parser.load(path + filename);
}

void updateVisualizer(const wave::LaserOdom *odom,
                      wave::PointCloudDisplay *display,
                      wave::VecE<wave::PoseVelStamped> *odom_trajectory,
                      Vec<TrackLengths> *track_lengths) {
    bool first_trajectory = true;
    wave::T_TYPE T0X;
    for (const wave::PoseVelStamped &val : odom->undistort_trajectory) {
        auto iter =
                std::find_if(odom_trajectory->begin(), odom_trajectory->end(), [&val](const wave::PoseVelStamped &item) {
                    auto diff = item.stamp - val.stamp;
                    // todo figure out a way to avoid floating point time and remove this workaround
                    if (diff > std::chrono::seconds(0)) {
                        return diff < std::chrono::microseconds(10);
                    } else {
                        return diff > std::chrono::microseconds(-10);
                    }
                });
        if (iter == odom_trajectory->end()) {
            odom_trajectory->emplace_back(val);
            if (!first_trajectory) {
                odom_trajectory->back().pose = T0X * odom_trajectory->back().pose;
            }
        } else {
            if (first_trajectory) {
                T0X = iter->pose;
                first_trajectory = false;
            }
            auto index = (unsigned long) (iter - odom_trajectory->begin());
            odom_trajectory->at(index) = val;
            odom_trajectory->at(index).pose = T0X * odom_trajectory->at(index).pose;
        }
    }

    int ptcld_id = 100000;
    if (display) {
        display->removeAllShapes();
        pcl::PointCloud<pcl::PointXYZI>::Ptr viz_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        *viz_cloud = odom->undistorted_cld;
        display->addPointcloud(viz_cloud, ptcld_id, false, 6);
        ++ptcld_id;

        for (uint32_t i = 1; i < odom->undistort_trajectory.size(); ++i) {
            pcl::PointXYZ pt1, pt2;
            Eigen::Map<wave::Vec3f> m1(pt1.data), m2(pt2.data);
            m1 = odom->undistort_trajectory.at(i - 1).pose.storage.block<3, 1>(0, 3).cast<float>();
            m2 = odom->undistort_trajectory.at(i).pose.storage.block<3, 1>(0, 3).cast<float>();
            display->addLine(pt1, pt2, i - 1, i);
        }

        int id = odom->undistort_trajectory.size();

        float int_id = 0;

        for (uint32_t i = 0; i < 5; ++i) {
            int viewport_id = i + 1;
            int_id = 0;

            pcl::PointCloud<pcl::PointXYZI>::Ptr display_cld = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

            std::vector<int> counts(odom->getParams().n_window + 1);
            std::fill(counts.begin(), counts.end(), 0);

            for (const auto &track : odom->undis_tracks.at(i)) {
                try {
                    counts.at(track.length)++;
                } catch (...) {
                    throw std::runtime_error("track length longer than expected");
                }
                if (i == 2) {
                    pcl::PointXYZ pt1, pt2;
                    Eigen::Map<wave::Vec3f> m1(pt1.data), m2(pt2.data);
                    m1 = track.geometry.block<3, 1>(3, 0).cast<float>();
                    m2 = track.geometry.block<3, 1>(0, 0).cast<float>();
                    float sidelength = 1.5f;
                    display->addSquare(pt1, pt2, sidelength, id, false, viewport_id);
                    ++id;
                } else {
                    pcl::PointXYZ pt1, pt2;
                    Eigen::Map<wave::Vec3f> m1(pt1.data), m2(pt2.data);

                    double sidelength = 0.1 * track.mapping.size();

                    m1 =
                            (track.geometry.block<3, 1>(3, 0) - sidelength * track.geometry.block<3, 1>(0, 0)).cast<float>();
                    m2 =
                            (track.geometry.block<3, 1>(3, 0) + sidelength * track.geometry.block<3, 1>(0, 0)).cast<float>();
                    display->addLine(pt1, pt2, id, id + 1, false, viewport_id);
                    id += 2;
                }

                for (const auto &map : track.mapping) {
                    pcl::PointXYZI new_pt;
                    new_pt.x = odom->undis_features.at(i).at(map.pt_idx).x;
                    new_pt.y = odom->undis_features.at(i).at(map.pt_idx).y;
                    new_pt.z = odom->undis_features.at(i).at(map.pt_idx).z;

                    new_pt.intensity = int_id;
                    display_cld->push_back(new_pt);
                }
                int_id += 1;
            }

            for (uint32_t j = 0; j < counts.size(); ++j) {
                track_lengths->at(i).lengths.at(j).emplace_back(counts.at(j));
            }

            display->addPointcloud(display_cld, ptcld_id, false, viewport_id);
            ++ptcld_id;

            pcl::PointCloud<pcl::PointXYZ>::Ptr candidate_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            *candidate_cloud = odom->undis_candidates_cur.at(i);
            display->addPointcloud(candidate_cloud, ptcld_id, false, viewport_id);
            ++ptcld_id;
        }
    }
}

void plotResults(const wave::VecE<wave::PoseVelStamped> &ground_truth,
                 const wave::VecE<wave::PoseVelStamped> &odom_trajectory) {
    std::vector<double> ground_truth_x, ground_truth_y, ground_truth_z, ground_truth_stamp, ground_truth_vx,
            ground_truth_vy, ground_truth_vz, ground_truth_wx, ground_truth_wy, ground_truth_wz;
    if (ground_truth_x.empty()) {
        auto start = ground_truth.front().stamp;
        for (const auto &traj : ground_truth) {
            ground_truth_x.emplace_back(traj.pose.storage(0, 3));
            ground_truth_y.emplace_back(traj.pose.storage(1, 3));
            ground_truth_z.emplace_back(traj.pose.storage(2, 3));
            ground_truth_vx.emplace_back(traj.vel(3));
            ground_truth_vy.emplace_back(traj.vel(4));
            ground_truth_vz.emplace_back(traj.vel(5));
            ground_truth_wx.emplace_back(traj.vel(0));
            ground_truth_wy.emplace_back(traj.vel(1));
            ground_truth_wz.emplace_back(traj.vel(2));
            ground_truth_stamp.emplace_back(std::chrono::duration_cast<f64_seconds>(traj.stamp - start).count());
        }
    }

    std::vector<double> odom_x, odom_y, odom_z, odom_stamp, odom_vx, odom_vy, odom_vz, odom_wx, odom_wy, odom_wz;
    auto start = odom_trajectory.front().stamp;
    for (const auto &traj : odom_trajectory) {
        odom_x.emplace_back(traj.pose.storage(0, 3));
        odom_y.emplace_back(traj.pose.storage(1, 3));
        odom_z.emplace_back(traj.pose.storage(2, 3));
        odom_vx.emplace_back(traj.vel(3));
        odom_vy.emplace_back(traj.vel(4));
        odom_vz.emplace_back(traj.vel(5));
        odom_wx.emplace_back(traj.vel(0));
        odom_wy.emplace_back(traj.vel(1));
        odom_wz.emplace_back(traj.vel(2));
        odom_stamp.emplace_back(std::chrono::duration_cast<f64_seconds>(traj.stamp - start).count());
    }

    plot::clf();
    plot::subplot(2, 2, 1);
    plot::named_plot("Ground Truth", ground_truth_x, ground_truth_y);
    plot::named_plot("Estimate", odom_x, odom_y);
    plot::legend();

    plot::subplot(2, 2, 2);
    plot::named_plot("Ground Truth X(m)", ground_truth_stamp, ground_truth_x);
    plot::named_plot("Ground Truth Y(m)", ground_truth_stamp, ground_truth_y);
    plot::named_plot("Ground Truth Z(m)", ground_truth_stamp, ground_truth_z);
    plot::named_plot("Odom X(m)", odom_stamp, odom_x);
    plot::named_plot("Odom Y(m)", odom_stamp, odom_y);
    plot::named_plot("Odom Z(m)", odom_stamp, odom_z);
    plot::legend();

    plot::subplot(2, 2, 3);
    plot::named_plot("Ground Truth wx(rad/s)", ground_truth_stamp, ground_truth_wx);
    plot::named_plot("Ground Truth wy(rad/s)", ground_truth_stamp, ground_truth_wy);
    plot::named_plot("Ground Truth wz(rad/s)", ground_truth_stamp, ground_truth_wz);
    plot::named_plot("Odom wx(rad/s)", odom_stamp, odom_wx);
    plot::named_plot("Odom wy(rad/s)", odom_stamp, odom_wy);
    plot::named_plot("Odom wz(rad/s)", odom_stamp, odom_wz);
    plot::legend();

    plot::subplot(2, 2, 4);
    plot::named_plot("Ground Truth Vx(m/s)", ground_truth_stamp, ground_truth_vx);
    plot::named_plot("Ground Truth Vy(m/s)", ground_truth_stamp, ground_truth_vy);
    plot::named_plot("Ground Truth Vz(m/s)", ground_truth_stamp, ground_truth_vz);
    plot::named_plot("Odom Vx(m/s)", odom_stamp, odom_vx);
    plot::named_plot("Odom Vy(m/s)", odom_stamp, odom_vy);
    plot::named_plot("Odom Vz(m/s)", odom_stamp, odom_vz);
    plot::legend();

    plot::show(true);
}

void plotResults(const wave::VecE<wave::PoseStamped> &ground_truth,
                 const wave::VecE<wave::PoseVelStamped> &odom_trajectory) {
    std::vector<double> ground_truth_x, ground_truth_y, ground_truth_z, ground_truth_stamp, ground_truth_vx,
            ground_truth_vy, ground_truth_vz, ground_truth_wx, ground_truth_wy, ground_truth_wz;
    if (ground_truth_x.empty()) {
        auto start = ground_truth.front().stamp;
        for (const auto &traj : ground_truth) {
            ground_truth_x.emplace_back(traj.pose.storage(0, 3));
            ground_truth_y.emplace_back(traj.pose.storage(1, 3));
            ground_truth_z.emplace_back(traj.pose.storage(2, 3));
            ground_truth_stamp.emplace_back(std::chrono::duration_cast<f64_seconds>(traj.stamp - start).count());
        }
    }

    std::vector<double> odom_x, odom_y, odom_z, odom_stamp, odom_vx, odom_vy, odom_vz, odom_wx, odom_wy, odom_wz;
    auto start = odom_trajectory.front().stamp;
    for (const auto &traj : odom_trajectory) {
        odom_x.emplace_back(traj.pose.storage(0, 3));
        odom_y.emplace_back(traj.pose.storage(1, 3));
        odom_z.emplace_back(traj.pose.storage(2, 3));
        odom_vx.emplace_back(traj.vel(3));
        odom_vy.emplace_back(traj.vel(4));
        odom_vz.emplace_back(traj.vel(5));
        odom_wx.emplace_back(traj.vel(0));
        odom_wy.emplace_back(traj.vel(1));
        odom_wz.emplace_back(traj.vel(2));
        odom_stamp.emplace_back(std::chrono::duration_cast<f64_seconds>(traj.stamp - start).count());
    }

    plot::clf();
    plot::subplot(2, 2, 1);
    plot::named_plot("Ground Truth", ground_truth_x, ground_truth_y);
    plot::named_plot("Estimate", odom_x, odom_y);
    plot::legend();

    plot::subplot(2, 2, 2);
    plot::named_plot("Ground Truth X(m)", ground_truth_stamp, ground_truth_x);
    plot::named_plot("Ground Truth Y(m)", ground_truth_stamp, ground_truth_y);
    plot::named_plot("Ground Truth Z(m)", ground_truth_stamp, ground_truth_z);
    plot::named_plot("Odom X(m)", odom_stamp, odom_x);
    plot::named_plot("Odom Y(m)", odom_stamp, odom_y);
    plot::named_plot("Odom Z(m)", odom_stamp, odom_z);
    plot::legend();

    plot::subplot(2, 2, 3);
    plot::named_plot("Odom wx(rad/s)", odom_stamp, odom_wx);
    plot::named_plot("Odom wy(rad/s)", odom_stamp, odom_wy);
    plot::named_plot("Odom wz(rad/s)", odom_stamp, odom_wz);
    plot::legend();

    plot::subplot(2, 2, 4);
    plot::named_plot("Odom Vx(m/s)", odom_stamp, odom_vx);
    plot::named_plot("Odom Vy(m/s)", odom_stamp, odom_vy);
    plot::named_plot("Odom Vz(m/s)", odom_stamp, odom_vz);
    plot::legend();

    plot::show(true);
}

void plotResults(const wave::VecE<wave::PoseVelStamped> &odom_trajectory) {
    std::vector<double> odom_x, odom_y, odom_z, odom_stamp, odom_vx, odom_vy, odom_vz, odom_wx, odom_wy, odom_wz;
    auto start = odom_trajectory.front().stamp;
    for (const auto &traj : odom_trajectory) {
        odom_x.emplace_back(traj.pose.storage(0, 3));
        odom_y.emplace_back(traj.pose.storage(1, 3));
        odom_z.emplace_back(traj.pose.storage(2, 3));
        odom_vx.emplace_back(traj.vel(3));
        odom_vy.emplace_back(traj.vel(4));
        odom_vz.emplace_back(traj.vel(5));
        odom_wx.emplace_back(traj.vel(0));
        odom_wy.emplace_back(traj.vel(1));
        odom_wz.emplace_back(traj.vel(2));
        odom_stamp.emplace_back(std::chrono::duration_cast<f64_seconds>(traj.stamp - start).count());
    }

    plot::clf();
    plot::subplot(2, 2, 1);
    plot::named_plot("Estimate", odom_x, odom_y);
    plot::legend();

    plot::subplot(2, 2, 2);
    plot::named_plot("Odom X(m)", odom_stamp, odom_x);
    plot::named_plot("Odom Y(m)", odom_stamp, odom_y);
    plot::named_plot("Odom Z(m)", odom_stamp, odom_z);
    plot::legend();

    plot::subplot(2, 2, 3);
    plot::named_plot("Odom wx(rad/s)", odom_stamp, odom_wx);
    plot::named_plot("Odom wy(rad/s)", odom_stamp, odom_wy);
    plot::named_plot("Odom wz(rad/s)", odom_stamp, odom_wz);
    plot::legend();

    plot::subplot(2, 2, 4);
    plot::named_plot("Odom Vx(m/s)", odom_stamp, odom_vx);
    plot::named_plot("Odom Vy(m/s)", odom_stamp, odom_vy);
    plot::named_plot("Odom Vz(m/s)", odom_stamp, odom_vz);
    plot::legend();

    plot::show(true);
}

void plotError(const wave::VecE<wave::PoseStamped> &ground_truth, const wave::VecE<wave::PoseVelStamped> &odom_trajectory) {
    if (ground_truth.size() != odom_trajectory.size()) {
        LOG_ERROR("Ground truth size mismatched with estimate size!");
    }
    //absolute error since start
    std::vector<double> x, y, z, rx, ry, rz;
    //pairwise error
    std::vector<double> px, py, pz, prx, pry, prz;

    for (uint32_t frame_id = 0; frame_id < ground_truth.size(); ++frame_id) {
        Vec6 diff = odom_trajectory.at(frame_id).pose.manifoldMinus(ground_truth.at(frame_id).pose);
        rx.emplace_back(diff(0));
        ry.emplace_back(diff(1));
        rz.emplace_back(diff(2));
        x.emplace_back(diff(3));
        y.emplace_back(diff(4));
        z.emplace_back(diff(5));
        if (frame_id > 0) {
            px.emplace_back(x.at(frame_id) - x.at(frame_id - 1));
            py.emplace_back(y.at(frame_id) - y.at(frame_id - 1));
            pz.emplace_back(z.at(frame_id) - z.at(frame_id - 1));
            prx.emplace_back(rx.at(frame_id) - rx.at(frame_id - 1));
            pry.emplace_back(ry.at(frame_id) - ry.at(frame_id - 1));
            prz.emplace_back(rz.at(frame_id) - rz.at(frame_id - 1));
        }
    }

    plot::clf();
    plot::subplot(2, 2, 1);
    plot::named_plot("Tx Error", x);
    plot::named_plot("Ty Error", y);
    plot::named_plot("Tz Error", z);
    plot::legend();
    plot::title("Translation Error");

    plot::subplot(2, 2, 2);
    plot::named_plot("Rx Error", rx);
    plot::named_plot("Ry Error", ry);
    plot::named_plot("Rz Error", rz);
    plot::legend();
    plot::title("Rotation Error");

    plot::subplot(2, 2, 3);
    plot::named_plot("Tx Error", px);
    plot::named_plot("Ty Error", py);
    plot::named_plot("Tz Error", pz);
    plot::legend();
    plot::title("Incremental Translation Error");

    plot::subplot(2, 2, 4);
    plot::named_plot("Rx Error", prx);
    plot::named_plot("Ry Error", pry);
    plot::named_plot("Rz Error", prz);
    plot::legend();
    plot::title("Incremental Rotation Error");

    plot::show(true);
}

void plotTrackLengths(const Vec<TrackLengths> &track_lengths) {
    int feat_id = 0;
    for (const auto &lengths : track_lengths) {
        plot::clf();
        auto n_series = lengths.lengths.size();
        std::vector<double> length_cnt(lengths.lengths.at(0).size());
        std::fill(length_cnt.begin(), length_cnt.end(), 0.0);
        for (uint32_t length_id = 0; length_id < n_series; ++length_id) {
            for (uint32_t frame_id = 0; frame_id < length_cnt.size(); ++frame_id) {
                length_cnt.at(frame_id) += lengths.lengths.at(length_id).at(frame_id);
            }
            plot::named_plot("length " + std::to_string(length_id), length_cnt);
        }
        plot::legend();
        plot::title("Feature Id: " + std::to_string(feat_id));
        plot::show(true);
        feat_id++;
    }
}

wave::TimeType parseTime(const std::string &date_time) {
    std::tm tm = {};
    auto nano_seconds = static_cast<unsigned long>(std::stoi(date_time.substr(20, 9)));
    std::chrono::nanoseconds n_secs(nano_seconds);
    std::stringstream ss(date_time);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    std::chrono::system_clock::time_point sys_clock = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    wave::TimeType stamp;
    stamp = stamp + sys_clock.time_since_epoch() + n_secs;
    return stamp;
}

/**
 * Mapping between raw data and odometry sequences
 * Nr.     Sequence name     Start   End
---------------------------------------
00: 2011_10_03_drive_0027 000000 004540
01: 2011_10_03_drive_0042 000000 001100
02: 2011_10_03_drive_0034 000000 004660
03: 2011_09_26_drive_0067 000000 000800
04: 2011_09_30_drive_0016 000000 000270
05: 2011_09_30_drive_0018 000000 002760
06: 2011_09_30_drive_0020 000000 001100
07: 2011_09_30_drive_0027 000000 001100
08: 2011_09_30_drive_0028 001100 005170
09: 2011_09_30_drive_0033 000000 001590
10: 2011_09_30_drive_0034 000000 001200
 */

/**
 * The GPS/IMU information is given in a single small text file which is
written for each synchronized frame. Each text file contains 30 values
        which are:

- lat:     latitude of the oxts-unit (deg)
- lon:     longitude of the oxts-unit (deg)
- alt:     altitude of the oxts-unit (m)
- roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
- pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
- yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
- vn:      velocity towards north (m/s)
- ve:      velocity towards east (m/s)
- vf:      forward velocity, i.e. parallel to earth-surface (m/s)
- vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
- vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
- ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
- ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
- az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
- af:      forward acceleration (m/s^2)
- al:      leftward acceleration (m/s^2)
- au:      upward acceleration (m/s^2)
- wx:      angular rate around x (rad/s)
- wy:      angular rate around y (rad/s)
- wz:      angular rate around z (rad/s)
- wf:      angular rate around forward axis (rad/s)
- wl:      angular rate around leftward axis (rad/s)
- wu:      angular rate around upward axis (rad/s)
- posacc:  velocity accuracy (north/east in m)
- velacc:  velocity accuracy (north/east in m/s)
- navstat: navigation status
- numsats: number of satellites tracked by primary GPS receiver
- posmode: position mode of primary GPS receiver
- velmode: velocity mode of primary GPS receiver
- orimode: orientation mode of primary GPS receiver
 **/

// order is lat, long, (deg), alt, r, p, y (radians)
wave::Mat34 poseFromGPS(const std::vector<double> &vals, const wave::Mat34 &reference, wave::Mat3 &R_RLOCAL_CAR) {
    double T_ecef_enu[4][4];

    wave::ecefFromENUTransformMatrix(vals.data(), T_ecef_enu);

    double sr = std::sin(vals.at(3)), cr = std::cos(vals.at(3)), sp = std::sin(vals.at(4)), cp = std::cos(vals.at(4)),
            sy = std::sin(vals.at(5)), cy = std::cos(vals.at(5));

    wave::Mat3 Rx, Ry, Rz;
    Rx << 1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr;

    Ry << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;

    Rz << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0;

    wave::Mat34 T_ECEF_ENU, T_ECEF_CAR;

    for (uint32_t i = 0; i < 3; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            T_ECEF_ENU(i, j) = T_ecef_enu[i][j];
        }
    }

    R_RLOCAL_CAR.noalias() = Ry * Rx;

    T_ECEF_CAR.block<3, 3>(0, 0).noalias() = reference.block<3, 3>(0, 0) * (T_ECEF_ENU.block<3, 3>(0, 0) * Rz * R_RLOCAL_CAR);
    T_ECEF_CAR.block<3, 1>(0, 3).noalias() =
            reference.block<3, 3>(0, 0) * T_ECEF_ENU.block<3, 1>(0, 3) + reference.block<3, 1>(0, 3);

    return T_ECEF_CAR;
}

void fillGroundTruth(wave::VecE<wave::PoseVelStamped> &trajectory, const std::string &data_path) {
    std::string oxt_data_path = data_path + "oxts/data/";
    std::string timestamp_file = data_path + "oxts/timestamps.txt";

    boost::filesystem::path p(oxt_data_path);
    std::vector<boost::filesystem::path> v;
    std::copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), std::back_inserter(v));
    std::sort(v.begin(), v.end());

    wave::Mat34 reference;
    reference.block<3, 3>(0, 0).setIdentity();
    reference.block<3, 1>(0, 3).setZero();
    for (const auto &iter : v) {
        fstream oxt_data;
        oxt_data.open(iter.string(), ios::in);
        std::string data_string;
        std::getline(oxt_data, data_string);
        std::stringstream ss(data_string);
        std::vector<double> vals;
        while (ss.rdbuf()->in_avail() > 0) {
            double new_val;
            ss >> new_val;
            vals.emplace_back(new_val);
        }
        wave::PoseVelStamped new_measurement;
        Mat3 R_RLOCAL_CAR;
        new_measurement.pose.storage = poseFromGPS(vals, reference, R_RLOCAL_CAR);
        Vec3 RLOCAL_V_EARTH_CAR;
        RLOCAL_V_EARTH_CAR << vals.at(8), vals.at(9), vals.at(10);
        Vec3 CAR_V_EARTH_CAR = R_RLOCAL_CAR.transpose() * RLOCAL_V_EARTH_CAR;
        new_measurement.vel << vals.at(20), vals.at(21), vals.at(22), CAR_V_EARTH_CAR(0), CAR_V_EARTH_CAR(1), CAR_V_EARTH_CAR(2);
        trajectory.emplace_back(new_measurement);
        if (trajectory.size() == 1) {
            wave::T_TYPE temp = trajectory.front().pose;
            trajectory.front().pose.storage = reference;
            reference = temp.transformInverse().storage;
        }
        oxt_data.close();
    }

    fstream stamp_file(timestamp_file, ios::in);
    std::string date_time;
    for (auto &traj : trajectory) {
        std::getline(stamp_file, date_time);
        traj.stamp = parseTime(date_time);
    }
    stamp_file.close();
}

}
