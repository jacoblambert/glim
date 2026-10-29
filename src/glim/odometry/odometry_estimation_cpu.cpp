#include <glim/odometry/odometry_estimation_cpu.hpp>
#include <glim/odometry/scan_geometry_information.hpp>

#include <spdlog/spdlog.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <glim/util/config.hpp>
#include <glim/common/imu_integration.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <glim/odometry/callbacks.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

using Callbacks = OdometryEstimationCallbacks;

using gtsam::symbol_shorthand::B;  // IMU bias
using gtsam::symbol_shorthand::V;  // IMU velocity   (v_world_imu)
using gtsam::symbol_shorthand::X;  // IMU pose       (T_world_imu)

OdometryEstimationCPUParams::OdometryEstimationCPUParams() : OdometryEstimationIMUParams() {
  // odometry config
  Config config(GlobalConfig::get_config_path("config_odometry"));

  registration_type = config.param<std::string>("odometry_estimation", "registration_type", "VGICP");
  max_iterations = config.param<int>("odometry_estimation", "max_iterations", 5);
  lru_thresh = config.param<int>("odometry_estimation", "lru_thresh", 100);
  target_downsampling_rate = config.param<double>("odometry_estimation", "target_downsampling_rate", 0.1);

  ivox_resolution = config.param<double>("odometry_estimation", "ivox_resolution", 0.5);
  ivox_min_dist = config.param<double>("odometry_estimation", "ivox_min_dist", 0.1);

  vgicp_resolution = config.param<double>("odometry_estimation", "vgicp_resolution", 0.2);
  vgicp_voxelmap_levels = config.param<int>("odometry_estimation", "vgicp_voxelmap_levels", 2);
  vgicp_voxelmap_scaling_factor = config.param<double>("odometry_estimation", "vgicp_voxelmap_scaling_factor", 2.0);

  // Scan-match kinematic gate (default ON; set scan_gate_trans/rot huge to disable).
  scan_factor_precision = config.param<double>("odometry_estimation", "scan_factor_precision", 1e3);
  // Split relative/absolute scan precisions; default to scan_factor_precision (no behaviour change if unset).
  scan_between_precision = config.param<double>("odometry_estimation", "scan_between_precision", scan_factor_precision);
  scan_prior_precision = config.param<double>("odometry_estimation", "scan_prior_precision", scan_factor_precision);
  scan_gate_trans = config.param<double>("odometry_estimation", "scan_gate_trans", 0.5);
  scan_gate_rot = config.param<double>("odometry_estimation", "scan_gate_rot", 0.35);
  scan_gate_reject_precision = config.param<double>("odometry_estimation", "scan_gate_reject_precision", 1.0);
  revise_target_poses = config.param<bool>("odometry_estimation", "revise_target_poses", false);
  use_scan_geometry_information = config.param<bool>("odometry_estimation", "use_scan_geometry_information", false);
  use_registration_mean_information = config.param<bool>("odometry_estimation", "use_registration_mean_information", false);
}

OdometryEstimationCPUParams::~OdometryEstimationCPUParams() {}

OdometryEstimationCPU::OdometryEstimationCPU(const OdometryEstimationCPUParams& params) : OdometryEstimationIMU(std::make_unique<OdometryEstimationCPUParams>(params)) {
  last_T_target_imu.setIdentity();
  if (params.revise_target_poses && params.lru_thresh <= 0) {
    throw std::invalid_argument("revisable target requires a positive bounded observation horizon");
  }
  if (params.use_registration_mean_information && (params.use_scan_geometry_information || params.registration_type != "GICP")) {
    throw std::invalid_argument("mean registration information requires one GICP model and no geometry-shape ablation");
  }
  reset_target();
}

void OdometryEstimationCPU::reset_target() {
  const auto& params = *static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  if (params.registration_type == "GICP") {
    target_ivox.reset(new gtsam_points::iVox(params.ivox_resolution));
    target_ivox->voxel_insertion_setting().set_min_dist_in_cell(params.ivox_min_dist);
    target_ivox->set_lru_horizon(params.lru_thresh);
    target_ivox->set_neighbor_voxel_mode(1);
  } else if (params.registration_type == "VGICP") {
    target_voxelmaps.resize(params.vgicp_voxelmap_levels);
    for (int i = 0; i < params.vgicp_voxelmap_levels; i++) {
      const double resolution = params.vgicp_resolution * std::pow(params.vgicp_voxelmap_scaling_factor, i);
      target_voxelmaps[i] = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
      target_voxelmaps[i]->set_lru_horizon(params.lru_thresh);
    }
  } else {
    spdlog::error("unknown registration type for odometry_estimation_cpu ({})", params.registration_type);
    abort();
  }
}

OdometryEstimationCPU::~OdometryEstimationCPU() {}

gtsam::NonlinearFactorGraph OdometryEstimationCPU::create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) {
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  const int last = current - 1;

  if (current == 0) {
    last_T_target_imu = frames[current]->T_world_imu;
    update_target(current, frames[current]->T_world_imu);
    return gtsam::NonlinearFactorGraph();
  }

  const Eigen::Isometry3d pred_T_last_current = frames[last]->T_world_imu.inverse() * frames[current]->T_world_imu;
  // A reconstructed target and the last pose now share the fused world chart.
  // Frozen-target mode deliberately retains the historical scan-map chart.
  if (params->revise_target_poses) last_T_target_imu = frames[last]->T_world_imu;
  const Eigen::Isometry3d pred_T_target_imu = last_T_target_imu * pred_T_last_current;

  gtsam::Values values;
  values.insert(X(current), gtsam::Pose3(pred_T_target_imu.matrix()));

  // Create frame-to-model matching factor
  gtsam::NonlinearFactorGraph matching_cost_factors;
  if (params->registration_type == "GICP") {
    auto gicp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
      gtsam::Pose3(),
      X(current),
      target_ivox,
      frames[current]->frame,
      target_ivox);
    gicp_factor->set_max_correspondence_distance(params->ivox_resolution * 2.0);
    gicp_factor->set_num_threads(params->num_threads);
    matching_cost_factors.add(gicp_factor);
  } else if (params->registration_type == "VGICP") {
    for (const auto& voxelmap : target_voxelmaps) {
      auto vgicp_factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(gtsam::Pose3(), X(current), voxelmap, frames[current]->frame);
      vgicp_factor->set_num_threads(params->num_threads);
      matching_cost_factors.add(vgicp_factor);
    }
  }

  gtsam::NonlinearFactorGraph graph;
  graph.add(matching_cost_factors);

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(params->max_iterations);
  lm_params.setAbsoluteErrorTol(0.1);

  gtsam::Pose3 last_estimate = values.at<gtsam::Pose3>(X(current));
  lm_params.termination_criteria = [&](const gtsam::Values& values) {
    const gtsam::Pose3 current_pose = values.at<gtsam::Pose3>(X(current));
    const gtsam::Pose3 delta = last_estimate.inverse() * current_pose;

    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.rotation().matrix()).angle();
    last_estimate = current_pose;

    if (delta_t < 1e-10 && delta_r < 1e-10) {
      // Maybe failed to solve the linear system
      return false;
    }

    // Convergence check
    return delta_t < 1e-3 && delta_r < 1e-3 * M_PI / 180.0;
  };

  // Optimize
  // lm_params.setDiagonalDamping(true);
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

#ifdef GTSAM_USE_TBB
  auto arena = static_cast<tbb::task_arena*>(this->tbb_task_arena.get());
  arena->execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  Eigen::Isometry3d T_target_imu = Eigen::Isometry3d(values.at<gtsam::Pose3>(X(current)).matrix());
  Eigen::Isometry3d T_last_current = last_T_target_imu.inverse() * T_target_imu;
  T_last_current.linear() = Eigen::Quaterniond(T_last_current.linear()).normalized().toRotationMatrix();

  // Kinematic gate on the scan match. GICP can be CONFIDENTLY WRONG -- its own covariance can't flag it
  // (verified: at a 4 m divergence the GICP info eigenvalues stayed ~1e4), so we check the GICP relative
  // pose against the IMU-pre-integrated prediction (an INDEPENDENT reference). If the match implies motion
  // the IMU says is impossible (> scan_gate_trans / scan_gate_rot in one keyframe), REJECT it: fall back
  // entirely to the IMU prediction for this frame's pose, target-map update, AND factors (down-weighted),
  // so IMU + RTK carry the frame. NOTE: merely down-weighting the smoother factor is NOT enough -- the raw
  // GICP pose still corrupts frames[].T_world_imu and the target map, and once jumped the next frames look
  // self-consistent so the gate stops firing and the jump persists. Substituting the IMU prediction
  // neutralizes the bad match fully.
  const Eigen::Isometry3d gate_delta = pred_T_last_current.inverse() * T_last_current;
  const double gate_trans = gate_delta.translation().norm();
  const double gate_rot = Eigen::AngleAxisd(gate_delta.linear()).angle();
  double between_precision = params->scan_between_precision;  // relative: strong (odometry/exploration)
  double prior_precision = params->scan_prior_precision;      // absolute: weak < RTK (let RTK bound the drift)
  bool scan_rejected = false;
  if (gate_trans > params->scan_gate_trans || gate_rot > params->scan_gate_rot) {
    between_precision = params->scan_gate_reject_precision;
    prior_precision = params->scan_gate_reject_precision;
    scan_rejected = true;
    spdlog::warn(
      "scan gate: GICP-vs-IMU discrepancy trans={:.2f}m rot={:.1f}deg exceeds gate ({:.2f}m/{:.1f}deg) -> REJECT match, IMU+RTK carry this frame",
      gate_trans, gate_rot * 180.0 / M_PI, params->scan_gate_trans, params->scan_gate_rot * 180.0 / M_PI);
    // Reject the match: use the IMU prediction for pose, target update, and factor values.
    T_target_imu = pred_T_target_imu;
    T_last_current = pred_T_last_current;
  }

  frames[current]->T_world_imu = frames[last]->T_world_imu * T_last_current;
  new_values.insert_or_assign(X(current), gtsam::Pose3(frames[current]->T_world_imu.matrix()));

  gtsam::NonlinearFactorGraph factors;

  // TODO: Extract a relative pose covariance from a frame-to-model matching result?
  gtsam::SharedNoiseModel between_noise = gtsam::noiseModel::Isotropic::Precision(6, between_precision);
  gtsam::SharedNoiseModel prior_noise = gtsam::noiseModel::Isotropic::Precision(6, prior_precision);
  if (params->use_scan_geometry_information && !scan_rejected) {
    const auto& frame = frames[current]->frame;
    const auto information = scan_geometry_information(frame->points, frame->normals, frame->size());
    between_noise = gtsam::noiseModel::Gaussian::Information(between_precision*information);
    prior_noise = gtsam::noiseModel::Gaussian::Information(prior_precision*information);
  }
  if (params->use_registration_mean_information && !scan_rejected) {
    // Keep the actual converged correspondences/covariances and right-increment
    // chart of registration instead of discarding them for a six-axis scalar.
    // Mean, not sum: merely duplicating correlated returns must not increase
    // confidence. This remains a local-model approximation, NOT a calibrated
    // covariance or a correspondence correctness certificate. Keep the relative
    // factor and rejected-match policy unchanged in this declared ablation.
    const auto linearized = matching_cost_factors.linearize(values);
    const Eigen::Matrix<double,6,6> h = linearized->hessianBlockDiagonal().at(X(current));
    prior_noise = gtsam::noiseModel::Gaussian::Information(
      registration_mean_information(h, frames[current]->frame->size()));
  }
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(last), X(current), gtsam::Pose3(T_last_current.matrix()), between_noise);
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(current), gtsam::Pose3(T_target_imu.matrix()), prior_noise);

  // Anti-spiral: when the match was rejected, do NOT insert this frame's points into the target map.
  // The (IMU-predicted) pose is uncertain; inserting points at it would drift the map, which makes the
  // NEXT GICP disagree -> keeps the gate firing -> open-loop IMU drift (the 48-consecutive-fire spiral
  // that diverged even with the scan rejected). Keeping the last-good map lets a subsequent good scan
  // re-anchor. last_T_target_imu STILL advances (= IMU prediction) so the frame-to-model prediction
  // stays composition-consistent; only the map insertion is skipped.
  if (!scan_rejected) {
    update_target(current, T_target_imu);
  }
  last_T_target_imu = T_target_imu;

  return factors;
}

void OdometryEstimationCPU::fallback_smoother() {
  // A failed fixed-lag solve is not a valid state to continue from. The process
  // supervisor gates outputs, stops navigation, and starts a clean estimator.
  throw std::runtime_error("CPU smoother corrupted; supervised reinitialization required");
}

void OdometryEstimationCPU::update_frames(int current, const gtsam::NonlinearFactorGraph& new_factors) {
  OdometryEstimationIMU::update_frames(current, new_factors);
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  if (!params->revise_target_poses) return;
  rebuild_target();
  last_T_target_imu = frames[current]->T_world_imu;
}

void OdometryEstimationCPU::rebuild_target() {
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  reset_target();
  // Marginalized observations retain their last fused pose. Live observations
  // reference the exact frame objects just updated by the smoother. Rebuild is
  // bounded by lru_thresh accepted observations and never re-samples evidence.
  for (const auto& observation : target_observations) {
    auto transformed = gtsam_points::transform(observation.points, observation.pose->T_world_imu);
    if (params->registration_type == "GICP") target_ivox->insert(*transformed);
    else for (auto& map : target_voxelmaps) map->insert(*transformed);
  }
}

void OdometryEstimationCPU::update_target(const int current, const Eigen::Isometry3d& T_target_imu) {
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  auto frame = frames[current]->frame;
  if (current >= 5) {
    frame = gtsam_points::random_sampling(frames[current]->frame, params->target_downsampling_rate, mt);
  }

  if (params->revise_target_poses) {
    target_observations.push_back({frames[current], frame});
    while (target_observations.size() > static_cast<size_t>(params->lru_thresh)) target_observations.pop_front();
    return;  // Insert only after fusion, in update_frames().
  }

  auto transformed = gtsam_points::transform(frame, T_target_imu);
  if (params->registration_type == "GICP") {
    target_ivox->insert(*transformed);
  } else if (params->registration_type == "VGICP") {
    for (auto& target_voxelmap : target_voxelmaps) {
      target_voxelmap->insert(*transformed);
    }
  }

  // Update target point cloud (just for visualization).
  // This is not necessary for mapping and can safely be removed.
  if (frames.size() % 50 == 0) {
    EstimationFrame::Ptr frame(new EstimationFrame);
    frame->id = frames.size() - 1;
    frame->stamp = frames.back()->stamp;

    frame->T_lidar_imu = frames.back()->T_lidar_imu;
    frame->T_world_lidar = frame->T_lidar_imu.inverse();
    frame->T_world_imu.setIdentity();

    frame->v_world_imu.setZero();
    frame->imu_bias.setZero();

    frame->frame_id = FrameID::IMU;

    if (params->registration_type == "GICP") {
      frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_ivox->voxel_points());
    } else if (params->registration_type == "VGICP") {
      frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_voxelmaps[0]->voxel_points());
    }

    std::vector<EstimationFrame::ConstPtr> keyframes = {frame};
    Callbacks::on_update_keyframes(keyframes);

    if (target_ivox_frame) {
      std::vector<EstimationFrame::ConstPtr> marginalized_keyframes = {target_ivox_frame};
      Callbacks::on_marginalized_keyframes(marginalized_keyframes);
    }

    target_ivox_frame = frame;
  }
}

}  // namespace glim
