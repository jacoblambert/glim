#pragma once

#include <deque>

#include <glim/odometry/odometry_estimation_imu.hpp>

namespace gtsam_points {

class GaussianVoxelMapCPU;
struct FlatContainer;
template <typename VoxelContents>
class IncrementalVoxelMap;
using iVox = IncrementalVoxelMap<FlatContainer>;
}  // namespace gtsam_points

namespace glim {

/**
 * @brief Parameters for OdometryEstimationCPU
 */
struct OdometryEstimationCPUParams : public OdometryEstimationIMUParams {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationCPUParams();
  virtual ~OdometryEstimationCPUParams();

public:
  // Registration params
  std::string registration_type;    ///< Registration type (GICP or VGICP)
  int max_iterations;               ///< Maximum number of iterations
  int lru_thresh;                   ///< LRU cache threshold
  double target_downsampling_rate;  ///< Downsampling rate for points to be inserted into the target

  double ivox_resolution;  ///< iVox resolution (for GICP)
  double ivox_min_dist;    ///< Minimum distance between points in an iVox cell (for GICP)

  double vgicp_resolution;               ///< Voxelmap resolution (for VGICP)
  int vgicp_voxelmap_levels;             ///< Multi-resolution voxelmap levesl (for VGICP)
  double vgicp_voxelmap_scaling_factor;  ///< Multi-resolution voxelmap scaling factor (for VGICP)

  // Scan-match kinematic gate: GICP can converge CONFIDENTLY to a WRONG match (its covariance stays
  // tight, so covariance-weighting can't catch it -- verified: at a 4 m divergence the GICP info
  // eigenvalues stayed ~1e4, far above the 1e3 prior). We instead check the GICP relative pose against
  // the IMU-pre-integrated prediction (an independent reference); if they disagree by more than these
  // thresholds in one keyframe, the match is implausible -> down-weight the scan factors to
  // scan_gate_reject_precision so IMU + RTK carry that frame (prevents the multi-metre jump).
  double scan_factor_precision;        ///< Default isotropic precision for the matching factors (was hardcoded 1e3); fallback for the two below
  // The scan enters as TWO factors with DISTINCT roles, split so they can be weighted independently:
  //  - the relative BetweenFactor(X_last,X_cur) gives smooth/detailed odometry the BEV+exploration need -> keep STRONG;
  //  - the absolute PriorFactor(X_cur, map-match) pulls the pose toward the (slowly-drifting iVox) map and FIGHTS RTK on
  //    the same key -> keep WEAK (< RTK precision) so RTK sets absolute position (bounds the gradual drift) while the
  //    strong relative factor still carries detail. Both default to scan_factor_precision (no-op) if unset.
  double scan_between_precision;       ///< Relative BetweenFactor precision (strong: odometry/exploration)
  double scan_prior_precision;         ///< Absolute PriorFactor precision (weak < RTK: lets RTK win the absolute contest)
  double scan_gate_trans;              ///< Max GICP-vs-IMU translation discrepancy per keyframe [m] before gating
  double scan_gate_rot;                ///< Max GICP-vs-IMU rotation discrepancy per keyframe [rad] before gating
  double scan_gate_reject_precision;   ///< Isotropic precision applied to the matching factors when gated
  bool revise_target_poses;            ///< Opt-in bounded target reconstruction from fused frame poses
  bool use_scan_geometry_information; ///< Opt-in geometry-shaped information, not a covariance certificate
  bool use_registration_mean_information; ///< Experimental mean registration curvature for the unary prior
};

/**
 * @brief CPU-based semi-tightly coupled LiDAR-IMU odometry
 */
class OdometryEstimationCPU : public OdometryEstimationIMU {
  friend struct TargetReconstructionTest;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationCPU(const OdometryEstimationCPUParams& params = OdometryEstimationCPUParams());
  virtual ~OdometryEstimationCPU() override;

private:
  virtual gtsam::NonlinearFactorGraph create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) override;

  virtual void fallback_smoother() override;
  virtual void update_frames(int current, const gtsam::NonlinearFactorGraph& new_factors) override;

  void update_target(const int current, const Eigen::Isometry3d& T_target_imu);
  void reset_target();
  void rebuild_target();

private:
  // Registration params
  std::mt19937 mt;                                                                   ///< RNG
  Eigen::Isometry3d last_T_target_imu;                                               ///< Last IMU pose w.r.t. target model
  std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> target_voxelmaps;  ///< VGICP target voxelmap
  std::shared_ptr<gtsam_points::iVox> target_ivox;                                   ///< GICP target iVox
  EstimationFrame::ConstPtr target_ivox_frame;                                       ///< Target points (just for visualization)
  // Sample once; keep sensor-frame evidence separate from its revisable pose.
  struct TargetObservation {
    EstimationFrame::ConstPtr pose;
    std::shared_ptr<const gtsam_points::PointCloud> points;
  };
  std::deque<TargetObservation> target_observations;
};

}  // namespace glim
