#include <glim/util/trajectory_manager.hpp>
#include <cmath>

namespace glim {

TrajectoryManager::TrajectoryManager() {
  odom_priority = -1;

  T_world_odom.setIdentity();
}

TrajectoryManager::~TrajectoryManager() {}

void TrajectoryManager::update_anchor(const Eigen::Isometry3d& world, const Eigen::Isometry3d& odom) {
  if (!world.matrix().allFinite() || !odom.matrix().allFinite()) return;
  T_world_odom = world * odom.inverse();
}

void TrajectoryManager::add_odom(double stamp, const Eigen::Isometry3d& T_odom_sensor, int priority) {
  if (!std::isfinite(stamp) || !T_odom_sensor.matrix().allFinite()) return;
  if (odom_priority < priority) {
    odom_priority = priority;
    odom_stamps.clear();
    this->T_odom_sensor.clear();
  } else if (odom_priority != priority) {
    return;
  }

  // Predicted/corrected callbacks share a stamp; replace, never append a
  // duplicate. Historical revisions must preserve the sorted lookup invariant.
  const auto found = std::lower_bound(odom_stamps.begin(), odom_stamps.end(), stamp);
  const auto idx = std::distance(odom_stamps.begin(), found);
  if (found != odom_stamps.end() && *found == stamp) {
    this->T_odom_sensor[idx] = T_odom_sensor;
  } else {
    odom_stamps.insert(found, stamp);
    this->T_odom_sensor.insert(this->T_odom_sensor.begin() + idx, T_odom_sensor);
  }
  // This is a live anchor interpolator, not the historical graph archive.
  size_t erase_count = 0;
  while (erase_count + 1 < odom_stamps.size() &&
         (odom_stamps.size() - erase_count > 10000 || odom_stamps.back() - odom_stamps[erase_count] > 120.0)) ++erase_count;
  odom_stamps.erase(odom_stamps.begin(), odom_stamps.begin() + erase_count);
  this->T_odom_sensor.erase(this->T_odom_sensor.begin(), this->T_odom_sensor.begin() + erase_count);
}

void TrajectoryManager::update_anchor(double stamp, const Eigen::Isometry3d& T_world_sensor) {
  if (!std::isfinite(stamp) || !T_world_sensor.matrix().allFinite() || odom_stamps.empty() ||
      stamp < odom_stamps.front() || stamp > odom_stamps.back()) return;
  const auto found = std::lower_bound(odom_stamps.begin(), odom_stamps.end(), stamp);
  const int idx = std::distance(odom_stamps.begin(), found);

  if (std::abs(stamp - odom_stamps[idx]) < 1e-6) {
    T_world_odom = T_world_sensor * T_odom_sensor[idx].inverse();
  } else {
    const double t0 = odom_stamps[idx - 1];
    const double t1 = odom_stamps[idx];
    if (t0 > stamp || t1 < stamp) {
      // T_world_odom = T_world_sensor * T_odom_sensor[idx].inverse();
      return;
    }

    const double p = (stamp - t0) / (t1 - t0);

    const Eigen::Isometry3d T0 = T_odom_sensor[idx - 1];
    const Eigen::Isometry3d T1 = T_odom_sensor[idx];
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = T0.translation() * (1.0 - p) + T1.translation() * p;
    T.linear() = Eigen::Quaterniond(T0.linear()).slerp(p, Eigen::Quaterniond(T1.linear())).toRotationMatrix();
    T_world_odom = T_world_sensor * T.inverse();
  }

  if (idx > 1) {
    odom_stamps.erase(odom_stamps.begin(), odom_stamps.begin() + idx - 1);
    T_odom_sensor.erase(T_odom_sensor.begin(), T_odom_sensor.begin() + idx - 1);
  }
}

Eigen::Isometry3d TrajectoryManager::current_pose() const {
  if (T_odom_sensor.empty()) return T_world_odom;
  return T_world_odom * T_odom_sensor.back();
}

Eigen::Isometry3d TrajectoryManager::odom2world(const Eigen::Isometry3d& pose) const {
  return T_world_odom * pose;
}

Eigen::Vector3d TrajectoryManager::odom2world(const Eigen::Vector3d& point) const {
  return T_world_odom * point;
}

const Eigen::Isometry3d TrajectoryManager::get_T_world_odom() const {
  return T_world_odom;
}

}  // namespace glim
