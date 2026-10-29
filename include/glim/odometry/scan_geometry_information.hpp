#pragma once
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

namespace glim {
inline Eigen::Matrix<double,6,6> registration_mean_information(
    const Eigen::Matrix<double,6,6>& summed, size_t count) {
  if (!count || !summed.allFinite() || (summed-summed.transpose()).norm()>1e-8*std::max(1.,summed.norm()))
    throw std::invalid_argument("invalid registration curvature");
  const Eigen::Matrix<double,6,6> mean=(summed+summed.transpose())/(2.*count);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eigen(mean);
  if (eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff() < -1e-8*std::max(1.,mean.norm()))
    throw std::invalid_argument("registration curvature is not positive semidefinite");
  return eigen.eigenvectors()*eigen.eigenvalues().cwiseMax(1e-6).asDiagonal()*eigen.eigenvectors().transpose();
}

// Mean point-to-plane Jacobian information in the source/right-increment chart.
// A plane cannot measure translation along itself or rotation about its normal.
// Normalize by count so repeated correlated returns do not inflate confidence.
// This shapes the existing scalar precision; it is NOT calibrated covariance
// or a correspondence/outlier validity certificate.
inline Eigen::Matrix<double,6,6> scan_geometry_information(
    const Eigen::Vector4d* points, const Eigen::Vector4d* normals, size_t count) {
  if (!count || !points || !normals) throw std::invalid_argument("missing scan geometry");
  Eigen::Matrix<double,6,6> h = Eigen::Matrix<double,6,6>::Zero();
  for (size_t i=0;i<count;++i) {
    const Eigen::Vector3d p=points[i].head<3>(), n=normals[i].head<3>();
    if (!p.allFinite() || !n.allFinite() || std::abs(n.norm()-1.)>1e-6)
      throw std::invalid_argument("invalid point/normal geometry");
    Eigen::Matrix<double,6,1> j;
    j << p.cross(n), n;
    h.noalias() += j*j.transpose();
  }
  h /= count;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eigen(h);
  if (eigen.info()!=Eigen::Success) throw std::runtime_error("scan geometry decomposition failed");
  // Numerical conditioning only: six orders below unit translation information.
  return eigen.eigenvectors()*eigen.eigenvalues().cwiseMax(1e-6).asDiagonal()*eigen.eigenvectors().transpose();
}
}
