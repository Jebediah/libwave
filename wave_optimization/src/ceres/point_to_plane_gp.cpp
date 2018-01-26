#include "wave/optimization/ceres/point_to_plane_gp.hpp"

namespace wave {

SE3PointToPlaneGP::SE3PointToPlaneGP(const double *const p,
                                     const double *const pA,
                                     const double *const pB,
                                     const double *const pC,
                                     const Eigen::Matrix<double, 6, 12> &hat,
                                     const Eigen::Matrix<double, 6, 12> &candle,
                                     const Mat3 &covZ,
                                     bool use_weighting)
    : pt(p),
      ptA(pA),
      ptB(pB),
      ptC(pC),
      hat(hat),
      candle(candle) {
    this->JP_T.setZero();
    this->JP_T.block<3, 3>(0, 3).setIdentity();
    this->Jr_Tk.block<1, 6>(0, 6).setZero();
    this->Jr_Tkp1.block<1, 6>(0, 6).setZero();

    this->calculateJr_P(this->Jr_P);

    if (use_weighting) {
        double covR = (Jr_P * covZ * Jr_P.transpose())(0);
        this->weight = std::sqrt(1.0 / covR);
    } else {
        this->weight = 1;
    }

    double dBA[3] = {ptB[0] - ptA[0], ptB[1] - ptA[1], ptB[2] - ptA[2]};
    double dBC[3] = {ptB[0] - ptC[0], ptB[1] - ptC[1], ptB[2] - ptC[2]};

    ceres::CrossProduct(dBA, dBC, this->cBA_BC);

    this->inv_den = 1.0 / ceres::sqrt(cBA_BC[0] * cBA_BC[0] + cBA_BC[1] * cBA_BC[1] + cBA_BC[2] * cBA_BC[2]);
}

bool SE3PointToPlaneGP::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 3, 4>> Tk_mat(parameters[0], 3, 4);
    Eigen::Map<const Eigen::Matrix<double, 3, 4>> Tkp1_mat(parameters[1], 3, 4);
    Eigen::Map<const Vec6> vel_k(parameters[2], 6, 1);
    Eigen::Map<const Vec6> vel_kp1(parameters[3], 6, 1);

    // todo(ben) Template transformation class to avoid copy operation
    Transformation Tk(Tk_mat);
    Transformation Tkp1(Tkp1_mat);

    if (jacobians) {
        this->T_current = Transformation::interpolateAndJacobians(Tk, Tkp1, vel_k, vel_kp1, this->hat, this->candle,
                                                                  this->JT_Ti, this->JT_Tip1, this->JT_Wi, this->JT_Wip1);
    } else {
        this->T_current = Transformation::interpolate(Tk, Tkp1, vel_k, vel_kp1, this->hat, this->candle);
    }

    Eigen::Map<const Vec3> PT(this->pt, 3, 1);
    Vec3 point = this->T_current.transform(PT);

    // point is the transformed point.
    double d_B[3] = {point(0) - ptB[0], point(1) - ptB[1], point(2) - ptB[2]};

    double num = this->cBA_BC[0] * d_B[0] + this->cBA_BC[1] * d_B[1] + this->cBA_BC[2] * d_B[2];

    residuals[0] = this->weight * (num * (this->inv_den));

    if (jacobians) {
        this->JP_T(0,1) = point(2);
        this->JP_T(0,2) = -point(1);
        this->JP_T(1,0) = -point(2);
        this->JP_T(1,2) = point(0);
        this->JP_T(2,0) = point(1);
        this->JP_T(2,1) = -point(0);


        this->Jr_T = this->Jr_P * this->JP_T;
        if (jacobians[0]) {
            this->Jr_Tk.block<1,6>(0,0) = this->weight * this->Jr_T * this->JT_Ti;
            Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>>(jacobians[0], 1, 12) = this->Jr_Tk;
        }
        if (jacobians[1]) {
            this->Jr_Tkp1.block<1,6>(0,0) = this->weight * this->Jr_T * this->JT_Tip1;
            Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>>(jacobians[1], 1, 12) = this->Jr_Tkp1;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jac_map(jacobians[2], 1, 6);
            jac_map = this->weight * this->Jr_T * this->JT_Wi;
        }
        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jac_map(jacobians[3], 1, 6);
            jac_map = this->weight * this->Jr_T * this->JT_Wip1;
        }
    }

    return true;
}

/// Jacobian of the residual wrt the transformed point
void SE3PointToPlaneGP::calculateJr_P(Eigen::Matrix<double, 1, 3> &Jr_P) const {
    const double &XA1 = this->ptA[0], &XA2 = this->ptA[1], &XA3 = this->ptA[2], &XB1 = this->ptB[0],
                 &XB2 = this->ptB[1], &XB3 = this->ptB[2], &XC1 = this->ptC[0], &XC2 = this->ptC[1],
                 &XC3 = this->ptC[2];

    Jr_P << -((XA2 - XB2) * (XB3 - XC3) - (XA3 - XB3) * (XB2 - XC2)) * 1.0 /
              sqrt(pow(((XA1 - XB1) * (XB2 - XC2) - (XA2 - XB2) * (XB1 - XC1)), 2) +
                   pow(((XA1 - XB1) * (XB3 - XC3) - (XA3 - XB3) * (XB1 - XC1)), 2) +
                   pow(((XA2 - XB2) * (XB3 - XC3) - (XA3 - XB3) * (XB2 - XC2)), 2)),
      ((XA1 - XB1) * (XB3 - XC3) - (XA3 - XB3) * (XB1 - XC1)) * 1.0 /
        sqrt(pow(((XA1 - XB1) * (XB2 - XC2) - (XA2 - XB2) * (XB1 - XC1)), 2) +
             pow(((XA1 - XB1) * (XB3 - XC3) - (XA3 - XB3) * (XB1 - XC1)), 2) +
             pow(((XA2 - XB2) * (XB3 - XC3) - (XA3 - XB3) * (XB2 - XC2)), 2)),
      -((XA1 - XB1) * (XB2 - XC2) - (XA2 - XB2) * (XB1 - XC1)) * 1.0 /
        sqrt(pow(((XA1 - XB1) * (XB2 - XC2) - (XA2 - XB2) * (XB1 - XC1)), 2) +
             pow(((XA1 - XB1) * (XB3 - XC3) - (XA3 - XB3) * (XB1 - XC1)), 2) +
             pow(((XA2 - XB2) * (XB3 - XC3) - (XA3 - XB3) * (XB2 - XC2)), 2));
}

}  // namespace wave
