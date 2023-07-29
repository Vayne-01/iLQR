#include "Dynamic.h"

void Dynamic::Jacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& ctrl, Eigen::VectorXd& out
                       , Eigen::MatrixXd& jacob_state, Eigen::MatrixXd& jacob_ctrl)
{
    autodiff::VectorXreal autodiff_state(state), autodiff_ctrl(ctrl), autodiff_out;
        
    jacob_state = jacobian(std::bind(&Dynamic::state_space, this, std::placeholders::_1, std::placeholders::_2),
                           autodiff::wrt(autodiff_state), autodiff::at(autodiff_state, autodiff_ctrl), autodiff_out);
    jacob_ctrl = jacobian(std::bind(&Dynamic::state_space, this, std::placeholders::_1, std::placeholders::_2),
                          autodiff::wrt(autodiff_ctrl), autodiff::at(autodiff_state, autodiff_ctrl), autodiff_out);
    out.resize(state_dim_);
    out.block(0, 0, state_dim_, 1) = autodiff_out.cast<double>();
}

autodiff::VectorXreal MultiVehicle::state_space(const autodiff::VectorXreal& state
                                                , const autodiff::VectorXreal& ctrl)
{
    autodiff::VectorXreal out(state_dim_);
    out(0) = state(0) + state(4) * dt_;
    out(1) = state(1) + state(5) * dt_;
    out(2) = state(2) + state(6) * dt_;
    out(3) = state(3) + state(7) * dt_;
    out(4) = state(4) + acceleration(state(4), ctrl(0)) * dt_;
    out(5) = state(5) + acceleration(state(5), ctrl(1)) * dt_;
    out(6) = state(6) + acceleration(state(6), ctrl(2)) * dt_;
    out(7) = state(7) + acceleration(state(7), ctrl(3)) * dt_;
        
    return out;
}

autodiff::real MultiVehicle::acceleration(const autodiff::real x_dot, const autodiff::real u)
{
    return x_dot * (1.0 - alpha_ * dt_ / m_) + u * dt_ / m_;
}

autodiff::VectorXreal MassPoint::state_space(const autodiff::VectorXreal& state
                                             , const autodiff::VectorXreal& ctrl)
{
    autodiff::VectorXreal out(state_dim_);
    out(0) = state(0) + state(2) * dt_ + 0.5 * ctrl(0) / m_ * dt_ * dt_;
    out(1) = state(1) + state(3) * dt_ + 0.5 * ctrl(1) / m_ * dt_ * dt_;
    out(2) = state(2) + ctrl(0) / m_ * dt_;
    out(3) = state(3) + ctrl(1) / m_ * dt_;

    return out;
}
