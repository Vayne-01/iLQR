#include "QRCost.h"

QRCost::QRCost(Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::VectorXd x_goal, Eigen::VectorXd u_goal):
               Q_(Q), R_(R), x_goal_(x_goal), u_goal_(u_goal)
{
    state_dim_ = Q_.rows();
    ctrl_dim_  = R_.rows();
    Q_plus_Q_T_ = Q_ + Q_.transpose();
    R_plus_R_T_ = R_ + R_.transpose();
}

double QRCost::l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    Eigen::VectorXd x_diff = x - x_goal_;
    if(is_terminal)
    {
        return x_diff.dot(Q_ * x_diff);
    }
    Eigen::VectorXd u_diff = u - u_goal_;
    return x_diff.dot(Q_ * x_diff) + u_diff.dot(R_ * u_diff);
}

void QRCost::l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u
                 , uint16_t i, bool is_terminal)
{
    Eigen::VectorXd x_diff = x - x_goal_;
    lx = Q_plus_Q_T_ * x_diff;
}

void QRCost::l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u
                 , uint16_t i, bool is_terminal)
{
    if(is_terminal)
    {
        lu.resize(ctrl_dim_);
        lu.setZero();
    }
    else
    {
        Eigen::VectorXd u_diff = u - u_goal_;
        lu = R_plus_R_T_ * u_diff;
    }
}

void QRCost::l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u
                  , uint16_t i, bool is_terminal)
{
    lxx = Q_plus_Q_T_;
}

void QRCost::l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u
                  , uint16_t i, bool is_terminal)
{
    lux.resize(ctrl_dim_, state_dim_);
    lux.setZero();
}

void QRCost::l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u
                  , uint16_t i, bool is_terminal)
{
    if(is_terminal)
    {
        luu.resize(ctrl_dim_, ctrl_dim_);
        luu.setZero();
    }
    else
    {
        luu = R_plus_R_T_;
    }
}

PotentialCost::PotentialCost(Eigen::MatrixXd Q, Eigen::MatrixXd Q_term, Eigen::MatrixXd Q_repell, Eigen::MatrixXd R
                             , Eigen::VectorXd x_goal, std::vector<CircleObstacle>* obstacle):
                             Q_(Q), Q_term_(Q_term), Q_repell_(Q_repell), R_(R), x_goal_(x_goal), obstacle_(obstacle)
{
    state_dim_ = Q_.rows();
    ctrl_dim_  = R_.rows();
    Q_plus_Q_T_        = Q_ + Q_.transpose();
    Q_plus_Q_T_term_   = Q_term_ + Q_term_.transpose();
    Q_plus_Q_T_repell_ = Q_repell_ + Q_repell_.transpose();
    R_plus_R_T_        = R_ + R_.transpose();
}

double PotentialCost::l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    Eigen::VectorXd x_diff = x - x_goal_;
    double cost = 0.0;
    if(!is_terminal)
    {
        cost += x_diff.dot(Q_ * x_diff);
        cost += u.dot(R_ * u);
    }
    else
    {
        cost += x_diff.dot(Q_term_ * x_diff);
    }
    for(uint16_t k = 0; k < obstacle_->size(); k++)
    {
        double cost_repell = (*obstacle_)[k].r_.dot(Q_repell_ * (*obstacle_)[k].r_) 
                             - (x - (*obstacle_)[k].p_c_).dot(Q_repell_ * (x - (*obstacle_)[k].p_c_));
        if(cost_repell > 0)
            cost += cost_repell;
    }
    return cost;
}

void PotentialCost::l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    Eigen::VectorXd x_diff = x - x_goal_;
    if(!is_terminal)
        lx = Q_plus_Q_T_ * x_diff;
    else
        lx = Q_plus_Q_T_term_ * x_diff;
    for(uint16_t k = 0; k < obstacle_->size(); k++)
    {
        double cost_repell = (*obstacle_)[k].r_.dot(Q_repell_ * (*obstacle_)[k].r_) 
                             - (x - (*obstacle_)[k].p_c_).dot(Q_repell_ * (x - (*obstacle_)[k].p_c_));
        if(cost_repell > 0)
            lx -= Q_plus_Q_T_repell_ * (x - (*obstacle_)[k].p_c_);
    }
}

void PotentialCost::l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    if(is_terminal)
    {
        lu.resize(ctrl_dim_);
        lu.setZero();
    }
    else
    {
        lu = R_plus_R_T_ * u;
    }
}

void PotentialCost::l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    if(!is_terminal)
        lxx = Q_plus_Q_T_;
    else
        lxx = Q_plus_Q_T_term_;
    for(uint16_t k = 0; k < obstacle_->size(); k++)
    {
        double cost_repell = (*obstacle_)[k].r_.dot(Q_repell_ * (*obstacle_)[k].r_) 
                             - (x - (*obstacle_)[k].p_c_).dot(Q_repell_ * (x - (*obstacle_)[k].p_c_));
        if(cost_repell > 0)
            lxx -= Q_plus_Q_T_repell_;
    }
}
void PotentialCost::l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    lux.resize(ctrl_dim_, state_dim_);
    lux.setZero();
}

void PotentialCost::l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal)
{
    if(is_terminal)
    {
        luu.resize(ctrl_dim_, ctrl_dim_);
        luu.setZero();
    }
    else
    {
        luu = R_plus_R_T_;
    }
}
