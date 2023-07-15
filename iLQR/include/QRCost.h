#include "stdint.h"
#include <Eigen/Dense>

class QRCost
{
private:
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Q_plus_Q_T_;
    Eigen::MatrixXd R_plus_R_T_;

    Eigen::VectorXd x_goal_;
    Eigen::VectorXd u_goal_;

    uint16_t state_dim_;
    uint16_t ctrl_dim_;
public:
    QRCost(Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::VectorXd x_goal, Eigen::VectorXd u_goal);

    double l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    void l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    void l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    void l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    void l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    void l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);
};
