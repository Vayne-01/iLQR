#include "stdint.h"
#include "Obstacle.h"

class Cost
{
public:
    virtual double l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;

    virtual void l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;

    virtual void l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;

    virtual void l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;

    virtual void l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;

    virtual void l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false) = 0;
};


class QRCost: public Cost
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

    virtual double l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);
};

class PotentialCost: public Cost
{
private:
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Q_term_;
    Eigen::MatrixXd Q_repell_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Q_plus_Q_T_;
    Eigen::MatrixXd Q_plus_Q_T_term_;
    Eigen::MatrixXd Q_plus_Q_T_repell_;
    Eigen::MatrixXd R_plus_R_T_;

    Eigen::VectorXd x_goal_;

    uint16_t state_dim_;
    uint16_t ctrl_dim_;
    std::vector<CircleObstacle>* obstacle_;

public:
    PotentialCost(Eigen::MatrixXd Q, Eigen::MatrixXd Q_term, Eigen::MatrixXd Q_repell, Eigen::MatrixXd R, Eigen::VectorXd x_goal, std::vector<CircleObstacle>* obstacle);

    virtual double l(Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_x(Eigen::VectorXd& lx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_u(Eigen::VectorXd& lu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_xx(Eigen::MatrixXd& lxx, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_ux(Eigen::MatrixXd& lux, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);

    virtual void l_uu(Eigen::MatrixXd& luu, Eigen::VectorXd x, Eigen::VectorXd u, uint16_t i, bool is_terminal = false);
};
