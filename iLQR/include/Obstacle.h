#include <Eigen/Dense>

struct CircleObstacle
{
    Eigen::VectorXd p_c_;
    Eigen::VectorXd r_;
    CircleObstacle(Eigen::VectorXd p_c, double r): p_c_(p_c)
    {
        r_.resize(p_c_.size());
        r_(0) = r;
    }
};
