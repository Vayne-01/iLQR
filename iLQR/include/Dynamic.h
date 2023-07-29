#include "stdint.h"
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

class Dynamic
{
protected:
    /* data */
    uint16_t state_dim_;
    uint16_t ctrl_dim_;
public:
    Dynamic(uint16_t state_dim, uint16_t ctrl_dim): state_dim_(state_dim), ctrl_dim_(ctrl_dim) {}
    virtual autodiff::VectorXreal state_space(const autodiff::VectorXreal& state, const autodiff::VectorXreal& ctrl) = 0;

    void Jacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& ctrl, Eigen::VectorXd& out
                  , Eigen::MatrixXd& jacob_state, Eigen::MatrixXd& jacob_ctrl);
};

class MultiVehicle: public Dynamic
{
private:
    double dt_;     // Discrete time step.
    double m_;      // Mass.
    double alpha_;  // Friction coefficient.
public:
    MultiVehicle(uint16_t state_dim, uint16_t ctrl_dim, double dt, double m, double alpha):
                     Dynamic(state_dim, ctrl_dim), dt_(dt), m_(m), alpha_(alpha){}

    virtual autodiff::VectorXreal state_space(const autodiff::VectorXreal& state, const autodiff::VectorXreal& ctrl);

    autodiff::real acceleration(const autodiff::real x_dot, const autodiff::real u);
};

class MassPoint: public Dynamic
{
private:
    double dt_;     // Discrete time step.
    double m_;      // Mass.
public:
    MassPoint(uint16_t state_dim, uint16_t ctrl_dim, double dt, double m): Dynamic(state_dim, ctrl_dim), dt_(dt), m_(m){}

    virtual autodiff::VectorXreal state_space(const autodiff::VectorXreal& state, const autodiff::VectorXreal& ctrl);
};
