#include <stack>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Dynamic.h"
#include "QRCost.h"

class iLQR
{
private:
    Dynamic* model_;
    Cost*  cost_;
    uint16_t state_dim_;
    uint16_t ctrl_dim_;
    uint16_t N_;

    double _mu;
    double _mu_min;
    double _mu_max;
    double _delta_0;
    double _delta;
public:
    iLQR(uint16_t state_dim, uint16_t ctrl_dim, uint16_t N, Dynamic* model, Cost* cost
         , double max_reg = 1e10);

    void fit(Eigen::VectorXd& x0, std::vector<Eigen::VectorXd>& us, std::vector<Eigen::VectorXd>& xs,
             uint16_t n_iterations = 100, double tol = 1e-6, bool on_iteration = true);

    void _control(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us
                  , std::vector<Eigen::VectorXd>& xs_new, std::vector<Eigen::VectorXd>& us_new
                  , std::vector<Eigen::VectorXd>& k , std::vector<Eigen::MatrixXd>& K, double alpha);

    void _forward_rollout(const Eigen::VectorXd& x0, const std::vector<Eigen::VectorXd>& us
                          , std::vector<Eigen::VectorXd>& xs, std::vector<Eigen::MatrixXd>& F_x
                          , std::vector<Eigen::MatrixXd>& F_u, std::vector<double>& L
                          , std::vector<Eigen::VectorXd>& L_x, std::vector<Eigen::VectorXd>& L_u
                          , std::vector<Eigen::MatrixXd>& L_xx, std::vector<Eigen::MatrixXd>& L_ux
                          , std::vector<Eigen::MatrixXd>& L_uu);

    void _backward_pass(const std::vector<Eigen::MatrixXd>& F_x, const std::vector<Eigen::MatrixXd>& F_u
                        , const std::vector<Eigen::VectorXd>& L_x, const std::vector<Eigen::VectorXd>& L_u
                        , const std::vector<Eigen::MatrixXd>& L_xx, const std::vector<Eigen::MatrixXd>& L_ux
                        , const std::vector<Eigen::MatrixXd>& L_uu, std::vector<Eigen::VectorXd>& k
                        , std::vector<Eigen::MatrixXd>& K);

    void _Q(const Eigen::MatrixXd& f_x, const Eigen::MatrixXd& f_u
            , const Eigen::VectorXd& l_x, const Eigen::VectorXd& l_u
            , const Eigen::MatrixXd& l_xx, const Eigen::MatrixXd& l_ux
            , const Eigen::MatrixXd& l_uu, const Eigen::VectorXd& V_x
            , const Eigen::MatrixXd& V_xx, Eigen::VectorXd& Q_x
            , Eigen::VectorXd& Q_u, Eigen::MatrixXd& Q_xx
            , Eigen::MatrixXd& Q_ux, Eigen::MatrixXd& Q_uu);
};
