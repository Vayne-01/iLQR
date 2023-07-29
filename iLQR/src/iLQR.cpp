#include "iLQR.h"
#include <string>
#define _USE_MATH_DEFINES
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

iLQR::iLQR(uint16_t state_dim, uint16_t ctrl_dim, uint16_t N, Dynamic* model, Cost* cost
           , double max_reg): state_dim_(state_dim), ctrl_dim_(ctrl_dim), N_(N)
{
    model_ = model;
    cost_  = cost;

    _mu = 1.0;
    _mu_min = 1e-6;
    _mu_max = max_reg;
    _delta_0 = 2.0;
    _delta = _delta_0;
}

void iLQR::fit(Eigen::VectorXd& x0, std::vector<Eigen::VectorXd>& us, std::vector<Eigen::VectorXd>& xs,
             uint16_t n_iterations, double tol, bool on_iteration)
{
    _mu = 1.0;
    _delta = _delta_0;

    double alphas[10] = {0};
    alphas[0] = 1.0;
    for(uint8_t i = 1; i < 10; i++)
    {
        alphas[i] = alphas[i-1] / 1.1;
    }

    bool changed = true;
    bool converged = false;

    std::vector<Eigen::VectorXd> L_x, L_u, k;
    std::vector<Eigen::MatrixXd> F_x, F_u, L_xx, L_ux, L_uu, K;
    std::vector<double> L;
    double J_opt = 0.0f;
    for(uint16_t i = 0; i < n_iterations; i++)
    {
        bool accepted = false;

        if(changed)
        {
            _forward_rollout(x0, us, xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu);
            J_opt = 0.0;
            for(uint16_t j = 0; j < N_+1; j++)
            {
                J_opt += L[j];
            }
            changed = false;
        }
        _backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu, k, K);
        for(uint8_t j = 0; j < 10; j++)
        {
            std::vector<Eigen::VectorXd> xs_new;
            std::vector<Eigen::VectorXd> us_new;
            _control(xs, us, xs_new, us_new, k, K, alphas[j]);

            double J_new = 0.0;
            for(uint16_t i = 0; i < N_; i++)
            {
                J_new += cost_->l(xs_new[i], us_new[i], i);
            }
            J_new += cost_->l(xs_new[N_], us_new[0], N_, true);

            if(J_new < J_opt)
            {
                /*
                std::vector<double> point_x, point_y;
                for(uint16_t i = 0; i < xs.size(); i++)
                {
                    point_x.emplace_back(xs_new[i](0));
                    point_y.emplace_back(xs_new[i](1));
                }
                plt::named_plot("mass point", point_x, point_y, "o");
                plt::title("Trajectory of the two omnidirectional vehicles");
                plt::xlim(-1, 6);
                plt::ylim(-2, 2);
                plt::legend();
                plt::show();
                */
                if(std::fabs((J_opt - J_new) / J_opt) < tol)
                    converged = true;

                J_opt = J_new;
                xs.swap(xs_new);
                us.swap(us_new);
                changed = true;

                _delta = std::min(1.0, _delta) / _delta_0;
                _mu *= _delta;
                if(_mu <= _mu_min)
                    _mu = 0.0;

                accepted = true;
                break;
            }
        }

        if(!accepted)
        {
            _delta = std::max(1.0, _delta) * _delta_0;
            _mu = std::max(_mu_min, _mu * _delta);
            if(_mu_max && (_mu >= _mu_max))
            {
                std::cout << "exceeded max regularization term" << std::endl;
                break;
            }
        }

        if(on_iteration)
        {
            std::string info = converged ? "converged" : (accepted ? "accepted" : "failed");
            std::cout << "iteration " << i << " " << info << " " << J_opt << std::endl;
        }

        if(converged)
            break;
    }
}

void iLQR::_control(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us
                    , std::vector<Eigen::VectorXd>& xs_new, std::vector<Eigen::VectorXd>& us_new
                    , std::vector<Eigen::VectorXd>& k , std::vector<Eigen::MatrixXd>& K, double alpha)
{
    xs_new.clear();
    us_new.clear();
    xs_new.emplace_back(xs[0]);
    for(uint16_t i = 0; i < N_; i++)
    {
        us_new.emplace_back(us[i] + alpha * k[i] + K[i] * (xs_new[i] - xs[i]));
        xs_new.emplace_back(model_->state_space(xs_new[i], us_new[i]).cast<double>());
    }
}

void iLQR::_forward_rollout(const Eigen::VectorXd& x0, const std::vector<Eigen::VectorXd>& us
                            , std::vector<Eigen::VectorXd>& xs, std::vector<Eigen::MatrixXd>& F_x
                            , std::vector<Eigen::MatrixXd>& F_u, std::vector<double>& L
                            , std::vector<Eigen::VectorXd>& L_x, std::vector<Eigen::VectorXd>& L_u
                            , std::vector<Eigen::MatrixXd>& L_xx, std::vector<Eigen::MatrixXd>& L_ux
                            , std::vector<Eigen::MatrixXd>& L_uu)
{
    xs.clear();
    F_x.clear();
    F_u.clear();
    L.clear();
    L_x.clear();
    L_u.clear();
    L_xx.clear();
    L_ux.clear();
    L_uu.clear();

    xs.emplace_back(x0);

    Eigen::VectorXd x;
    Eigen::VectorXd u;

    double L_1 = 0.0f;
    Eigen::VectorXd x_1, L_x_1, L_u_1;
    Eigen::MatrixXd F_x_1, F_u_1, L_xx_1, L_ux_1, L_uu_1;
    for(uint16_t i = 0; i < N_; i++)
    {
        x = xs[i];
        u = us[i];

            
        model_->Jacobian(x, u, x_1, F_x_1, F_u_1);
        xs.emplace_back(x_1);
        F_x.emplace_back(F_x_1);
        F_u.emplace_back(F_u_1);
            
        L_1 = cost_->l(x, u, i);
        cost_->l_x(L_x_1, x, u, i);
        cost_->l_u(L_u_1, x, u, i);
        cost_->l_xx(L_xx_1, x, u, i);
        cost_->l_ux(L_ux_1, x, u, i);
        cost_->l_uu(L_uu_1, x, u, i);
        L.emplace_back(L_1);
        L_x.emplace_back(L_x_1);
        L_u.emplace_back(L_u_1);
        L_xx.emplace_back(L_xx_1);
        L_ux.emplace_back(L_ux_1);
        L_uu.emplace_back(L_uu_1);
    }
    x = xs[N_];
    L_1 = cost_->l(x, u, N_, true);
    cost_->l_x(L_x_1, x, u, N_, true);
    cost_->l_xx(L_xx_1, x, u, N_, true);
    L.emplace_back(L_1);
    L_x.emplace_back(L_x_1);
    L_xx.emplace_back(L_xx_1);
}

void iLQR::_backward_pass(const std::vector<Eigen::MatrixXd>& F_x, const std::vector<Eigen::MatrixXd>& F_u
                          , const std::vector<Eigen::VectorXd>& L_x, const std::vector<Eigen::VectorXd>& L_u
                          , const std::vector<Eigen::MatrixXd>& L_xx, const std::vector<Eigen::MatrixXd>& L_ux
                          , const std::vector<Eigen::MatrixXd>& L_uu, std::vector<Eigen::VectorXd>& k
                          , std::vector<Eigen::MatrixXd>& K)
{
    k.clear();
    K.clear();
    k.resize(N_);
    K.resize(N_);

    Eigen::VectorXd V_x  = L_x[N_];
    Eigen::MatrixXd V_xx = L_xx[N_];

    for(int16_t i = N_-1; i >= 0; i--)
    {
        Eigen::VectorXd Q_x, Q_u;
        Eigen::MatrixXd Q_xx, Q_ux, Q_uu;
        _Q(F_x[i], F_u[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i]
           , V_x, V_xx, Q_x, Q_u, Q_xx, Q_ux, Q_uu);
            
        k[i] = -Q_uu.llt().solve(Q_u);
        K[i].resize(ctrl_dim_, state_dim_);
        for(uint16_t j = 0; j < state_dim_; j++)
        {
            K[i].block(0, j, ctrl_dim_, 1) = -Q_uu.llt().solve(Q_ux.block(0, j, ctrl_dim_, 1));
        }

        V_x = Q_x + K[i].transpose() * Q_uu * k[i];
        V_x += K[i].transpose() * Q_u + Q_ux.transpose() * k[i];

        V_xx = Q_xx + K[i].transpose() * Q_uu * K[i];
        V_xx += K[i].transpose() * Q_ux + Q_ux.transpose() * K[i];
        V_xx = 0.5 * (V_xx + V_xx.transpose());
    }
}

void iLQR::_Q(const Eigen::MatrixXd& f_x, const Eigen::MatrixXd& f_u
              , const Eigen::VectorXd& l_x, const Eigen::VectorXd& l_u
              , const Eigen::MatrixXd& l_xx, const Eigen::MatrixXd& l_ux
              , const Eigen::MatrixXd& l_uu, const Eigen::VectorXd& V_x
              , const Eigen::MatrixXd& V_xx, Eigen::VectorXd& Q_x
              , Eigen::VectorXd& Q_u, Eigen::MatrixXd& Q_xx
              , Eigen::MatrixXd& Q_ux, Eigen::MatrixXd& Q_uu)
{
    Q_x = l_x + f_x.transpose() * V_x;
    Q_u = l_u + f_u.transpose() * V_x;
    Q_xx = l_xx + f_x.transpose() * V_xx * f_x;

    Eigen::MatrixXd reg = _mu * Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    Q_ux = l_ux + f_u.transpose() * (V_xx + reg) * f_x;
    Q_uu = l_uu + f_u.transpose() * (V_xx + reg) * f_u;
}
