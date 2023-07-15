// C++ includes
#include <iostream>
#include "iLQR.h"

int main()
{
    ParticleDynamic model(8, 4, 0.1, 1.0, 0.1);

    Eigen::MatrixXd Q;
    Q.setIdentity(8, 8);
    Q(0, 2) = Q(2, 0) = -1;
    Q(1, 3) = Q(3, 1) = -1;
    Eigen::MatrixXd R;
    R.setIdentity(4, 4);
    R *= 0.1;
    Eigen::VectorXd x_goal(8);
    x_goal.setZero();
    Eigen::VectorXd u_goal(4);
    u_goal.setZero();
    QRCost cost(Q, R, x_goal, u_goal);

    uint16_t N = 200;
    Eigen::VectorXd x0(8);
    x0 << 0.0, 0.0, 10.0, 10.0, 0.0, -5.0, 5.0, 0.0;
    std::vector<Eigen::VectorXd>* us = new std::vector<Eigen::VectorXd>;
    std::vector<Eigen::VectorXd>* xs = new std::vector<Eigen::VectorXd>;
    us->clear();
    for(uint16_t i = 0; i < N; i++)
    {
        us->emplace_back(Eigen::VectorXd::Random(4));
    }
    iLQR lqr_sover(8, 4, N, &model, &cost);
    lqr_sover.fit(x0, us, xs);
}

