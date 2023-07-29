// C++ includes
#define _USE_MATH_DEFINES
#include <iostream>
#include "iLQR.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
    // init th Multi-Vehicle model.
    MultiVehicle model(8, 4, 0.1, 1.0, 0.1);

    // set cost function and target state.
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

    // set init state and solver.
    uint16_t N = 200;
    Eigen::VectorXd x0(8);
    x0 << 0.0, 0.0, 10.0, 10.0, 0.0, -5.0, 5.0, 0.0;
    std::vector<Eigen::VectorXd> us;
    std::vector<Eigen::VectorXd> xs;
    us.clear();
    for(uint16_t i = 0; i < N; i++)
    {
        us.emplace_back(Eigen::VectorXd::Random(4));
    }
    iLQR lqr_sover(8, 4, N, &model, &cost);
    lqr_sover.fit(x0, us, xs);

    // display the solve result.
    std::vector<double> vehicle1_x, vehicle1_y, vehicle2_x, vehicle2_y;
    for(uint16_t i = 0; i < xs.size(); i++)
    {
        vehicle1_x.emplace_back(xs[i](0));
        vehicle1_y.emplace_back(xs[i](1));
        vehicle2_x.emplace_back(xs[i](2));
        vehicle2_y.emplace_back(xs[i](3));
    }
    plt::named_plot("Vehicle 1", vehicle1_x, vehicle1_y, "r");
    plt::named_plot("Vehicle 2", vehicle2_x, vehicle2_y, "b");
    plt::title("Trajectory of the two omnidirectional vehicles");
    plt::legend();
    // save figure
    const char* filename = "../multivehicle.png";
    std::cout << "Saving result to " << filename << std::endl;;
    plt::save(filename);
}
