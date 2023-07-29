// C++ includes
#define _USE_MATH_DEFINES
#include <iostream>
#include "iLQR.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
    MassPoint model(4, 2, 0.05, 1.0);

    CircleObstacle obstacle1(Eigen::Vector<double, 4>(6.0, 5.0, 0.0, 0.0), 2.0);
    std::vector<CircleObstacle> obstacle;
    obstacle.clear();
    obstacle.emplace_back(obstacle1);

    Eigen::MatrixXd Q, Q_term, Q_repell, R;
    Q.setZero(4, 4);
    Q_term.setZero(4, 4);
    Q_term(0, 0) = 200.0;
    Q_term(1, 1) = 200.0;
    Q_term(2, 2) = 0.0;
    Q_term(3, 3) = 0.0;
    Q_repell.setZero(4, 4);
    Q_repell(0, 0) = 500.0;
    Q_repell(1, 1) = 500.0;
    R.setIdentity(2, 2);
    R *= 0.01;
    
    Eigen::VectorXd x_goal(4);
    x_goal << 12.0, 12.0, 0.0, 0.0;

    PotentialCost cost(Q, Q_term, Q_repell, R, x_goal, &obstacle);
    
    uint16_t N = 100;
    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.0, 0.0, 0.0;
    std::vector<Eigen::VectorXd> us;
    std::vector<Eigen::VectorXd> xs;
    us.clear();
    for(uint16_t i = 0; i < N; i++)
    {
        us.emplace_back(Eigen::VectorXd::Random(2));
    }
    iLQR lqr_sover(4, 2, N, &model, &cost);
    lqr_sover.fit(x0, us, xs, 500);

    std::vector<double> point_x, point_y;
    for(uint16_t i = 0; i < xs.size(); i++)
    {
        point_x.emplace_back(xs[i](0));
        point_y.emplace_back(xs[i](1));
    }
    std::vector<double> circle_x, circle_y;
    for(uint16_t i = 0; i < 101; i++)
    {
        circle_x.emplace_back(std::cos(0.0628 * double(i)) * 2.0 + 6.0);
        circle_y.emplace_back(std::sin(0.0628 * double(i)) * 2.0 + 5.0);
    }
    plt::named_plot("mass point", point_x, point_y, "b");
    plt::named_plot("obstacle", circle_x, circle_y, "r");
    plt::title("Trajectory of the two omnidirectional vehicles");
    plt::grid(true);
    plt::legend();

    // save figure
    const char* filename = "../mass_point_planning.png";
    std::cout << "Saving result to " << filename << std::endl;;
    plt::save(filename);
}