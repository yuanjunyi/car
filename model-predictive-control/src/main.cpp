#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <limits>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

using namespace Eigen;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Derivative of coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x
double derivative(Eigen::VectorXd coeffs, double x) {
  return coeffs[1] + 2*coeffs[2]*x + 3*coeffs[3]*x*x;
}

double distance(double x1, double y1, double x2, double y2) {
  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

void transform(double x0, double y0, double psi, double& x, double& y) {
  const double alpha = atan2(y - y0, x - x0);
  const double beta = psi - alpha;
  const double d = distance(x0, y0, x, y);
  x = d * cos(beta);
  y = -d * sin(beta);
}

int main() {
  uWS::Hub h;
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          v *= 0.44704; // Convert from mph to m/s

          // Compute the vehicle position after 100ms latency
          const double dt = 0.1;  // 100ms = 0.1s
          px = px + v * cos(psi) * dt;
          py = py + v * sin(psi) * dt;

          // Find the nearest waypoint
          double minDistance = numeric_limits<double>::max();
          size_t nearest = 0;
          for (size_t i = 0; i < ptsx.size(); ++i) {
            double d = distance(px, py, ptsx[i], ptsy[i]);
            if (d < minDistance) {
              minDistance = d;
              nearest = i;
            }
          }

          // Discard the waypoints that are behind the vehicle
          if (nearest != 0) {
            ptsx.erase(ptsx.cbegin(), ptsx.cbegin() + nearest);
            ptsy.erase(ptsy.cbegin(), ptsy.cbegin() + nearest);
          }

          // Transform waypoints from map coordinate to car coordinate
          vector<double> waypointx;
          vector<double> waypointy;
          for (size_t i = 0; i < ptsx.size(); ++i)
          {
            double x = ptsx[i];
            double y = ptsy[i];
            transform(px, py, psi, x, y);
            waypointx.push_back(x);
            waypointy.push_back(y);
          }

          // Transfrom car position and orientation to car coordinate
          px = 0;
          py = 0;
          psi = 0;

          // Fit 3rd order polynomial and pass to MPC
          Map<VectorXd> ptsx_eigen(waypointx.data(), waypointx.size());
          Map<VectorXd> ptsy_eigen(waypointy.data(), waypointy.size());
          VectorXd coeffs = polyfit(ptsx_eigen, ptsy_eigen, 3);
          const double cte = polyeval(coeffs, px) - py;
          const double epsi = psi - atan(derivative(coeffs, px));
          VectorXd state(6);
          state << px, py, psi, v, cte, epsi;
          auto solution = mpc.Solve(state, coeffs);

          // Acceleration from MPC is in m/s^2.
          // Throttle in simulator is unknown but seems quite close to m/s^2.
          double throttle_value = solution.back();
          solution.pop_back();

          // Delta from MPC is in radian.
          // Steering angle in simulator is also in radian.
          double steer_value = -solution.back();
          solution.pop_back();

          json msgJson;

          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          cout << " steer: " << steer_value
               << " throttle: " << throttle_value << endl;

          vector<double> mpc_x_vals(solution.begin(), solution.begin() + solution.size() / 2);
          vector<double> mpc_y_vals(solution.begin() + solution.size() / 2, solution.end());

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          msgJson["next_x"] = waypointx;
          msgJson["next_y"] = waypointy;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
