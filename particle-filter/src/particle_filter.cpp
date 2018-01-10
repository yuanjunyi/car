#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: 1. Set the number of particles.
    //       2. Initialize all particles to first position (based on estimates of 
    //          x, y, theta and their uncertainties from GPS) and all weights to 1. 
    //       3. Add random Gaussian noise to each particle.

    num_particles = 100;

    weights.clear();
    weights.insert(weights.begin(), num_particles, 1);

    particles.clear();
    particles.insert(particles.begin(), num_particles, Particle());

    default_random_engine generator;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        particles[i].x = dist_x(generator);
        particles[i].y = dist_y(generator);
        particles[i].theta = dist_theta(generator);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.

    default_random_engine generator;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; ++i) {
        const double theta = particles[i].theta;
        const double new_theta = theta + yaw_rate * delta_t;
        particles[i].x = particles[i].x / yaw_rate * (sin(new_theta) - sin(theta)) + dist_x(generator);
        particles[i].y = particles[i].y / yaw_rate * (cos(theta) - cos(new_theta)) + dist_y(generator);
        particles[i].theta = new_theta + dist_theta(generator);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); ++i)
    {
        double min_distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
        int min_distance_id = predicted[0].id;
        for (int j = 1; j < predicted.size(); ++j)
        {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (distance < min_distance) {
                min_distance = distance;
                min_distance_id = predicted[j].id;
            }
        }
        observations[i].id = min_distance_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    for (auto& particle: particles)
    {
        vector<LandmarkObs> predicted;
        for (auto landmark : map_landmarks.landmark_list)
        {
            if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
                LandmarkObs obs;
                obs.id = landmark.id_i;
                obs.x = landmark.x_f;
                obs.y = landmark.y_f;
                predicted.push_back(obs);
            }
        }

        vector<LandmarkObs> observations_map;
        for (auto obs_car : observations)
        {
            LandmarkObs obs_map;
            obs_map.x = particle.x + cos(particle.theta) * obs_car.x - sin(particle.theta) * obs_car.y;
            obs_map.y = particle.y + sin(particle.theta) * obs_car.x + cos(particle.theta) * obs_car.y;
            observations_map.push_back(obs_map);
        }

        dataAssociation(predicted, observations_map);

        double weight = 1;
        for (auto obs_map : observations_map)
        {
            int nearest_landmark_id = obs_map.id;
            LandmarkObs nearest = SearchLandmark(map_landmarks, nearest_landmark_id);

            weight *= ComputeMultivariateGaussianProbability(obs_map.x,
                                                             obs_map.y,
                                                             nearest.x,
                                                             nearest.y,
                                                             std_landmark[0],
                                                             std_landmark[1]);
        }
        particle.weight = weight;
    }
}

double ParticleFilter::ComputeMultivariateGaussianProbability(double x,
                                                              double y,
                                                              double mux,
                                                              double muy,
                                                              double sigmax,
                                                              double sigmay) {
    double gauss_norm = 1 / (2 * M_PI * sigmax * sigmay);
    double exponent = (x-mux)*(x-mux)/(2*sigmax*sigmax) + (y-muy)*(y-muy)/(2*sigmay*sigmay);
    return gauss_norm * exp(-exponent);
}

LandmarkObs ParticleFilter::ToLandmarkObs(const Map::single_landmark_s& maplandmark) {
    LandmarkObs obs;
    obs.id = maplandmark.id_i;
    obs.x = maplandmark.x_f;
    obs.y = maplandmark.y_f;
    return obs;
}

LandmarkObs ParticleFilter::SearchLandmark(const Map& map, int id) {
    for (auto maplandmark : map.landmark_list) {
        if (maplandmark.id_i == id) {
            return ToLandmarkObs(maplandmark);
        }
    }
    return LandmarkObs();
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
