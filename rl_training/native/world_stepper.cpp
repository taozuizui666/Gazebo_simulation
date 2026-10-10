// Persistent Gazebo Classic transport client. ACK includes the completed
// physics time and contacts; Python also requires timestamped sensor data.
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <iostream>
#include <string>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iomanip>

std::mutex result_mutex;
std::condition_variable result_changed;
bool received = false;
double sim_time = 0., collided = 0., current = 0.;

void Receive(ConstVector3dPtr &message) {
  std::lock_guard<std::mutex> lock(result_mutex);
  sim_time = message->x();
  collided = message->y();
  current = message->z();
  received = true;
  result_changed.notify_all();
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  if (!gazebo::client::setup()) return 3;
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init(argv[1]);
  auto subscriber = node->Subscribe("~/aicar/step_result", Receive);
  auto publisher = node->Advertise<gazebo::msgs::WorldControl>("~/aicar/step");
  if (!publisher->WaitForConnection(gazebo::common::Time(15, 0))) return 4;
  std::cout << "READY" << std::endl;
  unsigned int iterations;
  while (std::cin >> iterations) {
    if (iterations == 0) break;
    gazebo::msgs::WorldControl control;
    control.set_pause(true);
    control.set_multi_step(iterations);
    {
      std::lock_guard<std::mutex> lock(result_mutex);
      received = false;
    }
    publisher->Publish(control, true);
    std::unique_lock<std::mutex> lock(result_mutex);
    if (!result_changed.wait_for(lock, std::chrono::seconds(30), [] { return received; })) return 5;
    std::cout << std::setprecision(15) << "ACK " << iterations << " "
              << sim_time << " " << collided << " " << current << std::endl;
  }
  node->Fini();
  gazebo::client::shutdown();
  return 0;
}
