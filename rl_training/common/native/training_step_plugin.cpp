// Exact paused steps and physical contact ground truth, independent of the
// asynchronous ROS sensors. Privileged policies may consume the returned pose.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <iomanip>

namespace gazebo {
class TrainingStepPlugin : public WorldPlugin {
 public:
  ~TrainingStepPlugin() override {
    if (worker_.joinable()) worker_.join();
  }
  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override {
    world_ = world;
    entity_ = sdf->Get<std::string>("entity", "my_car").first;
    std::istringstream ignored(sdf->Get<std::string>("ignored_models", "ground_plane").first);
    for (std::string name; ignored >> name;) ignored_.insert(name);
    manager_ = world_->Physics()->GetContactManager();
    manager_->SetNeverDropContacts(true);
    node_.reset(new transport::Node());
    node_->Init(world_->Name());
    result_ = node_->Advertise<msgs::GzString>("~/aicar/step_result");
    update_ = event::Events::ConnectWorldUpdateEnd(std::bind(&TrainingStepPlugin::OnUpdate, this));
    request_ = node_->Subscribe("~/aicar/step", &TrainingStepPlugin::OnStep, this);
  }

 private:
  void OnUpdate() {
    bool hit = false;
    for (unsigned i = 0; i < manager_->GetContactCount(); ++i) {
      auto *contact = manager_->GetContact(i);
      const auto a = contact->collision1->GetModel()->GetName();
      const auto b = contact->collision2->GetModel()->GetName();
      if ((a == entity_) == (b == entity_)) continue;
      if (ignored_.count(a == entity_ ? b : a)) continue;
      hit = true;
    }
    std::lock_guard<std::mutex> lock(contacts_mutex_);
    current_ = hit;
    collided_ = collided_ || hit;
  }

  void OnStep(ConstWorldControlPtr &request) {
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    if (!request->has_multi_step() || request->multi_step() == 0) return;
    if (worker_.joinable()) worker_.join();
    const auto iterations = request->multi_step();
    // World::Step blocks; never block Gazebo's transport callback thread,
    // which must keep delivering the sensor/physics transport messages.
    worker_ = std::thread([this, iterations] { RunStep(iterations); });
  }

  void RunStep(unsigned int iterations) {
    {
      std::lock_guard<std::mutex> lock(contacts_mutex_);
      collided_ = false;
    }
    world_->SetPaused(true);
    world_->Step(iterations);
    msgs::GzString result;
    std::ostringstream values;
    const auto car = world_->ModelByName(entity_);
    if (!car) {
      result.set_data("ERROR missing vehicle");
      result_->Publish(result, true);
      return;
    }
    // The physics world is paused here: contacts, pose and velocity belong to
    // exactly the same completed step, not asynchronous/stale ROS messages.
    const auto pose = car->WorldPose();
    const auto velocity = pose.Rot().RotateVectorReverse(car->WorldLinearVel());
    {
      std::lock_guard<std::mutex> lock(contacts_mutex_);
      values << std::setprecision(17) << world_->SimTime().Double() << " "
             << collided_ << " " << current_ << " "
             << pose.Pos().X() << " " << pose.Pos().Y() << " "
             << pose.Rot().Yaw() << " " << velocity.X() << " "
             << car->WorldAngularVel().Z() << " " << pose.Pos().Z();
    }
    result.set_data(values.str());
    result_->WaitForConnection(common::Time(5, 0));
    result_->Publish(result, true);
  }

  physics::WorldPtr world_;
  physics::ContactManager *manager_ = nullptr;
  transport::NodePtr node_;
  transport::PublisherPtr result_;
  transport::SubscriberPtr request_;
  event::ConnectionPtr update_;
  std::mutex contacts_mutex_, command_mutex_;
  std::thread worker_;
  std::string entity_;
  std::set<std::string> ignored_;
  bool collided_ = false, current_ = false;
};
GZ_REGISTER_WORLD_PLUGIN(TrainingStepPlugin)
}  // namespace gazebo
