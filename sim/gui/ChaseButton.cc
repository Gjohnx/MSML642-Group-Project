#include "ChaseButton.hh"
#include <gz/plugin/Register.hh>
#include <QMetaObject>
#include <QJsonDocument>
#include <QJsonObject>

ChaseButton::~ChaseButton()
{
  this->node.Unsubscribe("/demo/mode");
  this->node.Unsubscribe("/demo/status");
}

void ChaseButton::LoadConfig(const tinyxml2::XMLElement *)
{
  this->publisher = this->node.Advertise<gz::msgs::StringMsg>("/demo/chase");
  this->controlPublisher = this->node.Advertise<gz::msgs::StringMsg>("/demo/control");
  this->node.Subscribe<gz::msgs::StringMsg>("/demo/mode",
      [this](const gz::msgs::StringMsg &msg)
      {
        const auto value = QString::fromStdString(msg.data());
        QMetaObject::invokeMethod(this, [this, value]()
        {
          if (this->mode != value)
          {
            this->mode = value;
            emit this->ModeChanged();
          }
        }, Qt::QueuedConnection);
      });
  this->node.Subscribe<gz::msgs::StringMsg>("/demo/status",
      [this](const gz::msgs::StringMsg &msg)
      {
        const auto object = QJsonDocument::fromJson(QByteArray::fromStdString(msg.data())).object();
        const auto value = QString("Target %1\nCollisions %2    Steps %3\nSimulation %4 s\nExplore %5\nChase %6")
            .arg(object["visible"].toBool() ? "visible" : "not visible")
            .arg(object["collisions"].toInt()).arg(object["steps"].toInt())
            .arg(object["sim_elapsed_s"].toDouble(), 0, 'f', 1)
            .arg(object["explore_policy"].toString() == "rule" ? "rule" : "Q-table")
            .arg(object["chase_policy"].toString() == "rule" ? "rule" : "Q-table");
        QMetaObject::invokeMethod(this, [this, value]()
        {
          this->details = value;
          emit this->DetailsChanged();
        }, Qt::QueuedConnection);
      });
}

void ChaseButton::Control(const QString &command)
{
  if (this->mode == "idle")
    return;
  gz::msgs::StringMsg msg;
  msg.set_data(command.toStdString());
  this->controlPublisher.Publish(msg);
}

void ChaseButton::Chase()
{
  if (this->mode != "explore")
    return;
  gz::msgs::StringMsg msg;
  msg.set_data("chase");
  this->publisher.Publish(msg);
}

GZ_ADD_PLUGIN(ChaseButton, gz::gui::Plugin)
