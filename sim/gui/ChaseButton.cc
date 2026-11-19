#include "ChaseButton.hh"
#include <gz/plugin/Register.hh>
#include <QMetaObject>

void ChaseButton::LoadConfig(const tinyxml2::XMLElement *)
{
  this->publisher = this->node.Advertise<gz::msgs::StringMsg>("/demo/chase");
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
