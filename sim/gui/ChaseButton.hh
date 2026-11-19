#pragma once

#include <gz/gui/Plugin.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>

class ChaseButton : public gz::gui::Plugin
{
  Q_OBJECT
  Q_PROPERTY(QString mode READ Mode NOTIFY ModeChanged)

  public: void LoadConfig(const tinyxml2::XMLElement *) override;
  public: QString Mode() const { return this->mode; }
  public slots: void Chase();
  signals: void ModeChanged();

  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: QString mode = "idle";
};
