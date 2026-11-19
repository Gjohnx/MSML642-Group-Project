#pragma once

#include <gz/gui/Plugin.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>

class ChaseButton : public gz::gui::Plugin
{
  Q_OBJECT
  Q_PROPERTY(QString mode READ Mode NOTIFY ModeChanged)
  Q_PROPERTY(QString details READ Details NOTIFY DetailsChanged)

  public: void LoadConfig(const tinyxml2::XMLElement *) override;
  public: QString Mode() const { return this->mode; }
  public: QString Details() const { return this->details; }
  public: ~ChaseButton() override;
  public slots: void Chase();
  public slots: void Control(const QString &command);
  signals: void ModeChanged();
  signals: void DetailsChanged();

  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: gz::transport::Node::Publisher controlPublisher;
  private: QString mode = "idle";
  private: QString details;
};
