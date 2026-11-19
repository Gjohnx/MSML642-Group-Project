import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3

Rectangle {
  anchors.fill: parent
  color: "#f4f4f4"
  Layout.minimumWidth: 290
  Layout.minimumHeight: 245
  ColumnLayout {
    anchors.fill: parent
    anchors.margins: 12
    Label {
      Layout.fillWidth: true
      wrapMode: Text.WordWrap
      text: ChaseButton.mode === "explore" ? "Exploring. Press Chase to pursue the NPC."
          : ChaseButton.mode === "chase" ? "Chasing the NPC."
          : ChaseButton.mode === "stop" ? "Stopped. Press Reset to start again."
          : ChaseButton.mode === "capture" ? "NPC captured. Press Reset to start again."
          : ChaseButton.mode !== "idle" ? "Episode finished. Press Reset to start again."
          : "Start the explore-then-chase demo to enable Chase."
    }
    Label {
      Layout.fillWidth: true
      wrapMode: Text.WrapAnywhere
      text: ChaseButton.mode === "idle" ? "" : ChaseButton.details
    }
    RowLayout {
      Layout.fillWidth: true
      Button {
        objectName: "chaseButton"
        Layout.fillWidth: true
        text: "Chase"
        enabled: ChaseButton.mode === "explore"
        onClicked: ChaseButton.Chase()
      }
      Button {
        objectName: "stopButton"
        Layout.fillWidth: true
        text: "Stop"
        enabled: ChaseButton.mode === "explore" || ChaseButton.mode === "chase"
        onClicked: ChaseButton.Control("stop")
      }
      Button {
        objectName: "resetButton"
        Layout.fillWidth: true
        text: "Reset"
        enabled: ChaseButton.mode !== "idle"
        onClicked: ChaseButton.Control("reset")
      }
    }
  }
}
