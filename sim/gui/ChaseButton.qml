import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3

Rectangle {
  anchors.fill: parent
  color: "transparent"
  Layout.minimumWidth: 250
  Layout.minimumHeight: 130
  ColumnLayout {
    anchors.fill: parent
    anchors.margins: 12
    Label {
      Layout.fillWidth: true
      wrapMode: Text.WordWrap
      text: ChaseButton.mode === "explore" ? "Exploring. Press Chase to pursue the NPC."
          : ChaseButton.mode === "chase" ? "Chasing the NPC."
          : "Start the explore-then-chase demo to enable Chase."
    }
    Button {
      objectName: "chaseButton"
      Layout.fillWidth: true
      text: "Chase"
      enabled: ChaseButton.mode === "explore"
      onClicked: ChaseButton.Chase()
    }
  }
}
