#include <Adafruit_GPS.h>
#include <MQTT.h>

#define RX_PIN 2  // RX pin connected to GPS TX
#define TX_PIN 3  // TX pin connected to GPS RX
#define GPS_BAUD 9600

// Define the MQTT parameters

// Fill in these MQTT Details
const char* mqtt_server = "**MQTT_BROKER_ADDRESS**";
const char* mqtt_username = "**MQTT_USERNAME**";
const char* mqtt_password = "**MQTT_PASSWORD**";
const char* device_id = "**DEVICE_ID**";

//Constants for different MQTT topics
const char* mqtt_topic = "iot/gps/gps_data";
const char* mqtt_response_topic = "iot/gps/gps_response";
const char* mqtt_control_topic = "iot/gps/gps_control";
const char* mqtt_republish_topic = "iot/gps/gps_republish";

// Create instances
Adafruit_GPS GPS(&Serial1); // Create an Adafruit GPS object

MQTT client(mqtt_server, 1883, callback);

String lastGpsData = ""; // Store the last GPS data received
bool publishingEnabled = true; // Variable to control publishing of location data

void setup() {
  // Initialize Serial and GPS
  Serial.begin(9600);
  Serial1.begin(GPS_BAUD);
  GPS.begin(GPS_BAUD);
  // Set GPS module to publish data according to NMEA standards
  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
  // Set the update frequency of the GPS data output in NMEA format
  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);

  // Connect to MQTT
  connectToMQTT();
}

void loop() {

  // Check MQTT connection status and update the LED
  if (client.isConnected()) {
    digitalWrite(D7, HIGH); // Turn on the LED when connected to MQTT
  } else {
    digitalWrite(D7, LOW);  // Turn off the LED when disconnected
    // Attempt to reconnect to MQTT
    connectToMQTT();
  }
  
  // Read GPS data
  char c = GPS.read();
  if (GPS.newNMEAreceived() && GPS.parse(GPS.lastNMEA())) {
    String gpsData = "DeviceID: " + String(device_id) + ", Lat: " + String(GPS.latitudeDegrees, 6) + " Lon: " + String(GPS.longitudeDegrees, 6) + ", Time: " + GPS.time;
    
    // Publish GPS data via MQTT only if publishing is enabled
    if (client.isConnected() && publishingEnabled) {
      client.publish(mqtt_topic, gpsData);
    }

    lastGpsData = gpsData; // Store the latest GPS data

    // Prints GPS data to the serial monitor for debugging
    Serial.println(gpsData);
  }
}

void connectToMQTT() {
  while (!client.connect(device_id, mqtt_username, mqtt_password)) {
    Serial.println("MQTT connection failed. Retrying..."); // Comment out if not needed for debugging
    delay(2000);
  }
  //Debugging line
  Serial.println("MQTT connected!");

  // Subscribe to the MQTT control topic
  client.subscribe(mqtt_control_topic);
}

void callback(char* topic, byte* payload, unsigned int length) {
  // Handle MQTT messages
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  // Parse the received JSON message
  StaticJsonDocument<200> doc; // Adjust the buffer size as needed
  DeserializationError error = deserializeJson(doc, message);

  if (!error) {
    // Check if the "device_id" field matches defined device ID
    const char* receivedDeviceID = doc["device_id"];
    if (strcmp(receivedDeviceID, device_id) == 0) {
      // The message is intended for this Particle device

      // Handle different message types based on the "command" field
      const char* command = doc["command"];

      if (strcmp(command, "pause_publishing") == 0) {
        // Stop publishing GPS data
        publishingEnabled = false;
        //Publish MQTT response to acknowledge GPS data publishing paused
        client.publish(mqtt_response_topic, "Publishing paused by " + String(device_id) + ".");
        //Debugging line
        Serial.println("Data publishing paused.");
      } else if (strcmp(command, "resume_publishing") == 0) {
        // Resume publishing GPS data
        publishingEnabled = true;
        //Publish MQTT response to acknowledge GPS data publishing resumed
        client.publish(mqtt_response_topic, "Publishing resumed by " + String(device_id) + ".");
        //Debugging line
        Serial.println("Data publishing resumed."); 
      } else if (strcmp(command, "republish_current") == 0) {
        // Received a current republish message
        if (client.isConnected()) {
          // Publish current GPS data in related topic
          client.publish(mqtt_republish_topic, lastGpsData);
          //Debugging line
          Serial.println("GPS data dumped.");
        }
      } else {
        // Unknown command, comment out if not needed for debugging
        Serial.println("Received an unknown command: " + String(command));
      }
    } else {
      // Ignore the message because it doesn't match the expected device_id, comment out if not needed for debugging
      Serial.println("Received MQTT message, but it doesn't match the expected device_id.");
    }
  } else {
    // Failed to parse the JSON message, comment out if not needed for debugging
    Serial.println("Failed to parse MQTT message.");
  }
}
