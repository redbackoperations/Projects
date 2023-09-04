# MQTT Schema for initial GPS testing over MQTT

This serves to provide an explanation to the structure of MQTT messages published by the Particle and what messages are expected.

## Main published message

The device will publish its main message whenever the GPS location of the device changes based on readings from the GPS sensor.

The changes will be published in the **iot/gps/gps_data** topic and will have a payload consisting on the device ID as specified on the device, the latitude and longitude readings, as well as the time of the reading.

A sample of this is seen below:

DeviceID: RedbackGPS0001, Lat: 37.123456, Lon: -122.987654, Time: 13:45:30

## Control messages

The device will subscribe to **iot/gps/gps_control** and will listen for a set of predefined control messages. The messages are inspected and only actioned if the device ID in the message matches that of the device.

### Control message structure

Control messages sent to the device should be formatted as a JSON message and should be formatted as in the below example:
{
  "device_id": "TARGET_DEVICE_ID",
  "command": "DESIRED_COMMAND"
}

### Toggle Publishing Command

There is an included command that can be sent to the device to turn the publishing of GPS location updates on or off as needed. These commands should be sent on the **iot/gps/gps_control** topic and should be formatted as described above. The commands seen below are used to achieve the desired effect:

To pause publishing make use of the command **pause_publishing**

To resume publishing make use of the command **resume_publishing**

### Republish current GPS data

Should it be necessary, you can request that the device republishes its most recent position.

This is done by sending the command **republish_current** to the device on the **iot/gps/gps_control** topic. The device will respond with the republished data on the **iot/gps/gps_republish** topic.

## Control message responses

Besides the republish command having a specific topic on which the response is published, all other commands will receive an acknowledgement response on the topic **iot/gps/gps_response**

The **pause_publishing** and **resume_publishing** will trigger the responses "Publishing paused by **device_id**." and "Publishing resumed by **device_id**." respectively.
