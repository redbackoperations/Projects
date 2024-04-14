#include <ArduinoBLE.h>

// Function Declarations
void initializeBLE();
void discoverAndConnect();
void printData(const unsigned char data[], int length);

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Step 1: Starting Service");
  initializeBLE();
}

void loop() {
  discoverAndConnect();
  BLE.poll();  // Poll for BLE events and communication updates.
}

void initializeBLE() {
  Serial.println("Step 2: Initializing BLE");

  if (!BLE.begin()) {
    Serial.println("Step 2: BLE failed!");
    while (1);
  }

  Serial.println("Step 3: BLE Successful");
  BLE.scanForUuid("180D");
}

void discoverAndConnect() {
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    Serial.println("Step 4: Scanning Device");
    Serial.print("Peripheral Name: ");
    Serial.println(peripheral.localName());

    if (peripheral.localName() == "TICKR 0A5B") {
      Serial.println("Step 5: Device Found");
      Serial.println("Step 6: Connecting ...");
      BLE.stopScan();

      if (peripheral.connect()) {
        Serial.println("Step 7: Connected To Sensor");

        Serial.println("Step 8: Discovering attributes ...");
        if (peripheral.discoverAttributes()) {
          Serial.println("Step 9: Attributes Discovered Successfully");

          BLEService service180D = peripheral.service("180D");
          if (service180D) {
            BLECharacteristic characteristic2A37 = service180D.characteristic("2A37");
            if (characteristic2A37) {
              characteristic2A37.subscribe();
              Serial.println("Step 10: Connected to Data Stream of Sensor. Started Fetching Data ....");
              while (true) {
                peripheral.poll();
                if (characteristic2A37.valueUpdated()) {
                  printData(characteristic2A37.value(), characteristic2A37.valueLength());
                }
              }
            } else {
              Serial.println("Step 10: Characteristic 2A37 not found");
            }
          } else {
            Serial.println("Step 10: Service 180D not found");
          }
        } else {
          Serial.println("Step 9: Discovering attributes failed");
          peripheral.disconnect();
          return;
        }
      } else {
        Serial.println("Step 7: Connection Failed");
        return;
      }
    } else {
      Serial.println("Step 6: Sensor Not Found");
    }
  }
}

void printData(const unsigned char data[], int length) {
  for (int i = 0; i < length; i++) {
    if (i == 1) {
      unsigned char b = data[i];
      Serial.print("Heart Rate Value (BPM) = ");
      Serial.println(b);
    }
  }
}
