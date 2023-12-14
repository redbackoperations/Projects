const mongoose = require('mongoose');

const bluetoothSchema = new mongoose.Schema({
  deviceName: {
    type: String,
    required: true,
  },
  macAddress: {
    type: String,
    required: true,
  },
  proximity: {
    type: String,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const BluetoothData = mongoose.model('BluetoothData', bluetoothSchema);

module.exports = BluetoothData;
