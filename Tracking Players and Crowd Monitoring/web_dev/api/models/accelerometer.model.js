
const mongoose = require('mongoose');

const accelerometerSchema = new mongoose.Schema({
  sensorId: {
    type: String,
    required: true,
  },
  xAcceleration: {
    type: Number,
    required: true,
  },
  yAcceleration: {
    type: Number,
    required: true,
  },
  zAcceleration: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const AccelerometerData = mongoose.model('AccelerometerData', accelerometerSchema);

module.exports = AccelerometerData;
