const mongoose = require('mongoose');

const locationSchema = new mongoose.Schema({
  sensorId: {
    type: String,
    required: true,
  },
  latitude: {
    type: Number,
    required: true,
  },
  longitude: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const LocationData = mongoose.model('LocationData', locationSchema);

module.exports = LocationData;
