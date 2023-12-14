const mongoose = require('mongoose');

// Define the Temperature Schema
const temperatureSchema = new mongoose.Schema({
  sensorId: {
    type: String,
    required: true,
  },
  temperature: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

// Create a Temperature model using the schema
const Temperature = mongoose.model('Temperature', temperatureSchema);

module.exports = Temperature;
