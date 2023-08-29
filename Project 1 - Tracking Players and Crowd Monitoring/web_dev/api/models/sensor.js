const mongoose = require('mongoose');


const sensorSchema = new mongoose.Schema({
  temperature: Number,
  humidity: Number,
  // Add more fields as needed
});

// Create a Mongoose model based on the schema
const Sensor = mongoose.model('Sensor', sensorSchema);

module.exports = Sensor;
  