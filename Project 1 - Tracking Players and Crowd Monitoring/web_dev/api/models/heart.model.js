const mongoose = require('mongoose');

// Define the schema for heart rate data
const heartRateSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User', // Assuming you have a User model
    required: true,
  },
  timestamp: {
    type: Date,
    required: true,
  },
  heartRateValue: {
    type: Number,
    required: true,
  },
  // You can add more fields like 'location', 'activity', etc. as needed
});

// Create a model for the heart rate data
const HeartRate = mongoose.model('HeartRate', heartRateSchema);

module.exports = HeartRate;

