import mongoose from 'mongoose';

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
});



export const HeartRate = mongoose.model('HeartRate', heartRateSchema);

