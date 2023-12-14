import mongoose from "mongoose";

const locationSchema = new mongoose.Schema({
  lat: String,
  log: String,
});

const playerMovementSchema = new mongoose.Schema({
  time: String,
  date: String,
  location: locationSchema,
  alt: Number,
  sat: Number,
});

const playerStatisticsSchema = new mongoose.Schema({
  gamesPlayed: {
    type: Number,
    default: 0,
  },
  goalsScored: {
    type: Number,
    default: 0,
  },
  assists: {
    type: Number,
    default: 0,
  },
  yellowCards: {
    type: Number,
    default: 0,
  },
  redCards: {
    type: Number,
    default: 0,
  },
});

const playerSchema = new mongoose.Schema({
  firstName: {
    type: String,
    required: true,
  },
  lastName: {
    type: String,
    required: true,
  },
  age: {
    type: Number,
    required: true,
  },
  team: {
    type: String,
    required: true,
  },
  position: {
    type: String,
    required: true,
  },
  dateOfBirth: {
    type: Date,
    required: true,
  },
  sport: {
    type: String,
    required: true,
  },
  heartRate: {
    type: Number,
    required: false,
  },
  movements: [playerMovementSchema],
  statistics: playerStatisticsSchema, 
});

export const Player = mongoose.model("Player", playerSchema);
