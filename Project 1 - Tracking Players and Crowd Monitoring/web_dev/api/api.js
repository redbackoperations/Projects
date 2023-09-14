const express = require("express");
const bodyParser = require("body-parser");
const session = require("express-session");
const LocalStrategy = require("passport-local").Strategy;
const multer = require('multer');
const storage = multer.memoryStorage();
const upload = multer({ storage });
const mongoose = require("mongoose");
const mqtt = require('mqtt');
const cors = require('cors');
const data = require('./models/sensor');
const Temp = require('./models/temprature.model');
const HeartRate = require('./models/heart.model');
const AccelerometerData = require('./models/accelerometer.model');
const BluetoothData = require('./models/bluetooth.model');
const LocationData = require('./models/gps.model');

//This is just for demonstration purposes //Testing//


// Initialize Express
const app = express();
const port = 5004;

// MongoDB Connection URL
const mongoURL = "mongodb+srv://saksham4801be21:A0CtKk4Axeecht6C@db1.dkyykac.mongodb.net/";

// Connect to MongoDB
mongoose.connect(mongoURL, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log("Connected to MongoDB");
}).catch(err => {
  console.error("Error connecting to MongoDB:", err);
});

// Initialize MQTT client
const client = mqtt.connect('mqtt://broker.mqttdashboard.com');

// Middleware
app.use(cors());

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Methods', 'DELETE');
  next();
});

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Routes
app.get("/test", (req, res) => {
  res.send("Successfully connected to the API");
});

app.listen(port, function () {
    console.log(`\n Listening on port ${port} \n \n \t to access cmd+click on this link ====>\n \n \t \t \t \t \t http://localhost:${port}/test`);
  });
  