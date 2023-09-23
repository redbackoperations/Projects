import express from 'express';
import bodyParser from 'body-parser';
import session from 'express-session';
import { Strategy as LocalStrategy } from 'passport-local';
import { memoryStorage } from 'multer';
import multer from 'multer';
const storage = multer.memoryStorage();
const upload = multer({ storage });
import mongoose from 'mongoose';
import mqtt from 'mqtt';
import cors from 'cors';
import {HeartRate} from './models/heart.model.js';


import playerRoutes from './routes/playerRoutes.js';
import crowdRoutes from './routes/crowd.routes.js';


// Initialize Express
const app = express();
const port = 5004;

// MongoDB Connection URL
const mongoURL = "mongodb+srv://saksham4801be21:A0CtKk4Axeecht6C@project_a.dkyykac.mongodb.net/";

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

app.use('/players', playerRoutes);
app.use('/crowd', crowdRoutes); 

// Routes
app.get("/test", (req, res) => {
  res.send("Successfully connected to the API");
});

app.listen(port, function () {
    console.log(`\n Listening on port ${port} \n \n \t to access cmd+click on this link ====>\n \n \t \t \t \t \t http://localhost:${port}/test`);
  });
  