/**
 * Express API for managing players and crowd data.
 * @module api
 */

import express from 'express';
import bodyParser from 'body-parser';
import session from 'express-session';
import { Strategy as LocalStrategy } from 'passport-local';
import { memoryStorage } from 'multer';
import multer from 'multer';
const data = multer.memoryStorage();
const upload = multer({ data });
import mongoose from 'mongoose';
import mqtt from 'mqtt';
import cors from 'cors';
import {HeartRate} from './models/heart.model.js';
import { Storage } from '@google-cloud/storage';




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
const gcs = new Storage({
  projectId: 'sit-23t1-fit-data-pipe-ee8896e', // Replace with your GCP project ID
});

const bucketName = 'bucket_project_1_sensordata'; // Replace with your GCS bucket name
app.post('/upload', async (req, res) => {
  try {
    const jsonData = req.body;

    if (!jsonData) {
      return res.status(400).send('No JSON data provided.');
    }

    // Ensure the required fields exist in the JSON data
    const requiredFields = ['Time', 'Date', 'Fix', 'Quality', 'Location', 'Speed (knots)', 'Angle', 'Altitude', 'Satellites', 'Antenna status'];
    for (const field of requiredFields) {
      if (!(field in jsonData)) {
        return res.status(400).send(`Missing '${field}' field in JSON data.`);
      }
    }

    // Create a unique filename based on the Date and Time
    const date = jsonData.Date.replace(/\//g, '-'); // Replace slashes with dashes for valid filename
    const time = jsonData.Time.replace(/:/g, '-'); // Replace colons with dashes for valid filename
    const jsonFileName = `${date}_${time}_data.json`;

    const bucket = gcs.bucket(bucketName);
    const file = bucket.file(jsonFileName);

    await file.save(JSON.stringify(jsonData));

    res.status(200).send(`JSON data uploaded to ${bucketName}/${jsonFileName}`);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal server error');
  }
});




// Initialize MQTT client
const client = mqtt.connect('mqtt://broker.mqttdashboard.com');

/**
 * Middleware for enabling Cross-Origin Resource Sharing (CORS).
 * @function
 * @memberof module:api
 * @param {Object} req - Express request object.
 * @param {Object} res - Express response object.
 * @param {function} next - Express next middleware function.
 */
app.use(cors());

/**
 * Middleware for allowing DELETE HTTP method.
 * @function
 * @memberof module:api
 * @param {Object} req - Express request object.
 * @param {Object} res - Express response object.
 * @param {function} next - Express next middleware function.
 */
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

/**
 * Test route to check if the API is working.
 * @route GET /test
 * @group Testing
 * @returns {string} Successful connection message.
 * @throws {Error} API connection error.
 */
app.get("/test", (req, res) => {
  res.send("Successfully connected to the API");
});


  
  /**
 * Starts the Express server on the specified port.
 * @function
 * @memberof module:api
 * @param {number} port - Port number to listen on.
 */
app.listen(port, function () {
  console.log(`\n Listening on port ${port} \n \n \t to access cmd+click on this link ====>\n \n \t \t \t \t \t http://localhost:${port}/test`);
});