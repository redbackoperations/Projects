const mongoose = require('mongoose');

module.exports = mongoose.model('gps-sensor', new mongoose.Schema({
    device_id: Number,
    sensorData: [
        {
            time: String,
            date: String,
            location: {
                lat: {
                    type: String,
                    required: true
                }, log: {
                    type: String,
                    required: true
                }
            },
            alt: Number,
            sat: Number
        }
    ]
}, { collection: 'gps-data' }));