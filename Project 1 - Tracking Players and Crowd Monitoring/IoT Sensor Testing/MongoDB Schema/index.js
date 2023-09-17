const bodyParser = require('body-parser');
const mongoose = require('mongoose');
mongoose.connect('mongodb://127.0.0.1:27017/gps-data');

const sensor = require('./models/gps-data');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

const serial_port = new SerialPort({
    path: 'COM8',
    baudRate: 9600
})  

const parser = serial_port.pipe(new ReadlineParser({ delimiter: '\r\n' }));


let sum = 0;
parser.on('data', async (data) => {

    const array_data = data.split(" ");

    // 1 10:58:46 14:9:2023 3750.6430S 14506.7968E 107.90 5

    const sensor_id = array_data[0];
    const time = array_data[1];
    const date = array_data[2];
    const lat = array_data[3];
    const log = array_data[4];
    const alt = array_data[5];
    const sat = array_data[6];

    console.log("Sensor: " + sensor_id);
    console.log("Time: " + time);
    console.log("Date: " + date);
    console.log("Lat: " + lat);
    console.log("Log: " + log);
    console.log("Alt: " + alt);
    console.log("Satallite: " + sat);

    try {
        const abc = await sensor.findOne({ 'device_id': sensor_id }).exec();

        if (abc == null) {
            var document = {
                'device_id': parseInt(sensor_id),
                'sensorData': [
                    {
                        'time': time,
                        'date': date,
                        'location': {
                            'lat': lat,
                            'log': log,
                        },
                        alt: parseFloat(alt),
                        sat: parseInt(sat)
                    }
                ]
            }
            const create = await sensor.create(document);
            console.log(create);
        }

        if (abc != null) {
            console.log(abc);
        }

    } catch (error) {
        console.log(error);
    }

    try {
        const xyz = await sensor.updateOne({ 'device_id': sensor_id }, {
            $push: {
                'sensorData': [
                    {
                        'time': time,
                        'date': date,
                        'location': {
                            'lat': lat,
                            'log': log,
                        },
                        alt: parseFloat(alt),
                        sat: parseInt(sat)
                    }
                ]
            }
        })

        console.log(xyz);

    } catch (error) {
        console.log(error);
    }

    console.log(data);
});