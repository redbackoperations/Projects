const express = require("express");
const app = express();
const port = 3001;
const base = `${__dirname}/public`;
const bodyParser = require("body-parser");
const session = require("express-session");
// const passport = require("passport");
// const { initialize, isAuthenticated } = require('./passportconfig');
const cors = require('cors');

// Middleware for parsing JSON in request body
app.use(bodyParser.json());

// Enable CORS
app.use(cors());

// Serve static files
app.use(express.static("public"));

app.get("/", function (req,res)
{
    res.send("succesfully connected to the server");
});

// Start the server
app.listen(port, function () {
    console.log(`\n Listening on port {3001} \n \n \t to access cmd+click on this link ====>\n \n \t \t \t \t \t http://localhost:${port}/`);
  });
  