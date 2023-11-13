import 'package:flutter/material.dart';
import 'package:mobile_app/contact.dart';

class InformationScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF8F9E91),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
      ),
      body: Stack(
        children: [
          Container(
            color: Color(0xFF8F9E91),
          ),
          Positioned(
            top: 40, // Adjust the top value to position the header
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.all(16.0),
              child: Center(
                child: Text(
                  "Information",
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 120,
            left: 0,
            right: 0,
            bottom: 160, // Adjust the bottom value to create space for the button
            child: ListView(
              shrinkWrap: true,
              children: [
                Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Text(
                        "Redback Operation Mission",
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      SizedBox(height: 16),
                      Text(
                        "Redback Operations aims to turn small steps of virtuality into bigger steps of reality, making you Smarter, Fitter, and Better. Bad weather? Too much traffic? Worry not, our Smart Bike Project not only transforms your indoor cycling experience but also features an interactive VR Game and accessible Mobile App to bring the world to you.",
                        style: TextStyle(
                          fontSize: 16,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Positioned(
            left: 0,
            right: 0,
            bottom: 80,
            child: Container(
              padding: EdgeInsets.all(16.0),
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => ContactUsScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  primary: Color(0xFF758D7E), // Adjust the color here
                ),
                child: Text(
                  "Contact Us",
                  style: TextStyle(
                    fontSize: 22,
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 350,
            left: 16, // Adjust the left value to control the width
            right: 16, // Adjust the right value to control the width
            child: Container(
              padding: EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12), // Add rounded corners
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Latest News",
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Here are all the news from the team",
                    style: TextStyle(
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            ),
          ),

        ],
      ),
    );
  }
}
