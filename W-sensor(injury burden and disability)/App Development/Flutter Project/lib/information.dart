import 'package:flutter/material.dart';
import 'about_us.dart'; // Import your About Us page
import 'contact_us.dart'; // Import your Contact Us page

class InformationScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Info'),
        backgroundColor: Color(0xFF302C2C), // Set the app bar color
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              'Information',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Set text color to white
              ),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => AboutUsScreen()), // Navigate to About Us page
                  );
                },
                child: Text('About Us'),
              ),
              SizedBox(width: 16.0),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => ContactUsScreen()), // Navigate to Contact Us page
                  );
                },
                child: Text('Contact Us'),
              ),
            ],
          ),
          SizedBox(height: 24.0),
          Text(
            'Welcome to our app!',
            style: TextStyle(
              fontSize: 18,
              color: Colors.white, // Set text color to white
            ),
          ),
          SizedBox(height: 16.0),
          Text(
            'About Us: Lorem ipsum dolor sit amet, consectetur adipiscing elit....',
            style: TextStyle(
              fontSize: 16,
              color: Colors.white, // Set text color to white
            ),
          ),
          SizedBox(height: 16.0),
          Text(
            'Contact Us: Email: example@email.com\nPhone: +1 (123) 456-7890',
            style: TextStyle(
              fontSize: 16,
              color: Colors.white, // Set text color to white
            ),
          ),
        ],
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color
    );
  }
}
