import 'package:flutter/material.dart';

class ContactUsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Contact Us'),
        backgroundColor: Color(0xFF302C2C), // Set the app bar color
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              'Contact Us',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Set text color to white
              ),
            ),
            SizedBox(height: 8.0),
            Text(
              'Fill out the form below to get in contact',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white, // Set text color to white
              ),
            ),
            SizedBox(height: 24.0),
            TextFormField(
              style: TextStyle(
                color: Colors.white, // Set text color to white
              ),
              decoration: InputDecoration(
                labelText: 'Email', // Label text for the email field
                labelStyle: TextStyle(
                  color: Colors.white, // Set label text color to white
                ),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
              ),
            ),
            SizedBox(height: 16.0),
            TextFormField(
              style: TextStyle(
                color: Colors.white, // Set text color to white
              ),
              decoration: InputDecoration(
                labelText: 'Subject', // Label text for the subject field
                labelStyle: TextStyle(
                  color: Colors.white, // Set label text color to white
                ),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
              ),
            ),
            SizedBox(height: 16.0),
            TextFormField(
              maxLines: 4, // Allow multiple lines for the message
              style: TextStyle(
                color: Colors.white, // Set text color to white
              ),
              decoration: InputDecoration(
                labelText: 'Message', // Label text for the message field
                labelStyle: TextStyle(
                  color: Colors.white, // Set label text color to white
                ),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(
                    color: Colors.white, // Set border color to white
                  ),
                ),
              ),
            ),
            SizedBox(height: 24.0),
            ElevatedButton(
              onPressed: () {
                // Handle form submission logic here
              },
              child: Text('Submit'),
            ),
          ],
        ),
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color
    );
  }
}
