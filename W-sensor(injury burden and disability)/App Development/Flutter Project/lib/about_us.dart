import 'package:flutter/material.dart';

class AboutUsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('About Us'),
        backgroundColor: Color(0xFF302C2C), // Set the app bar color
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              'About Us',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Set text color to white
              ),
            ),
            SizedBox(height: 8.0),
            Text(
              'Welcome to our app!',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white, // Set text color to white
              ),
            ),
            SizedBox(height: 16.0),
            Text(
              'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '
                  'Sed euismod vulputate dui a feugiat. Proin ut urna vitae '
                  'arcu venenatis venenatis. Sed lacinia, ante eu egestas '
                  'vehicula, ipsum velit hendrerit arcu, nec aliquet mauris '
                  'nunc nec justo.',
              style: TextStyle(
                fontSize: 16,
                color: Colors.white, // Set text color to white
              ),
            ),
          ],
        ),
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color
    );
  }
}
