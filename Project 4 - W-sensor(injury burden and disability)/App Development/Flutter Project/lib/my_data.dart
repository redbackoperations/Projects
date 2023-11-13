import 'package:flutter/material.dart';
import 'package:untitled4/Walking.dart';
import 'running.dart'; // Import the file where RunningScreen is defined
import 'package:untitled4/Walking.dart';
import 'package:untitled4/Jogging.dart';
import 'package:untitled4/Cycling.dart';

class MyData extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Data'),
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color here
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            DataOption(title: 'Walking'),
            DataOption(title: 'Running'),
            DataOption(title: 'Jogging'),
            DataOption(title: 'Cycling'),
          ],
        ),
      ),
    );
  }
}

class DataOption extends StatelessWidget {
  final String title;

  DataOption({required this.title});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: ElevatedButton(
        onPressed: () {
          if (title == 'Running') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => RunningScreen()),
            );
          } else if (title == 'Walking') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => WalkingScreen()),
            );
          } else if (title == 'Jogging') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => JoggingScreen()),
            );
          } else if (title == 'Cycling') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => CyclingScreen()),
            );
          }
        },
        style: ElevatedButton.styleFrom(
          primary: Colors.blue, // Set the button color
          minimumSize: Size(200, 50), // Set the button size
        ),
        child: Text(
          title,
          style: TextStyle(
            fontSize: 18,
            color: Colors.white,
          ),
        ),
      ),
    );
  }
}
