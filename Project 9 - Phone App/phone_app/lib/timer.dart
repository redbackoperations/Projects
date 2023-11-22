import 'package:flutter/material.dart';
import 'dart:async';
import 'gps.dart';

class TimerPage extends StatefulWidget {
  final String title;

  TimerPage({required this.title});

  @override
  _TimerPageState createState() => _TimerPageState();
}

class _TimerPageState extends State<TimerPage> {
  int _seconds = 0;
  late Timer _timer;
  bool _isPaused = false;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(Duration(seconds: 1), _incrementTimer);
  }

  void _incrementTimer(Timer timer) {
    if (!_isPaused) {
      setState(() {
        _seconds++;
      });
    }
  }

  String _formatTime(int seconds) {
    int minutes = seconds ~/ 60;
    int remainingSeconds = seconds % 60;
    String minutesStr = minutes.toString().padLeft(2, '0');
    String secondsStr = remainingSeconds.toString().padLeft(2, '0');
    return '$minutesStr:$secondsStr';
  }

  void _togglePause() {
    setState(() {
      _isPaused = !_isPaused;
    });
  }

  void _stopTimer() {
    _timer.cancel();
    Navigator.of(context)
        .pop(); // Close the screen and go back to the previous screen
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: const Color(0xFF8F9E91),
        child: ListView(
          padding: EdgeInsets.all(16.0),
          children: [
            Text(
              widget.title,
              style: TextStyle(
                fontSize: 40,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 20),
            Container(
              width: 150,
              height: 150,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.transparent,
                border: Border.all(
                  color: Colors.black,
                  width: 2.0,
                ),
              ),
              child: Center(
                child: Text(
                  _formatTime(_seconds),
                  style: TextStyle(
                    fontSize: 40,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _togglePause,
              style: ElevatedButton.styleFrom(
                primary: Colors.black,
                minimumSize: Size(160, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text(_isPaused ? 'Resume' : 'Pause'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _stopTimer,
              style: ElevatedButton.styleFrom(
                primary: Colors.red,
                minimumSize: Size(160, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text('Stop'),
            ),
            SizedBox(height: 50),
            MapPage(),
          ],
        ),
      ),
    );
  }
}
