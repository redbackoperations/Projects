import 'package:flutter/material.dart';

class CyclingScreen extends StatefulWidget {
  @override
  _CyclingScreenState createState() => _CyclingScreenState();
}

class _CyclingScreenState extends State<CyclingScreen> {
  int _seconds = 0;
  bool _isCycling = false;
  bool _isPaused = false;
  late ValueNotifier<bool> _stopwatchNotifier;

  @override
  void initState() {
    super.initState();
    _stopwatchNotifier = ValueNotifier<bool>(false);
  }

  void _startTimer() {
    if (!_isCycling) {
      _isCycling = true;
      _stopwatchNotifier.value = true;
      _runTimer();
    }
  }

  void _pauseTimer() {
    if (_isCycling) {
      _isPaused = !_isPaused;
      if (_isPaused) {
        _stopwatchNotifier.value = false;
      } else {
        _stopwatchNotifier.value = true;
        _runTimer();
      }
    }
  }

  void _stopTimer() {
    _isCycling = false;
    _isPaused = false;
    _seconds = 0;
    _stopwatchNotifier.value = false;
  }

  void _runTimer() {
    Future.delayed(Duration(seconds: 1), () {
      if (_isCycling && !_isPaused) {
        setState(() {
          _seconds++;
        });
        _runTimer();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Cycling Timer'),
        backgroundColor: Color(0xFF090909),
        centerTitle: true,
      ),
      backgroundColor: Color(0xFF090909), // Set the background color here
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Cycling: $_seconds seconds',
              style: TextStyle(fontSize: 24, color: Colors.white), // Set the text color to white
            ),
            SizedBox(height: 20),
            ValueListenableBuilder<bool>(
              valueListenable: _stopwatchNotifier,
              builder: (context, stopwatchCycling, child) {
                return AnimatedSwitcher(
                  duration: Duration(milliseconds: 500),
                  child: stopwatchCycling
                      ? Image.asset(
                    'assets/running_person.gif', // Replace with your image asset
                    key: ValueKey<bool>(stopwatchCycling),
                    height: 200,
                  )
                      : Container(),
                );
              },
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _startTimer,
                  child: Text('Start'),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: _pauseTimer,
                  child: Text(_isPaused ? 'Resume' : 'Pause'),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: _stopTimer,
                  child: Text('Stop'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: CyclingScreen(),
  ));
}
