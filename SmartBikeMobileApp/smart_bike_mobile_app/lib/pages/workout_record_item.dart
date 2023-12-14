import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class WorkoutRecordItem extends StatelessWidget {
  const WorkoutRecordItem(
      {super.key, required this.workoutId, required this.data});
  final int workoutId;
  final Map data;

  String convertDate() {
    Timestamp t = data['timeStamp'];
    DateTime d = t.toDate();
    return d.toString();
  }

  String getAverageSpeed() {
    num speed = data['distance'] / data['duration'];
    return speed.toStringAsFixed(2);
  }

  String getAverageRPM() {
    num rpm = data['totalRPM'] / data['duration'];
    return rpm.toStringAsFixed(1);
  }

  String getAveragePower() {
    num power = data['totalPower'] / data['duration'];
    return power.toStringAsFixed(1);
  }

  String getDistance() {
    num distance = data['distance'];
    return distance.toStringAsFixed(2);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF6E2D51),
              Color(0xFFE97462),
              Color.fromRGBO(55, 14, 74, 0.94),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: Center(
          child: Column(
            children: [
              const SizedBox(
                height: 50,
              ),
              Text(
                'Workout Session #${workoutId + 1}',
                style: const TextStyle(
                  color: Color(0xFFE3E3E3),
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                ),
              ),
              const SizedBox(
                height: 30,
              ),
              Text(
                'Time: ${convertDate().substring(0, 19)}',
                style: const TextStyle(color: Color(0xFFE3E3E3)),
              ),
              const SizedBox(
                height: 20,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 1,
                    child: Center(
                      child: Column(
                        children: [
                          const Text(
                            'Duration',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 10,
                          ),
                          Text(
                            '${data['duration']} s',
                            style: const TextStyle(
                              color: Color(0xFFE3E3E3),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(
                height: 20,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 2,
                    child: Center(
                      child: Column(
                        children: [
                          const Text(
                            'Average Speed',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 10,
                          ),
                          Text(
                            '${getAverageSpeed()} m/s',
                            style: const TextStyle(
                              color: Color(0xFFE3E3E3),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Expanded(
                    flex: 2,
                    child: Center(
                      child: Column(
                        children: [
                          const Text(
                            'Average RPM',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 10,
                          ),
                          Text(
                            '${getAverageRPM()} RPM',
                            style: const TextStyle(
                              color: Color(0xFFE3E3E3),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(
                height: 20,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 2,
                    child: Center(
                      child: Column(
                        children: [
                          const Text(
                            'Average Power',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 10,
                          ),
                          Text(
                            '${getAveragePower()} Watts',
                            style: const TextStyle(
                              color: Color(0xFFE3E3E3),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Expanded(
                    flex: 2,
                    child: Center(
                      child: Column(
                        children: [
                          const Text(
                            'Distance',
                            style: TextStyle(
                              color: Color(0xFFE3E3E3),
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(
                            height: 10,
                          ),
                          Text(
                            '${getDistance()} m',
                            style: const TextStyle(
                              color: Color(0xFFE3E3E3),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(
                height: 50,
              ),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(
                    context,
                  );
                },
                style: ButtonStyle(
                  backgroundColor: MaterialStateColor.resolveWith(
                      (states) => const Color(0xFF370E4A)),
                  shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                    RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(5.0),
                    ),
                  ),
                  minimumSize: MaterialStateProperty.all<Size>(
                    const Size(100, 50),
                  ),
                ),
                child: const Text(
                  'Back',
                  style: TextStyle(
                    color: Color(0xFFE3E3E3),
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
