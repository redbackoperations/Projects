import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:smart_bike_mobile_app/pages/workout_record_item.dart';
import 'package:smart_bike_mobile_app/utils/firebase_auth_util.dart';

class WorkoutRecord extends StatefulWidget {
  const WorkoutRecord(
      {super.key, required this.snapshot, required this.listLength, required this.callback});
  final int listLength;
  final Iterable<Object?> snapshot;
  final Function callback;
  @override
  State<WorkoutRecord> createState() => _WorkoutRecordState();
}

class _WorkoutRecordState extends State<WorkoutRecord> {
  // Firebase auth user data
  FirebaseAuthUtil firebase = FirebaseAuthUtil();

  // Firestore collection
  final db = FirebaseFirestore.instance;

  @override
  Widget build(BuildContext context) {
    return (widget.listLength == 0)
        ? Center(
            child: Column(
              children: [
                const SizedBox(height: 30,),
                const Text('No record, start your first workout session now.',
                  style: TextStyle(color: Color(0xFFE3E3E3)),
                ),
                const SizedBox(height: 30,),
                ElevatedButton(
                    onPressed: (){widget.callback(0);},
                    style: ButtonStyle(
                      backgroundColor: MaterialStateColor.resolveWith((states) => const Color(0xFF370E4A)),
                      shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                        RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(5.0),
                        ),
                      ),
                      minimumSize: MaterialStateProperty.all<Size>(
                        const Size(100, 50),
                      ),
                    ),
                    child: const Text('Start Workout',
                        style: TextStyle(
                            color: Color(0xFFE3E3E3)
                        )
                    )
                ),
              ],
            ),
          )
        : ListView.separated(
            itemCount: widget.listLength,
            itemBuilder: (_, int index) {
              return ListTile(
                  onTap: () async {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => WorkoutRecordItem(
                              workoutId: index,
                              data: widget.snapshot.elementAt(index) as Map),
                        ));
                  },
                  title: Text(
                    'Workout Session #${index + 1}',
                    style: const TextStyle(
                      color: Color(0xFFE3E3E3),
                    ),
                  ));
            },
            separatorBuilder: (BuildContext context, int index) =>
                const Divider(
              thickness: 1,
              color: Color(0x33370E4A),
            ),
          );
  }
}
