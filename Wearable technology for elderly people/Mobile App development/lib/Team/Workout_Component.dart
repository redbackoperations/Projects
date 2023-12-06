import 'package:flutter/material.dart';
import 'Workout_type.dart';

class WorkoutComponent extends StatefulWidget
{
  const WorkoutComponent({super.key});
  @override
  State<WorkoutComponent> createState()=>_WorkoutComponent();
}
class _WorkoutComponent extends State<WorkoutComponent>
{

  @override
  Widget build(BuildContext context)
  {
    return Material(
        type: MaterialType.transparency,
        child: Container(
          padding: const EdgeInsets.fromLTRB(10, 50,10, 10),
          decoration: const BoxDecoration(
              color: Color(0xff87A395)
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Text('Workout Type', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 30,color: Colors.white),),
              ),
              Center(
                child: Text('Name/type of workout', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 20,color: Colors.white),),
              ),
              SizedBox(height:17),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  WorkoutCard(measure: "Speed",value: "15.0 KM/H",),
                  WorkoutCard(measure: "Cadence", value: "60 RPM",),
                ],
              ),
              SizedBox(height:17),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  WorkoutCard(measure: "Distance",value: "3.8 km",),
                  WorkoutCard(measure: "Oxygen Saturation", value: "96%",),
                ],
              ),
              SizedBox(height:17),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  WorkoutCard(measure: "Heart Rate",value: "140 BPM",),
                  WorkoutCard(measure: "Temperature", value: "36.7 C",)
                ],
              ),
            ],
          ),
        )
    );
  }

}

class WorkoutCard extends StatefulWidget
{
  const WorkoutCard({super.key, required this.measure, required this.value});
  final String measure;
  final String value;
  @override
  State<WorkoutCard> createState()=>_WorkoutCard();
}
class _WorkoutCard extends State<WorkoutCard>
{

  @override
  Widget build(BuildContext context)
  {
    return SizedBox(
      width: 180,
      height: 100,
      child: Stack(
        children: <Widget>[
          Container(
            padding: const EdgeInsets.fromLTRB(10, 50, 10, 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              color: const Color(0xff317751),
            ),
            child: Center(
              child: Text(widget.value, style: const TextStyle(fontWeight: FontWeight.bold,fontSize: 25,color: Colors.white),),
            ),
          ),
          Positioned.fill(
            child: Align(
              alignment: Alignment.topCenter,
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  color: Colors.white,
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(widget.measure, style: const TextStyle(fontWeight: FontWeight.bold,fontSize: 20,color: Colors.black),)
                  ],
                ),
              ),
            ),

          ),
        ],
      ),
    );
  }
}
