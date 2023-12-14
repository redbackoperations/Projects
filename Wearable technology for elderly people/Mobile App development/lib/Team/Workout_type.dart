import 'package:flutter/material.dart';
import 'Workout_Component.dart';

class WorkOutTypePage extends StatelessWidget
{
  const WorkOutTypePage({super.key});

  @override
  Widget build(BuildContext context)
  {
    return Material(
      type: MaterialType.transparency,
      child: Container
        (
        padding: const EdgeInsets.fromLTRB(20, 40,20, 20),
        decoration: const BoxDecoration(
            color: Color(0xff87A395)
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height:20),
            const Center(
                child: DefaultTextStyle(
                  style: TextStyle(fontWeight: FontWeight.bold,fontSize: 28, color: Colors.white),

                  child: Text('Workout Type'),
                )
            ),
            const SizedBox(height:27),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    backgroundColor: const Color(0xff317751),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),
                  ),
                  onPressed: (){
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context)=> const WorkoutComponent() )
                    );
                  },
                  child: const Text('Distance (KM)', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),),
                )
                ),
              ],
            ),
            const SizedBox(height:27),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    backgroundColor: const Color(0xff317751),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),
                  ),
                  onPressed: (){},
                  child: const Text('Time (Min)', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),),
                )
                ),
              ],
            ),
            const SizedBox(height:27),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    backgroundColor: const Color(0xff317751),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),
                  ),
                  onPressed: (){},
                  child: const Text('Interval Training)', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),),
                )
                ),
              ],
            ),
            const SizedBox(height:27),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    backgroundColor: const Color(0xff317751),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),
                  ),
                  onPressed: (){},
                  child: const Text('Track/Circuit', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),),
                )
                ),
              ],
            ),
            const SizedBox(height:27),
            Row(
              children: [
                Expanded(child: TextButton(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    backgroundColor: const Color(0xff317751),
                    textStyle: const TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),
                  ),
                  onPressed: (){},
                  child: const Text('Choose your coordinates', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 16,color: Colors.white),),
                )
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

}
