import 'package:flutter/material.dart';

class SettingPage extends StatelessWidget
{
  const SettingPage({super.key});

  @override
  Widget build(BuildContext context)
  {
    return Material(
        type: MaterialType.transparency,
        child: Container(
          padding: const EdgeInsets.fromLTRB(10, 70,10, 10),
          decoration: const BoxDecoration(
              color: Color(0xff87A395)
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Text('Setting', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 20,color: Colors.black),),
              )
            ],
          ),
        )
    );
  }
}
