
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF302C2C), // Set the background color here
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: <Widget>[
              Text(
                'You must sign in to join',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.white,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 100.0),
              buildLoginButton('Login with Google', 'google.jpeg'),
              SizedBox(height: 20.0),
              buildLoginButton('Sign In with Apple ID', 'apple.jpeg'),
              SizedBox(height: 20.0),
              Text(
                '----------------Login----------------',
                style: TextStyle(
                  fontSize: 24,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: 20.0),
              buildTextField('Username'),
              buildTextField('Password'),
              SizedBox(height: 16.0),
              buildLoginButton('Login', 'home'),
              SizedBox(height: 16.0),
              Text(
                "Don't have Account? Sign Up",
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildLoginButton(String text, String imageName) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.transparent,
        border: Border.all(color: Colors.white),
        borderRadius: BorderRadius.circular(5.0),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          if (imageName != null)
            Padding(
              padding: const EdgeInsets.only(right: 8.0),
              child: Image.asset(
                'assets/$imageName', // Assuming your images are in the 'assets' directory
                width: 24.0,
                height: 24.0,
                color: Colors.white,
              ),
            ),
          Text(
            text,
            style: TextStyle(
              fontSize: 20.0,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget buildTextField(String hintText) {
    return TextField(
      style: TextStyle(
        color: Color(0xFFD3D3D3),
      ),
      decoration: InputDecoration(
        hintText: hintText,
        hintStyle: TextStyle(
          color: Color(0xFFD3D3D3),
        ),
      ),
    );
  }
}
