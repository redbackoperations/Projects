import 'package:flutter/material.dart';

class MySignUpPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF302C2C), // Set the background color here
      appBar: AppBar(
        title: Text('Sign Up'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: <Widget>[
              Text(
                'Sign Up',
                style: TextStyle(
                  fontSize: 24,
                  color: Colors.white,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 16.0),
              buildTextField('Name'),
              // Add other fields (Age, Gender, Height, Weight) here
              buildTextField('Email ID'),
              buildTextField('New Password'),
              buildTextField('Confirm Password'),
              SizedBox(height: 16.0),
              buildSignUpButton('Sign Up'),
              SizedBox(height: 16.0),
              GestureDetector(
                onTap: () {
                  // Handle navigation to the login page
                  Navigator.of(context).pushReplacementNamed('/login');
                },
                child: Text(
                  "Already have an account? Log In",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),
            ],
          ),
        ),
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

  Widget buildSignUpButton(String text) {
    return ElevatedButton(
      onPressed: () {
        // Handle sign-up logic here
      },
      child: Text(
        text,
        style: TextStyle(
          fontSize: 16,
        ),
      ),
    );
  }
}
