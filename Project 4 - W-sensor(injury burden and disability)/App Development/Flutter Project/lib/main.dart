import 'package:flutter/material.dart';
import 'my_data.dart'; // Import your MyData screen
import 'login.dart';
import 'sign_up.dart';
import 'settings.dart';
import 'profile.dart';
import 'security.dart';
import 'notification.dart';
import 'friends.dart';
import 'home.dart';

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
      initialRoute: '/',
      routes: {
        '/': (context) => MyRelativeLayout(),
        '/login': (context) => MyHomePage(),
        '/signup': (context) => MySignUpPage(),
        '/home': (context) => HomeScreen(), // Use the existing HomeScreen class
        '/settings': (context) => SettingsScreen(),
        '/profile': (context) => ProfileScreen(),
        '/security': (context) => SecurityScreen(),
        '/notification': (context) => NotificationScreen(),
        '/friends': (context) => FriendsScreen(),
        '/mydata': (context) => MyData(), // Add MyData screen route here
      },
    );
  }
}

class MyRelativeLayout extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF090909),
      body: Column(
        children: <Widget>[
          SizedBox(height: 40.0), // Add space above the image
          Container(
            height: 470.0, // Set the fixed height for the image
            child: Image.asset(
              'assets/workout.jpeg',
              width: double.infinity,
              fit: BoxFit.fill,
            ),
          ),
          Spacer(), // Spacer to push the button to the bottom
          ElevatedButton(
            onPressed: () {
              Navigator.pushNamed(context, '/login');
            },
            child: Text(
              'Get Started',
              style: TextStyle(
                fontSize: 18,
              ),
            ),
          ),
        ],
      ),
    );
  }
}



class MyHomePage extends StatelessWidget {
  Widget buildLoginButton(String text, String imageName, BuildContext context) {
    return Container(
      width: double.infinity, // Set the width of the container
      child: ElevatedButton(
        onPressed: () {
          // Handle navigation to the home page
          Navigator.pushNamed(context, '/home');
        },
        style: ElevatedButton.styleFrom(
          primary: Colors.transparent,
          onPrimary: Colors.white,
          shape: RoundedRectangleBorder(
            side: BorderSide(color: Colors.white),
            borderRadius: BorderRadius.circular(5.0),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (imageName != null)
              Padding(
                padding: const EdgeInsets.only(right: 8.0),
                child: Image.asset(
                  'assets/google.jpeg', // Make sure to put your image assets in the 'assets' directory
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
      ),
    );
  }

  static Widget buildTextField(String hintText) {
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
              buildLoginButton('Login with Google', 'google', context),
              SizedBox(height: 20.0),
              buildLoginButton('Sign In with Apple ID', 'apple', context),
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
              buildLoginButton('Login', '', context),
              SizedBox(height: 16.0),
              GestureDetector(
                onTap: () {
                  // Handle navigation to the sign-up page
                  Navigator.pushNamed(context, '/signup');
                },
                child: Text(
                  "Don't have an account? Sign Up",
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
}

class MySignUpPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF302C2C),
      appBar: AppBar(
        title: Text('Sign Up'),
        backgroundColor: Color(0xFF302C2C),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: <Widget>[
            SizedBox(height: 16.0),
            Text(
              'Sign Up',
              style: TextStyle(
                fontSize: 24,
                color: Colors.white,
              ),
              textAlign: TextAlign.center,
            ),
            MyHomePage.buildTextField('Name'),
            MyHomePage.buildTextField('Email ID'),
            MyHomePage.buildTextField('New Password'),
            MyHomePage.buildTextField('Confirm Password'),
            SizedBox(height: 16.0),
            buildSignUpButton('Sign Up', context), // Pass the context here
          ],
        ),
      ),
    );
  }

  Widget buildSignUpButton(String text, BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Handle sign-up logic here

        // Navigate back to the login page
        Navigator.pop(context);
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
class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  final List<Widget> _pages = [
    // Replace with your pages
    Placeholder(), // Page 1
    Placeholder(), // Page 2
    Placeholder(), // Page 3
    Placeholder(), // Page 4
    Placeholder(), // Page 5 - Home Page Content
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color here
      body: _pages[_selectedIndex], // Show the selected page
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        // Set type to fixed for more than 3 items
        items: <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.data_usage),
            label: 'Data',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.refresh),
            label: 'Refresh',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.people),
            label: 'Friends',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }


  void _onItemTapped(int index) {
    if (index == 4) {
      // Navigate to the SettingsScreen when "Settings" is pressed
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => SettingsScreen()),
      );
    } else if (index == 3) {
      // Navigate to the FriendsScreen when "Friends" is pressed
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => FriendsScreen()),
      );

    } else if (index == 1) {
      // Navigate to the FriendsScreen when "Friends" is pressed
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => MyData()),
      );
    } else if (index == 0) {
      // Navigate to the FriendsScreen when "Friends" is pressed
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => HomeScreen1()),
      );
    }

    else {
      setState(() {
        _selectedIndex = index;
      });
    }
  }
}
