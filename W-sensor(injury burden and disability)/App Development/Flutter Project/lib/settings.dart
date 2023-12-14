import 'package:flutter/material.dart';
import 'account.dart'; // Import your account page
import 'information.dart'; // Import your information page
import 'help.dart'; // Import your help page

class SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(height: 16), // Add some space before the heading
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              'Settings',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Set text color to white
              ),
              textAlign: TextAlign.center, // Center-align the heading
            ),
          ),
          buildOption(context, 'Account', Icons.account_circle), // Pass context to buildOption
          buildOption(context, 'Information', Icons.info), // Pass context to buildOption
          buildOption(context, 'Help', Icons.help), // Pass context to buildOption
        ],
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color here
    );
  }


  Widget buildOption(BuildContext context, String title, IconData icon) {
    return Container(
      color: Color(0xFF302C2C), // Set the background color here
      child: ListTile(
        leading: Icon(
          icon,
          color: Colors.white, // Set icon color to white
        ),
        title: Text(
          title,
          style: TextStyle(
            color: Colors.white, // Set text color to white
          ),
        ),
        trailing: Icon(
          Icons.arrow_forward_ios,
          color: Colors.white, // Set icon color to white
        ),
        onTap: () {
          if (title == 'Account') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => AccountScreen()), // Replace AccountScreen() with your actual account page class.
            );
          } else if (title == 'Information') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => InformationScreen()), // Replace InformationScreen() with your actual information page class.
            );
          } else if (title == 'Help') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => HelpScreen()), // Navigate to HelpScreen when "Help" is tapped
            );
          } else {
            // Handle other options here
          }
        },
      ),
    );
  }
}
