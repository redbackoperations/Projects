import 'package:flutter/material.dart';

class AccountScreen extends StatelessWidget {
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
              'Account',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),
          buildOption(context, 'Profile', Icons.account_circle, '/profile'),
          buildOption(context, 'Security', Icons.security, '/security'),
          buildOption(context, 'Notifications', Icons.notifications, '/notifications'),
        ],
      ),
      backgroundColor: Color(0xFF302C2C),
    );
  }

  Widget buildOption(BuildContext context, String title, IconData icon, String routeName) {
    return Container(
      color: Color(0xFF302C2C),
      child: ListTile(
        leading: Icon(
          icon,
          color: Colors.white,
        ),
        title: Text(
          title,
          style: TextStyle(
            color: Colors.white,
          ),
        ),
        trailing: Icon(
          Icons.arrow_forward_ios,
          color: Colors.white,
        ),
        onTap: () {
          Navigator.pushNamed(context, routeName); // Navigate to the specified route
        },
      ),
    );
  }
}
