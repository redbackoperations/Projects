import 'package:flutter/material.dart';

class ContactUsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xff87A395),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
      ),
      body: Stack(
        children: [
          Container(
            color: Color(0xFF8F9E91),
          ),
          Positioned(
            top: 20,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.all(16.0),
              child: const Center(
                child: Text(
                  "Contact Us",
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF380E4A),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 80,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.all(16.0),
              child: Center(
                child: Text(
                  "Fill out the form to get in contact",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 180,
            left: 16,
            right: 16,
            child: Column(
              children: [
                Container(
                  margin: EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: TextField(
                    decoration: InputDecoration(
                      hintText: "Subject",
                      contentPadding: EdgeInsets.all(16),
                    ),
                  ),
                ),
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: TextField(
                    maxLines: 4, // Allow multiple lines for the message
                    decoration: InputDecoration(
                      hintText: "Message",
                      contentPadding: EdgeInsets.all(16),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Positioned(
            bottom: 130,
            left: 16,
            right: 16,
            child: ElevatedButton(
              onPressed: () {
                // Add your send message logic here
              },
              style: ElevatedButton.styleFrom(
                primary: Color(0xFF758D7E),
              ),
              child: Text(
                "Send",
                style: TextStyle(
                  fontSize: 22,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
