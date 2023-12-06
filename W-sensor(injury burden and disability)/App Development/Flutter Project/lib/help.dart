import 'package:flutter/material.dart';

class HelpScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Help'),
        backgroundColor: Color(0xFF302C2C), // Set the app bar color
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              'Frequently Asked Questions',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Set text color to white
              ),
            ),
          ),
          FAQItem(
            question: 'How do I reset my password?',
            answer: 'To reset your password, go to the login screen and click on the "Forgot Password" link. Follow the instructions to reset your password.',
          ),
          FAQItem(
            question: 'How can I contact customer support?',
            answer: 'You can contact our customer support team by sending an email to support@example.com or by calling +1 (123) 456-7890.',
          ),
          // Add more FAQ items here as needed
        ],
      ),
      backgroundColor: Color(0xFF302C2C), // Set the background color
    );
  }
}

class FAQItem extends StatelessWidget {
  final String question;
  final String answer;

  FAQItem({required this.question, required this.answer});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            question,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.white, // Set text color to white
            ),
          ),
          SizedBox(height: 8.0),
          Text(
            answer,
            style: TextStyle(
              fontSize: 16,
              color: Colors.white, // Set text color to white
            ),
          ),
        ],
      ),
    );
  }
}
